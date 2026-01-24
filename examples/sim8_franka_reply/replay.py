# file: playback_franka.py
"""
回放脚本（Franka 双臂 + 球形障碍）：
- 载入与 `bullet_robot_ros_franka.py` 相同的场景（地面 + 9 球）
- 从 .npz 读取：t(N), q1(N,7), q2(N,7), obs_names(M), obs_pos(N,M,3), obs_orn(N,M,4)
- 支持 --file/--speed/--loop；未指定 --file 时取 logs 目录下最新 .npz
- 夹爪按占位值 [0.01,0.01] 拼接到 18 维
"""
from __future__ import annotations

import os
import glob
import time
import argparse
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pybullet as pyb
import pybullet_data
import pyb_utils

CAM = dict(distance=1.2, yaw=50, pitch=-40, target=[0.0, 0.0, 0.1])


def _candidate_npz(log_dir: str = './logs') -> pathlib.Path:
    paths = sorted(glob.glob(str(pathlib.Path(log_dir) / '*.npz')))
    if not paths:
        raise FileNotFoundError(f'No .npz found under {log_dir!r}')
    return pathlib.Path(paths[-1])


def _load_npz(path: pathlib.Path):
    data = np.load(path, allow_pickle=True)
    t = np.asarray(data['t'], dtype=float)
    q1 = np.asarray(data['q1'], dtype=float)
    q2 = np.asarray(data['q2'], dtype=float)
    obs_names = list(np.asarray(data['obs_names']).tolist())
    obs_pos = np.asarray(data['obs_pos'], dtype=float)  # (N,M,3)
    obs_orn = np.asarray(data['obs_orn'], dtype=float)  # (N,M,4)

    if t.ndim != 1:
        raise ValueError('t must be (N,)')
    n = t.shape[0]
    if q1.shape != (n, 7) or q2.shape != (n, 7):
        raise ValueError('q1/q2 expect shape (N,7)')
    if obs_pos.shape[:2] != (n, len(obs_names)) or obs_orn.shape[:2] != (n, len(obs_names)):
        raise ValueError('obs_pos/obs_orn shape mismatch with obs_names length')
    return t, q1, q2, obs_names, obs_pos, obs_orn


def _setup_env():
    cid = pyb.connect(pyb.GUI)
    pyb.setTimeStep(1.0 / 125.0, physicsClientId=cid)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)

    ground_id = pyb.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True, physicsClientId=cid)
    robot_id = pyb.loadURDF(
        'model/dual_panda_model/dual_panda_r9_urdf.urdf',
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=cid,
    )
    robot = pyb_utils.Robot(robot_id, client_id=cid)

    # 9 spheres at the same canonical positions as in the logger env
    sphere_defs = {
        'sphere1': {'pose': [-0.125, 0.0, 1.0],   'radius': 0.1},
        'sphere2': {'pose': [-0.125, 0.125, 1.0], 'radius': 0.1},
        'sphere3': {'pose': [-0.125, -0.125, 1.0],'radius': 0.1},
        'sphere4': {'pose': [-0.125, 0.0, 0.7],   'radius': 0.1},
        'sphere5': {'pose': [-0.125, 0.125, 0.7], 'radius': 0.1},
        'sphere6': {'pose': [-0.125, -0.125, 0.7],'radius': 0.1},
        'sphere7': {'pose': [0.35, 0.0, 0.6],     'radius': 0.1},
        'sphere8': {'pose': [0.35, 0.125, 0.6],   'radius': 0.1},
        'sphere9': {'pose': [0.35, -0.125, 0.6],  'radius': 0.1},
    }

    def _spawn_sphere(radius: float, position: List[float]) -> int:
        col_id = pyb.createCollisionShape(pyb.GEOM_SPHERE, radius=radius, physicsClientId=cid)
        vis_id = pyb.createVisualShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1, 0, 0, 0.9],
            physicsClientId=cid,
        )
        body_id = pyb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=position,
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=cid,
        )
        return body_id

    obstacles: Dict[str, int] = {'ground': ground_id}
    for name, cfg in sphere_defs.items():
        obstacles[name] = _spawn_sphere(cfg['radius'], cfg['pose'])

    pyb.resetDebugVisualizerCamera(
        cameraDistance=CAM['distance'],
        cameraYaw=CAM['yaw'],
        cameraPitch=CAM['pitch'],
        cameraTargetPosition=CAM['target'],
    )
    return cid, robot, obstacles


def playback(file: str | None, speed: float, loop: bool) -> None:
    path = pathlib.Path(file) if file else _candidate_npz('./logs')
    print(f"[playback] loading: {path}")
    t, q1, q2, obs_names, obs_pos, obs_orn = _load_npz(path)
    cid, robot, obstacles = _setup_env()

    # 名称到 body id 的映射（日志顺序需匹配）
    idx_of_name = {name: i for i, name in enumerate(sorted([k for k in obstacles.keys() if k != 'ground']))}
    obs_body_ids: List[int] = []
    for name in obs_names:
        if name not in idx_of_name:
            raise KeyError(f'obstacle {name!r} not found in playback env')
        # obstacles 按 sorted 生成，ground 除外
        # 找到第 i 个有效 body：需+1跳过 ground
        sorted_keys = ['ground'] + sorted([k for k in obstacles.keys() if k != 'ground'])
        obs_body_ids.append(obstacles[name])

    n = t.shape[0]
    try:
        while True:
            start = time.monotonic()
            t0 = float(t[0])
            for i in range(n):
                target = (float(t[i]) - t0) / max(speed, 1e-6)
                while True:
                    now = time.monotonic() - start
                    if now >= target:
                        break
                    time.sleep(min(0.002, max(0.0, target - now)))

                gr1 = np.array([0.01, 0.01])
                gr2 = np.array([0.01, 0.01])
                q = np.concatenate([q1[i], gr1, q2[i], gr2])
                robot.reset_joint_configuration(q)

                # 回放障碍
                for j, bid in enumerate(obs_body_ids):
                    pyb.resetBasePositionAndOrientation(
                        bid,
                        obs_pos[i, j, :].tolist(),
                        obs_orn[i, j, :].tolist(),
                        physicsClientId=cid,
                    )
            if loop:
                continue
            break
    except KeyboardInterrupt:
        pass
    finally:
        if pyb.isConnected(cid):
            pyb.disconnect(cid)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Playback dual Franka with obstacles (PyBullet)')
    p.add_argument('--file', type=str, default="./logs/munihei.npz", help='path to .npz; default: latest under ./logs')
    p.add_argument('--speed', type=float, default=1.0, help='playback speed (1=realtime)')
    p.add_argument('--loop', action='store_true', help='loop playback')
    return p


def main(argv=None):
    args = _build_argparser().parse_args(argv)
    os.environ.setdefault('ROS_DOMAIN_ID', '16')
    playback(args.file, args.speed, args.loop)


if __name__ == '__main__':
    main()
