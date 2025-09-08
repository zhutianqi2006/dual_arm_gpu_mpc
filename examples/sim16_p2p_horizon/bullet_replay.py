# filepath: playback_joint_trajectory.py
"""
离线回放脚本：读取 .npz 轨迹文件，按时间戳回放到 PyBullet。
默认寻找 ./logs 目录下最新的 npz；也可 --file 指定。
支持 --speed 调速，--loop 循环播放。
"""
from __future__ import annotations

# Python stdlib
import os
import time
import glob
import pathlib
import argparse
from typing import Optional

# Third-party
import numpy as np
import pybullet as pyb
import pybullet_data
import pyb_utils

CAM = dict(distance=1.0, yaw=89, pitch=-28, target=[-0.0, 0.0, 0.0])


def _load_latest_npz(log_dir: str) -> pathlib.Path:
    paths = sorted(glob.glob(os.path.join(log_dir, '*.npz')))
    if not paths:
        raise FileNotFoundError(f'No .npz found under {log_dir!r}')
    return pathlib.Path(paths[-1])


def _load_traj(path: pathlib.Path):
    data = np.load(path)
    t = data['t'].astype(float)
    ur3 = data['ur3'].astype(float)
    ur3e = data['ur3e'].astype(float)
    dt = float(data['dt']) if 'dt' in data else float(np.median(np.diff(t)))
    if t.ndim != 1 or ur3.shape[0] != t.shape[0] or ur3e.shape[0] != t.shape[0]:
        raise ValueError('Invalid trajectory shapes: expect t:(N,), ur3:(N,6), ur3e:(N,6)')
    return t, ur3, ur3e, dt


def _setup_pybullet():
    cid = pyb.connect(pyb.GUI)
    pyb.setTimeStep(1.0 / 60.0, physicsClientId=cid)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    ground_id = pyb.loadURDF('plane.urdf', [0, 0, 0], useFixedBase=True, physicsClientId=cid)
    robot_id = pyb.loadURDF('model/dual_arm_model/dual_arm_notray_model.urdf', [0, 0, 0], useFixedBase=True, physicsClientId=cid)
    robot = pyb_utils.Robot(robot_id, client_id=cid)
    pyb.resetDebugVisualizerCamera(
        cameraDistance=CAM['distance'], cameraYaw=CAM['yaw'], cameraPitch=CAM['pitch'], cameraTargetPosition=CAM['target']
    )
    return cid, robot


def playback(file: Optional[str], speed: float, loop: bool) -> None:
    path = pathlib.Path(file) if file else _load_latest_npz('./logs')
    print(f"Loading: {path}")
    t, ur3, ur3e, _ = _load_traj(path)
    cid, robot = _setup_pybullet()

    try:
        while True:
            start = time.monotonic()
            t0 = float(t[0])
            for i in range(len(t)):
                # 目标时间点（加速/减速）
                target = (float(t[i]) - t0) / max(speed, 1e-6)
                # 忙等到达目标时间（避免 drift 使用绝对时钟）
                while True:
                    now = time.monotonic() - start
                    if now >= target:
                        break
                    # 小睡保持 CPU 友好
                    time.sleep(min(0.002, max(0.0, target - now)))
                dual = np.concatenate([ur3[i], ur3e[i]])
                robot.reset_joint_configuration(dual)
            if loop:
                continue
            break
    except KeyboardInterrupt:
        pass
    finally:
        if pyb.isConnected(cid):
            pyb.disconnect(cid)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Playback dual-arm trajectory (.npz) in PyBullet')
    parser.add_argument('--file', type=str, default='./logs/p2p_h20.npz', help='Path to .npz; default: latest under ./logs')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed (1.0 = realtime)')
    parser.add_argument('--loop', action='store_true', help='Loop playback')
    args = parser.parse_args(argv)

    playback(args.file, args.speed, args.loop)


if __name__ == '__main__':
    main()
