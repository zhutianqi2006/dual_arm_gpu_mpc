
# filepath: bullet_replay.py
"""
Playback for recorded trajectories (supports Franka + dynamic obstacles, and legacy UR3 dual-arm).

Usage examples:
  python bullet_replay.py --file ./logs/franka_with_obs.npz --speed 1.0
  python bullet_replay.py --loop --speed 0.5        # slow-motion loop
  python bullet_replay.py                            # auto-pick the latest .npz under ./logs

NPZ schemas supported:
- Franka (this script):
    t:(N,), franka1:(N,7), franka2:(N,7), obstacles:(N,K,7), obs_names:(K,), dt:float
- Legacy UR3 (for compatibility):
    t:(N,), ur3:(N,6), ur3e:(N,6)
"""
from __future__ import annotations

import os
import time
import glob
import json
import argparse
import pathlib
from typing import Optional, Tuple, Literal

import numpy as np
import pybullet as pyb
import pybullet_data
import pyb_utils

CAM = dict(distance=1.55, yaw=40, pitch=-40, target=[0.0, 0.0, 0.08])


# ------------------ IO ------------------
def _candidate_npz(log_dir: str = "./logs") -> pathlib.Path:
    paths = sorted(glob.glob(str(pathlib.Path(log_dir) / "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz found under {log_dir!r}")
    return pathlib.Path(paths[-1])


def _detect_schema(path: pathlib.Path) -> Literal['franka', 'ur3']:
    with np.load(path) as data:
        keys = set(data.keys())
    if {'franka1', 'franka2', 't'}.issubset(keys):
        return 'franka'
    if {'ur3', 'ur3e', 't'}.issubset(keys):
        return 'ur3'
    raise ValueError(f"Unrecognized schema in {path}")


# ------------------ Environments ------------------
def _setup_env_franka():
    cid = pyb.connect(pyb.GUI)
    pyb.setTimeStep(1.0 / 125.0, physicsClientId=cid)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)

    ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=cid)
    robot_id = pyb.loadURDF(
        "model/dual_panda_model/dual_panda_urdf.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=cid,
    )
    robot = pyb_utils.Robot(robot_id, client_id=cid)

    cube4_id = pyb.loadURDF("model/plane/plane.urdf", [0.50, 0.0, 0.85], useFixedBase=True, physicsClientId=cid)
    cube5_id = pyb.loadURDF("model/plane/plane.urdf", [0.50, 0.0, 0.85], useFixedBase=True, physicsClientId=cid)

    pyb.resetDebugVisualizerCamera(
        cameraDistance=CAM['distance'],
        cameraYaw=CAM['yaw'],
        cameraPitch=CAM['pitch'],
        cameraTargetPosition=CAM['target'],
    )
    obstacles = {"cube4": cube4_id, "cube5": cube5_id}
    return cid, robot, obstacles


def _setup_env_ur3():
    cid = pyb.connect(pyb.GUI)
    pyb.setTimeStep(1.0 / 60.0, physicsClientId=cid)
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)

    ground_id = pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=cid)
    robot_id = pyb.loadURDF(
        "model/dual_arm_model/dual_arm_model.urdf",
        [0, 0, 0],
        useFixedBase=True,
        physicsClientId=cid,
    )
    robot = pyb_utils.Robot(robot_id, client_id=cid)
    cube2_id = pyb.loadURDF("model/plane/thine_plane.urdf", [0.46, 0.0, 0.006], useFixedBase=True, physicsClientId=cid)
    cube3_id = pyb.loadURDF("model/plane/thine_plane.urdf", [0.46, 0.0, 0.256], useFixedBase=True, physicsClientId=cid)
    cube4_id = pyb.loadURDF("model/plane/thine_plane.urdf", [0.46, 0.0, 0.506], useFixedBase=True, physicsClientId=cid)

    pyb.resetDebugVisualizerCamera(
        cameraDistance=1.55, cameraYaw=50, cameraPitch=-38, cameraTargetPosition=[-0.0, 0.0, 0.0]
    )
    obstacles = {"cube2": cube2_id, "cube3": cube3_id, "cube4": cube4_id}
    return cid, robot, obstacles


# ------------------ Playback loops ------------------
def _playback_franka(path: pathlib.Path, speed: float, loop: bool) -> None:
    with np.load(path) as data:
        t = data['t'].astype(float)
        f1 = data['franka1'].astype(float)
        f2 = data['franka2'].astype(float)
        obs = data.get('obstacles')  # (N,K,7) or None
        obs_names = data.get('obs_names')  # optional

    if f1.shape[1] != 7 or f2.shape[1] != 7:
        raise ValueError("Invalid Franka joint shapes")

    cid, robot, obstacles = _setup_env_franka()

    # if obstacles trajectory present, set initial placement
    if obs is not None and obs.shape[0] == t.shape[0]:
        for j, name in enumerate(list(obs_names) if obs_names is not None else ['cube4', 'cube5'][: obs.shape[1]]):
            if name in obstacles:
                pos = obs[0, j, 0:3]
                orn = obs[0, j, 3:7]
                pyb.resetBasePositionAndOrientation(obstacles[name], pos, orn, physicsClientId=cid)

    # fixed grippers (visual only)
    g = np.array([0.01, 0.01])

    try:
        while True:
            start = time.monotonic()
            t0 = float(t[0])
            for i in range(len(t)):
                target = (float(t[i]) - t0) / max(speed, 1e-6)
                while True:
                    now = time.monotonic() - start
                    if now >= target:
                        break
                    time.sleep(min(0.002, max(0.0, target - now)))
                # joints
                robot.reset_joint_configuration(np.concatenate([f1[i], g, f2[i], g]))
                # obstacles
                if obs is not None and i < obs.shape[0]:
                    for j, name in enumerate(list(obs_names) if obs_names is not None else ['cube4', 'cube5'][: obs.shape[1]]):
                        if name in obstacles:
                            pos = obs[i, j, 0:3]
                            orn = obs[i, j, 3:7]
                            pyb.resetBasePositionAndOrientation(obstacles[name], pos, orn, physicsClientId=cid)
            if loop:
                continue
            break
    except KeyboardInterrupt:
        pass
    finally:
        if pyb.isConnected(cid):
            pyb.disconnect(cid)


def _playback_ur3(path: pathlib.Path, speed: float, loop: bool) -> None:
    with np.load(path) as data:
        t = data['t'].astype(float)
        ur3 = data['ur3'].astype(float)
        ur3e = data['ur3e'].astype(float)
    if ur3.shape[1] != 6 or ur3e.shape[1] != 6:
        raise ValueError("Invalid UR3 joint shapes")

    cid, robot, _ = _setup_env_ur3()

    try:
        while True:
            start = time.monotonic()
            t0 = float(t[0])
            for i in range(len(t)):
                target = (float(t[i]) - t0) / max(speed, 1e-6)
                while True:
                    now = time.monotonic() - start
                    if now >= target:
                        break
                    time.sleep(min(0.002, max(0.0, target - now)))
                robot.reset_joint_configuration(np.concatenate([ur3[i], ur3e[i]]))
            if loop:
                continue
            break
    except KeyboardInterrupt:
        pass
    finally:
        if pyb.isConnected(cid):
            pyb.disconnect(cid)


# ------------------ CLI ------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Playback recorded trajectories (Franka + obstacles, or legacy UR3)")
    parser.add_argument("--file", type=str, default='./logs/dy_h20.npz', help="Path to .npz; default: latest under ./logs")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0=realtime)")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args(argv)

    os.environ.setdefault("ROS_DOMAIN_ID", "16")

    path = pathlib.Path(args.file) if args.file else _candidate_npz("./logs")
    print(f"Loading: {path}")
    schema = _detect_schema(path)

    if schema == 'franka':
        _playback_franka(path, args.speed, args.loop)
    elif schema == 'ur3':
        _playback_ur3(path, args.speed, args.loop)
    else:
        raise RuntimeError("Unsupported schema")


if __name__ == "__main__":
    main()