# filepath: playback_with_static_obstacles.py
"""
回放脚本（含静态障碍环境）：
- 载入与 `bullet_robot_ros.py` 相同的场景（地面 + 3 片静态薄板 + 双臂 URDF）。
- 从 `.npz` 读取 t、ur3(6)、ur3e(6)，按时间戳回放。
- `--file` 未指定时优先使用 `./logs/st_h5.npz`，否则取 `./logs` 下最新 `.npz`。
- 支持 `--speed` 调速与 `--loop` 循环。
"""
from __future__ import annotations

import os
import time
import glob
import argparse
import pathlib
from typing import Optional, Tuple

import numpy as np
import pybullet as pyb
import pybullet_data
import pyb_utils

CAM = dict(distance=1.0, yaw=51, pitch=-32, target=[-0.0, 0.0, 0.0])


def _candidate_npz(log_dir: str = "./logs") -> pathlib.Path:
    prefer = pathlib.Path(log_dir) / "st_h5.npz"
    if prefer.exists():
        return prefer
    paths = sorted(glob.glob(str(pathlib.Path(log_dir) / "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz found under {log_dir!r}")
    return pathlib.Path(paths[-1])


def _load_npz(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    t = data["t"].astype(float)
    ur3 = data["ur3"].astype(float)
    ur3e = data["ur3e"].astype(float)
    if t.ndim != 1 or ur3.shape != (t.shape[0], 6) or ur3e.shape != (t.shape[0], 6):
        raise ValueError("Invalid trajectory shapes: expect t:(N,), ur3:(N,6), ur3e:(N,6)")
    return t, ur3, ur3e


def _setup_env():
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
        cameraDistance=CAM["distance"],
        cameraYaw=CAM["yaw"],
        cameraPitch=CAM["pitch"],
        cameraTargetPosition=CAM["target"],
    )

    obstacles = {"ground": ground_id, "cube2": cube2_id, "cube3": cube3_id, "cube4": cube4_id}
    return cid, robot, obstacles


def playback(file: Optional[str], speed: float, loop: bool) -> None:
    path = pathlib.Path(file) if file else _candidate_npz("./logs")
    print(f"Loading: {path}")
    t, ur3, ur3e = _load_npz(path)
    cid, robot, _ = _setup_env()

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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Playback dual-arm trajectory with static obstacles (PyBullet)")
    parser.add_argument("--file", type=str, default='./logs/cem_kmeans.npz', help="Path to .npz; default: ./logs/st_h5.npz or latest under ./logs")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (1.0=realtime)")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args(argv)

    os.environ.setdefault("ROS_DOMAIN_ID", "16")
    playback(args.file, args.speed, args.loop)


if __name__ == "__main__":
    main()
