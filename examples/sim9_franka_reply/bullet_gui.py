#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: bullet_gui_franka.py
"""
Franka Dual-Arm GUI (No ROS2) + DQ Robotics
==========================================
- 左侧 PyBullet 滑条直接控制两条 7-DOF Franka Panda 机械臂的关节角（忽略夹爪）。
- 后台持续输出：
    * 两臂关节位置（rad）
    * 相对位姿（双四元数，8 维）
    * 绝对位姿（双四元数，8 维）
- 依赖：
    pip install pybullet numpy pyyaml dqrobotics
- 期待的本地文件：
    - two_franka_r9.yaml  （DH、上下限、基座/末端等）
    - utils/config_module.py
    - model/dual_panda_model/dual_panda_r9_urdf.urdf
- 可选参数：见 `-h`

起始角策略（与原版一致）
----------------------
1) 命令行指定：
   --r1-init "0,0,0,0,0,0,0"  --r2-init "0,0,0,0,0,0,0"，配合 --init-deg 则为度。
2) 策略：--init-mode {mid,zero,min,max,yaml}
   - yaml: 若 YAML 含 robot1_q_init / robot2_q_init 则读取
3) 校验：默认越界自动截断；加 --strict-init 改为越界报错。

与 UR 版差异
------------
- URDF 默认切换到 Franka 双臂模型；
- 仅选择 **REVOLUTE** 关节映射，自动跳过夹爪 PRISMATIC 指；
- 去掉硬编码初始角，完全由策略/命令行/YAML 决定。
"""

from __future__ import annotations
import os
import sys
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pybullet as p
import pybullet_data

# 用户侧的 ConfigModule
try:
    from utils.config_module import ConfigModule
except ImportError:
    print("[ERROR] 找不到 utils/config_module.py，请将脚本与其放在同一目录或可导入路径。", file=sys.stderr)
    raise

# DQ Robotics
from dqrobotics import DQ, vec8
from dqrobotics.robot_modeling import (
    DQ_SerialManipulatorDH,
    DQ_SerialManipulatorMDH,
    DQ_CooperativeDualTaskSpace,
)

DEFAULT_DT = 1.0 / 240.0


@dataclass
class DualArmContext:
    robot1: object
    robot2: object
    dual: object
    r1_dof: int
    r2_dof: int


# ----------------------------- DQ helpers -----------------------------

def _as_unit_dq(x):
    dq = DQ(x)
    try:
        dq = dq.normalize()
    except Exception:
        pass
    return dq


def build_dual_arm_from_config(config: ConfigModule) -> DualArmContext:
    """从 YAML/ConfigModule 构建两臂与 cooperative dual-arm 模型，并返回上下文以保活。"""
    # ---- Robot 1 ----
    robot1_dh = np.array(config.robot1_dh_mat)
    if robot1_dh.shape[0] == 5:
        dh1 = robot1_dh
    elif robot1_dh.shape[1] == 5:
        dh1 = robot1_dh.T
    else:
        raise ValueError(f"[Robot1] DH 维度异常：{robot1_dh.shape}，应为 5×n 或 n×5")

    robot1_q_num = int(config.robot1_q_num)
    if dh1.shape[1] != robot1_q_num:
        raise ValueError(f"[Robot1] DH 列数({dh1.shape[1]}) != q_num({robot1_q_num})，请检查 YAML")

    robot1_base = _as_unit_dq(config.robot1_base)
    robot1_ee = _as_unit_dq(config.robot1_effector)

    if int(config.robot1_dh_type) == 1:
        robot1 = DQ_SerialManipulatorMDH(dh1)
    else:
        robot1 = DQ_SerialManipulatorDH(dh1)
    robot1.set_base_frame(robot1_base)
    robot1.set_reference_frame(robot1_base)
    robot1.set_effector(robot1_ee)

    # ---- Robot 2 ----
    robot2_dh = np.array(config.robot2_dh_mat)
    if robot2_dh.shape[0] == 5:
        dh2 = robot2_dh
    elif robot2_dh.shape[1] == 5:
        dh2 = robot2_dh.T
    else:
        raise ValueError(f"[Robot2] DH 维度异常：{robot2_dh.shape}，应为 5×n 或 n×5")

    robot2_q_num = int(config.robot2_q_num)
    if dh2.shape[1] != robot2_q_num:
        raise ValueError(f"[Robot2] DH 列数({dh2.shape[1]}) != q_num({robot2_q_num})，请检查 YAML")

    robot2_base = _as_unit_dq(config.robot2_base)
    robot2_ee = _as_unit_dq(config.robot2_effector)

    if int(config.robot2_dh_type) == 1:
        robot2 = DQ_SerialManipulatorMDH(dh2)
    else:
        robot2 = DQ_SerialManipulatorDH(dh2)
    robot2.set_base_frame(robot2_base)
    robot2.set_reference_frame(robot2_base)
    robot2.set_effector(robot2_ee)

    dual = DQ_CooperativeDualTaskSpace(robot1, robot2)
    return DualArmContext(robot1, robot2, dual, robot1_q_num, robot2_q_num)


# ----------------------------- PyBullet helpers -----------------------------

def init_pybullet(urdf_path: str, dt: float, gravity: float, realtime: bool) -> int:
    cid = p.connect(p.GUI)
    if cid < 0:
        raise RuntimeError("无法连接 PyBullet GUI。")

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(dt)
    p.setRealTimeSimulation(1 if realtime else 0)
    p.setGravity(0, 0, float(gravity))

    p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)

    # 查找 URDF
    candidate_paths = [
        urdf_path,
        os.path.join(os.getcwd(), urdf_path),
        os.path.join(os.path.dirname(__file__), urdf_path),
    ]
    model_uid = None
    for path in candidate_paths:
        if os.path.exists(path):
            model_uid = p.loadURDF(path, [0, 0, 0], useFixedBase=True)
            break
    if model_uid is None:
        raise FileNotFoundError(f"找不到 URDF：{candidate_paths}，请设置 --urdf 正确路径。")

    # 相机
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=50,
        cameraPitch=-40,
        cameraTargetPosition=[0.0, 0.0, 0.1],
    )
    return model_uid


def collect_arm_revolute_joints(body_uid: int) -> Tuple[List[int], List[str]]:
    """仅收集 URDF 中 **旋转关节**（忽略夹爪的 PRISMATIC）。返回 (ids, names)。"""
    n = p.getNumJoints(body_uid)
    ids, names = [], []
    for j in range(n):
        ji = p.getJointInfo(body_uid, j)
        joint_type = ji[2]
        if joint_type == p.JOINT_REVOLUTE:  # why: Franka 7 DOF are revolute; grippers are usually prismatic
            ids.append(j)
            names.append(ji[1].decode("utf-8", errors="ignore"))
    return ids, names


def reset_joint_configuration(body_uid: int, joint_ids: List[int], q: np.ndarray) -> None:
    for idx, qval in zip(joint_ids, q):
        p.resetJointState(body_uid, idx, float(qval))


def add_joint_sliders(title: str, qmin: np.ndarray, qmax: np.ndarray, q0: np.ndarray, prefix: str) -> List[int]:
    p.addUserDebugText(title, [-0.5, 0, 1.2], textSize=1.8, lifeTime=0)
    sliders = []
    for i in range(len(q0)):
        s = p.addUserDebugParameter(f"{prefix}{i+1:02d}", float(qmin[i]), float(qmax[i]), float(q0[i]))
        sliders.append(s)
    return sliders


# ----------------------------- 起始角处理 -----------------------------

def _parse_csv_floats(csv: str, expected_len: int) -> np.ndarray:
    parts = [x.strip() for x in csv.split(',') if x.strip() != '']
    vals = np.array([float(x) for x in parts], dtype=float)
    if len(vals) != expected_len:
        raise ValueError(f"起始角数量应为 {expected_len}，当前 {len(vals)}: {csv}")
    return vals


def _choose_init(
    args: argparse.Namespace,
    config: ConfigModule,
    r1_q_min: np.ndarray,
    r1_q_max: np.ndarray,
    r2_q_min: np.ndarray,
    r2_q_max: np.ndarray,
    r1_dof: int,
    r2_dof: int,
) -> (np.ndarray, np.ndarray):
    def mid(a, b):
        return (np.array(a, dtype=float) + np.array(b, dtype=float)) * 0.5

    # R1
    if args.r1_init:
        r1 = _parse_csv_floats(args.r1_init, r1_dof)
    elif args.init_mode == 'yaml' and hasattr(config, 'robot1_q_init'):
        r1 = np.array(getattr(config, 'robot1_q_init'), dtype=float)
        if len(r1) != r1_dof:
            raise ValueError(f"YAML robot1_q_init 长度应为 {r1_dof}，当前 {len(r1)}")
    elif args.init_mode == 'zero':
        r1 = np.zeros(r1_dof, dtype=float)
    elif args.init_mode == 'min':
        r1 = np.array(r1_q_min, dtype=float)
    elif args.init_mode == 'max':
        r1 = np.array(r1_q_max, dtype=float)
    else:
        r1 = mid(r1_q_min, r1_q_max)

    # R2
    if args.r2_init:
        r2 = _parse_csv_floats(args.r2_init, r2_dof)
    elif args.init_mode == 'yaml' and hasattr(config, 'robot2_q_init'):
        r2 = np.array(getattr(config, 'robot2_q_init'), dtype=float)
        if len(r2) != r2_dof:
            raise ValueError(f"YAML robot2_q_init 长度应为 {r2_dof}，当前 {len(r2)}")
    elif args.init_mode == 'zero':
        r2 = np.zeros(r2_dof, dtype=float)
    elif args.init_mode == 'min':
        r2 = np.array(r2_q_min, dtype=float)
    elif args.init_mode == 'max':
        r2 = np.array(r2_q_max, dtype=float)
    else:
        r2 = mid(r2_q_min, r2_q_max)

    if args.init_deg:
        r1 = np.deg2rad(r1)
        r2 = np.deg2rad(r2)

    def limit_or_raise(q, qmin, qmax, name: str):
        if args.strict_init:
            if np.any(q < qmin) or np.any(q > qmax):
                idxs = np.where((q < qmin) | (q > qmax))[0]
                raise ValueError(f"{name} 初值越界于关节 {idxs.tolist()}；使用 --strict-init 以外模式会自动截断。")
            return q.astype(float)
        clipped = np.clip(q, qmin, qmax)
        if not np.allclose(clipped, q):
            print(f"[WARN] {name} 初值超出上下限，已截断到有效范围。", file=sys.stderr)
        return clipped.astype(float)

    r1 = limit_or_raise(r1, r1_q_min, r1_q_max, 'R1')
    r2 = limit_or_raise(r2, r2_q_min, r2_q_max, 'R2')
    return r1, r2


# ----------------------------- CLI -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Franka Dual-Arm PyBullet GUI (no ROS2) + DQ Robotics")
    ap.add_argument("--yaml", type=str, default="two_franka_r9.yaml", help="配置 YAML 路径（DH/limit/base/ee）")
    ap.add_argument("--urdf", type=str, default="model/dual_panda_model/dual_panda_r9_urdf.urdf", help="URDF 相对/绝对路径")
    ap.add_argument("--print-hz", type=float, default=10.0, help="控制台打印频率 Hz")
    ap.add_argument("--dt", type=float, default=DEFAULT_DT, help="仿真步长秒 (默认 1/240)")
    ap.add_argument("--gravity", type=float, default=-9.81, help="重力加速度 (默认 -9.81)")
    ap.add_argument("--realtime", action="store_true", help="使用实时仿真 (RealTimeSimulation)")
    ap.add_argument("--no-overlay", action="store_true", help="关闭窗口内 overlay 文本")

    # 起始角
    ap.add_argument("--r1-init", type=str, default=None, help="Robot1 起始角，逗号分隔，例如 '0,0,0,0,0,0,0'。默认弧度，配合 --init-deg 则为度")
    ap.add_argument("--r2-init", type=str, default=None, help="Robot2 起始角，逗号分隔，例如 '0,0,0,0,0,0,0'。默认弧度，配合 --init-deg 则为度")
    ap.add_argument("--init-deg", action="store_true", help="将 --r1-init / --r2-init 解释为度制")
    ap.add_argument("--init-mode", choices=["mid", "zero", "min", "max", "yaml"], default="mid", help="当未提供 --r1-init/--r2-init 时的起始角策略")
    ap.add_argument("--strict-init", action="store_true", help="严格校验起始角长度与范围，若越界即报错（默认自动截断）")

    return ap.parse_args()


# ----------------------------- Main -----------------------------

def main():
    args = parse_args()

    # ---------- Load config ----------
    config_path = args.yaml
    if not os.path.exists(config_path):
        alt = os.path.join(os.getcwd(), os.path.basename(config_path))
        if os.path.exists(alt):
            config_path = alt
        else:
            alt2 = os.path.join(os.path.dirname(__file__), os.path.basename(config_path))
            if os.path.exists(alt2):
                config_path = alt2
            else:
                raise FileNotFoundError(f"找不到 YAML：{args.yaml}（搜索：cwd 与脚本目录）")
    config = ConfigModule(config_path)

    # ---------- Build models ----------
    ctx = build_dual_arm_from_config(config)
    dual = ctx.dual
    r1_dof, r2_dof = ctx.r1_dof, ctx.r2_dof

    # ---------- PyBullet world ----------
    body_uid = init_pybullet(args.urdf, dt=args.dt, gravity=args.gravity, realtime=args.realtime)

    # 仅使用 REVOLUTE 作为手臂关节，自动跳过夹爪 PRISMATIC
    rev_ids, rev_names = collect_arm_revolute_joints(body_uid)
    need = r1_dof + r2_dof
    if len(rev_ids) < need:
        raise RuntimeError(
            f"URDF 中可用 REVOLUTE 关节数为 {len(rev_ids)}，但 YAML 需要 {need}。请检查 URDF 与 YAML 是否匹配。"
        )

    # 先尝试基于命名拆分（panda1_/panda2_ 或 left/right）；否则按顺序切分
    names_lower = [n.lower() for n in rev_names]
    r1_mask = [
        ('panda1' in n) or ('arm1' in n) or ('left' in n) or ('l_' in n)
        for n in names_lower
    ]
    r2_mask = [
        ('panda2' in n) or ('arm2' in n) or ('right' in n) or ('r_' in n)
        for n in names_lower
    ]
    r1_joint_ids: List[int]
    r2_joint_ids: List[int]

    group1 = [jid for jid, m in zip(rev_ids, r1_mask) if m]
    group2 = [jid for jid, m in zip(rev_ids, r2_mask) if m]
    if len(group1) == r1_dof and len(group2) == r2_dof:
        r1_joint_ids, r2_joint_ids = group1, group2
    else:
        # why: 某些 URDF 名称不含 1/2/left/right，保底用顺序
        r1_joint_ids = rev_ids[:r1_dof]
        r2_joint_ids = rev_ids[r1_dof:r1_dof + r2_dof]

    # ---------- Sliders ----------
    r1_q_min, r1_q_max = np.array(config.robot1_q_min), np.array(config.robot1_q_max)
    r2_q_min, r2_q_max = np.array(config.robot2_q_min), np.array(config.robot2_q_max)

    init_r1, init_r2 = _choose_init(
        args, config, r1_q_min, r1_q_max, r2_q_min, r2_q_max, r1_dof, r2_dof
    )
    init_r1 = np.array([1.387536, 1.3089969, -1.5707963, -0.61086523,  0.0,  2.5307273, 1.3089969])
    init_r2 = np.array([-1.40499, 1.3089969, 1.5707963, -0.61086523, 0.0, 2.5307273, -1.3089969])
    # init_r1 = np.array([1.70393750617938, -1.31850797298515, -1.15903667275104, -0.688045912337351, -0.343113049074474, 1.99801745304141, -1.03151005642584])
    # init_r2 = np.array([0.940446443513692, 0.599129877099996, 0.125830284382108, -1.15833736922856, 1.19859474973311, 4.19633519586961, -1.29074359005816])

    slider_ids_r1 = add_joint_sliders("Franka 1 (Left)", r1_q_min, r1_q_max, init_r1, "R1_")
    slider_ids_r2 = add_joint_sliders("Franka 2 (Right)", r2_q_min, r2_q_max, init_r2, "R2_")

    # 初始化机器人姿态
    reset_joint_configuration(body_uid, r1_joint_ids, init_r1)
    reset_joint_configuration(body_uid, r2_joint_ids, init_r2)

    # ---------- Loop ----------
    last_print = 0.0
    overlay_id = None
    print_hz = max(0.5, float(args.print_hz))

    # 调试信息：打印映射
    print("[INFO] Revolute joints used for arms (id : name):")
    for jid, name in zip(r1_joint_ids + r2_joint_ids, [rev_names[rev_ids.index(j)] for j in r1_joint_ids + r2_joint_ids]):
        print(f"  {jid:3d} : {name}")
    print("[INFO] 拖动左侧滑条控制关节。Ctrl+C 结束。")

    while p.isConnected():
        # 读取滑条
        r1_q = np.array([p.readUserDebugParameter(sid) for sid in slider_ids_r1], dtype=float)
        r2_q = np.array([p.readUserDebugParameter(sid) for sid in slider_ids_r2], dtype=float)

        # 限幅
        r1_q = np.clip(r1_q, r1_q_min, r1_q_max)
        r2_q = np.clip(r2_q, r2_q_min, r2_q_max)

        # 应用到仿真
        reset_joint_configuration(body_uid, r1_joint_ids, r1_q)
        reset_joint_configuration(body_uid, r2_joint_ids, r2_q)

        # cooperative 的相对/绝对位姿
        q_all = np.concatenate([r1_q, r2_q], axis=0)
        if not np.all(np.isfinite(q_all)):
            raise ValueError("q 包含 NaN/Inf，请检查滑条与上下限。")
        q_all = np.ascontiguousarray(q_all, dtype=np.float64).reshape(-1)

        rel_dq = dual.relative_pose(q_all.tolist())
        abs_dq = dual.absolute_pose(q_all.tolist())

        # 打印
        t = time.time()
        if t - last_print >= (1.0 / print_hz):
            last_print = t
            rel_vec = np.array(vec8(rel_dq)).ravel()
            abs_vec = np.array(vec8(abs_dq)).ravel()
            np.set_printoptions(precision=6, suppress=True)

            print("\n================= CURRENT STATE =================")
            print("R1 q:", r1_q)
            print("R2 q:", r2_q)
            print("relative_pose (DQ, 8):", rel_vec)
            print("absolute_pose (DQ, 8):", abs_vec)

            if not args.no_overlay:
                overlay_text = (
                    f"R1 q[rad]: {np.array2string(r1_q, precision=2, suppress_small=True)}\n"
                    f"R2 q[rad]: {np.array2string(r2_q, precision=2, suppress_small=True)}\n"
                    f"relDQ[0:4]: {np.array2string(rel_vec[:4], precision=3, suppress_small=True)}   "
                    f"absDQ[0:4]: {np.array2string(abs_vec[:4], precision=3, suppress_small=True)}"
                )
                if overlay_id is not None:
                    p.removeUserDebugItem(overlay_id)
                overlay_id = p.addUserDebugText(
                    overlay_text, textPosition=[-0.9, -0.4, 1.6], textSize=1.4, lifeTime=1.2
                )

        if not args.realtime:
            p.stepSimulation()
            time.sleep(args.dt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
