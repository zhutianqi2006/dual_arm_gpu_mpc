#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: bullet_gui.py
"""
Bullet Dual-Arm GUI (No ROS2) + DQ Robotics
-------------------------------------------
- 左侧 PyBullet 滑条直接控制两条 6-DOF 机械臂的关节角。
- 后台持续输出：
    * 两臂关节位置（rad）
    * 相对位姿（双四元数，8 维）
    * 绝对位姿（双四元数，8 维）
- 依赖：
    pip install pybullet numpy pyyaml dqrobotics
- 期待的本地文件：
    - ur3_and_ur3e.yaml
    - config_module.py
- 可选参数：见 `-h`

新增：可设置起始角
==================
1) 命令行直接指定：
   --r1-init "0,0,0,0,0,0" --r2-init "0,0,0,0,0,0"
   搭配 --init-deg 表示度制输入，默认弧度。
2) 策略选择：--init-mode {mid,zero,min,max,yaml}
   - mid:    (q_min + q_max)/2（默认）
   - zero:   全 0
   - min:    q_min
   - max:    q_max
   - yaml:   若 YAML 含 robot1_q_init / robot2_q_init 则读取
3) 校验：默认越界会自动截断到范围；加 --strict-init 改为越界报错。
"""
import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pybullet as p
import pybullet_data

# 用户侧的 ConfigModule
try:
    from utils.config_module import ConfigModule
except ImportError:
    print("[ERROR] 找不到 config_module.py，请将本脚本与 config_module.py 放在同一目录。", file=sys.stderr)
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


def _as_unit_dq(x):
    """将任何可被 DQ() 构造的输入转成单位化的 DQ。"""
    dq = DQ(x)
    try:
        dq = dq.normalize()
    except Exception:
        # 某些情况下 normalize 不存在或失败，保持 dq 原样
        pass
    return dq


def build_dual_arm_from_config(config: ConfigModule) -> DualArmContext:
    """从 YAML/ConfigModule 构建两臂与 cooperative dual-arm 模型，并返回上下文以保活。"""
    # ---- Robot 1 ----
    robot1_dh = np.array(config.robot1_dh_mat)
    if robot1_dh.shape[0] == 5:
        dh1 = robot1_dh  # 5 x n
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

    # cooperative 模型（注意：C++ 层可能保存裸指针，因此必须保活 robot1/2）
    dual = DQ_CooperativeDualTaskSpace(robot1, robot2)

    return DualArmContext(robot1, robot2, dual, robot1_q_num, robot2_q_num)


def init_pybullet(urdf_path: str, dt: float, gravity: float, realtime: bool) -> int:
    """初始化 PyBullet，加载地面与机器人 URDF，返回 body uid。"""
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
        cameraDistance=1.1,
        cameraYaw=45,
        cameraPitch=-35,
        cameraTargetPosition=[0.0, 0.0, 0.2],
    )
    return model_uid


def collect_movable_joints(body_uid: int) -> List[int]:
    """收集 URDF 中所有可动关节索引（旋转/移动）。"""
    n = p.getNumJoints(body_uid)
    movable = []
    for j in range(n):
        ji = p.getJointInfo(body_uid, j)
        joint_type = ji[2]
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            movable.append(j)
    return movable


def reset_joint_configuration(body_uid: int, joint_ids: List[int], q: np.ndarray) -> None:
    """直接设置关节角（无动力学）。"""
    for idx, qval in zip(joint_ids, q):
        p.resetJointState(body_uid, idx, float(qval))


def add_joint_sliders(title: str, qmin: np.ndarray, qmax: np.ndarray, q0: np.ndarray, prefix: str) -> List[int]:
    """为一组 DOF 添加滑条，返回 slider id 列表。"""
    p.addUserDebugText(title, [-0.5, 0, 1.2], textSize=1.8, lifeTime=0)
    sliders = []
    dof = len(q0)
    for i in range(dof):
        # why: 将初值 q0 用作滑条默认值
        s = p.addUserDebugParameter(f"{prefix}{i+1:02d}", float(qmin[i]), float(qmax[i]), float(q0[i]))
        sliders.append(s)
    return sliders

# ----------------------------- 新增：起始角处理 -----------------------------

def _parse_csv_floats(csv: str, expected_len: int) -> np.ndarray:
    """解析逗号分隔的浮点数；长度不符直接报错，避免掩盖输入错误。"""
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
    """综合命令行/配置/策略，生成两臂起始角。"""
    def mid(a, b):
        return (np.array(a, dtype=float) + np.array(b, dtype=float)) * 0.5

    # 1) R1
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
    else:  # mid
        r1 = mid(r1_q_min, r1_q_max)

    # 2) R2
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
    else:  # mid
        r2 = mid(r2_q_min, r2_q_max)

    # 单位：度->弧度
    if args.init_deg:
        r1 = np.deg2rad(r1)
        r2 = np.deg2rad(r2)

    # 限幅/校验
    def limit_or_raise(q, qmin, qmax, name: str):
        if args.strict_init:
            if np.any(q < qmin) or np.any(q > qmax):
                idxs = np.where((q < qmin) | (q > qmax))[0]
                raise ValueError(f"{name} 初值越界于关节 {idxs.tolist()}；使用 --strict-init 以外模式会自动截断。")
            return q.astype(float)
        # why: 默认更友好，自动截断并提示
        clipped = np.clip(q, qmin, qmax)
        if not np.allclose(clipped, q):
            print(f"[WARN] {name} 初值超出上下限，已截断到有效范围。", file=sys.stderr)
        return clipped.astype(float)

    r1 = limit_or_raise(r1, r1_q_min, r1_q_max, 'R1')
    r2 = limit_or_raise(r2, r2_q_min, r2_q_max, 'R2')

    return r1, r2

# -------------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(description="Dual-Arm PyBullet GUI (no ROS2) + DQ Robotics")
    ap.add_argument("--urdf", type=str, default="model/dual_arm_model/dual_arm_model.urdf",
                    help="URDF 相对/绝对路径")
    ap.add_argument("--print-hz", type=float, default=10.0, help="控制台打印频率 Hz")
    ap.add_argument("--dt", type=float, default=DEFAULT_DT, help="仿真步长秒 (默认 1/240)")
    ap.add_argument("--gravity", type=float, default=-9.81, help="重力加速度 (默认 -9.81)")
    ap.add_argument("--realtime", action="store_true", help="使用实时仿真 (RealTimeSimulation)")
    ap.add_argument("--no-overlay", action="store_true", help="关闭窗口内 overlay 文本")

    # 新增：起始角相关
    ap.add_argument("--r1-init", type=str, default=None,
                    help="Robot1 起始角，逗号分隔，例如 \"0,0,0,0,0,0\"。默认单位为弧度，配合 --init-deg 则为度")
    ap.add_argument("--r2-init", type=str, default=None,
                    help="Robot2 起始角，逗号分隔，例如 \"0,0,0,0,0,0\"。默认单位为弧度，配合 --init-deg 则为度")
    ap.add_argument("--init-deg", action="store_true", help="将 --r1-init / --r2-init 解释为度制")
    ap.add_argument("--init-mode", choices=["mid", "zero", "min", "max", "yaml"], default="mid",
                    help="当未提供 --r1-init/--r2-init 时的起始角策略。yaml 会尝试读取 YAML 内 robot*_q_init")
    ap.add_argument("--strict-init", action="store_true",
                    help="严格校验起始角长度与范围，若越界即报错（默认会自动截断到范围内）")
    return ap.parse_args()


def main():
    args = parse_args()

    # ---------- Load config ----------
    config_path = os.path.join(os.path.dirname(__file__), "ur3_and_ur3e.yaml")
    if not os.path.exists(config_path):
        alt = os.path.join(os.getcwd(), "ur3_and_ur3e.yaml")
        if os.path.exists(alt):
            config_path = alt
        else:
            raise FileNotFoundError("找不到 ur3_and_ur3e.yaml，请将其放在脚本同目录或当前工作目录。")
    config = ConfigModule(config_path)

    # ---------- Build models ----------
    ctx = build_dual_arm_from_config(config)
    dual = ctx.dual
    r1_dof, r2_dof = ctx.r1_dof, ctx.r2_dof

    # ---------- PyBullet world ----------
    body_uid = init_pybullet(args.urdf, dt=args.dt, gravity=args.gravity, realtime=args.realtime)

    # 映射 URDF 关节
    movable = collect_movable_joints(body_uid)
    if len(movable) < (r1_dof + r2_dof):
        raise RuntimeError(
            f"URDF 可动关节 {len(movable)} 个，但配置需要 {r1_dof + r2_dof} 个。请检查 URDF 与 YAML 是否匹配。"
        )
    r1_joint_ids = movable[:r1_dof]
    r2_joint_ids = movable[r1_dof: r1_dof + r2_dof]

    # ---------- Sliders ----------
    r1_q_min, r1_q_max = np.array(config.robot1_q_min), np.array(config.robot1_q_max)
    r2_q_min, r2_q_max = np.array(config.robot2_q_min), np.array(config.robot2_q_max)

    # 计算起始角（覆盖原来的 mid 固定策略）
    init_r1, init_r2 = _choose_init(
        args, config, r1_q_min, r1_q_max, r2_q_min, r2_q_max, r1_dof, r2_dof
    )
    init_r1 = np.array([-1.8470081584056457, -2.7298507268179617, -0.6953932972144096,
                      -1.508942496823497,  2.0236098037789576, -0.31532559669045146])
    init_r2 = np.array([1.842840084853423, -0.48057750070854266,  0.8378998011418625,
                       -1.7586738880406665, -2.056763439048601,  3.415677557660605])

    slider_ids_r1 = add_joint_sliders("Robot 1 (Left)", r1_q_min, r1_q_max, init_r1, "R1_")
    slider_ids_r2 = add_joint_sliders("Robot 2 (Right)", r2_q_min, r2_q_max, init_r2, "R2_")

    # 初始化机器人姿态
    reset_joint_configuration(body_uid, r1_joint_ids, init_r1)
    reset_joint_configuration(body_uid, r2_joint_ids, init_r2)

    # ---------- Loop ----------
    last_print = 0.0
    overlay_id = None
    print_hz = max(0.5, float(args.print_hz))
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

        # 计算 cooperative 的相对/绝对位姿（防御式调用）
        q_all = np.concatenate([r1_q, r2_q], axis=0)
        if not np.all(np.isfinite(q_all)):
            raise ValueError("q 包含 NaN/Inf，请检查滑条与上下限。")
        q_all = np.ascontiguousarray(q_all, dtype=np.float64).reshape(-1)

        # 以 Python list 传入，避免底层与 numpy 的兼容问题
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
                # 窗口内 overlay（只显示前 4 个数，避免太长）
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
