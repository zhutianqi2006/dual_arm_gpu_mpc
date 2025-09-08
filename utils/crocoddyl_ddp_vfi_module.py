#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: crocoddyl_ddp_vfi_module.py
"""
Crocoddyl + DQ 双臂控制模块（无可视化）——与 MPPIAdpAnModule 使用方式类似

要点
- 采用 Crocoddyl（微分动力学为 qdot=u 的运动学模型）生成高层关节速度 u_des。
- 通过拉格朗日乘子法投影，严格满足相对位姿**速度层**等式约束：J_rel(q) u = 0。
- 在 Null(J_rel) 子空间内执行 VFI（向量场不等式）避障投影，确保避障不破坏相对位姿不变。
- 通过 HighROSModule 与 ROS2 交互：读取关节位置，发布关节速度（供 bullet_robot_ros.py 驱动可视化）。

依赖
- crocoddyl, dqrobotics, numpy, rclpy（由 utils.high_ros_module 间接使用）
- 项目内: utils.config_module.ConfigModule, utils.high_ros_module.HighROSModule

用法
>>> from utils.config_module import ConfigModule
>>> from crocoddyl_ddp_vfi_module import CrocoddylDDPVFIModule
>>> cfg = ConfigModule('ur3_and_ur3e.yaml')
>>> mod = CrocoddylDDPVFIModule(cfg, desire_abs_pose, desire_rel_pose)
>>> mod.warm_up()
>>> while True:
...     mod.play_once()

注：本文件**不**包含任何 PyBullet 可视化；可直接配合 bullet_robot_ros.py。
"""
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import crocoddyl as croco
from dqrobotics import DQ, vec8, vec4
from dqrobotics.robot_modeling import (
    DQ_SerialManipulatorDH,
    DQ_SerialManipulatorMDH,
    DQ_CooperativeDualTaskSpace,
)

# 可选：某些构建下提供解析的平移雅可比转换
try:
    from dqrobotics.kinematics import DQ_Kinematics as _DQ_Kinematics  # type: ignore
except Exception:
    try:
        from dqrobotics.robot_modeling import DQ_Kinematics as _DQ_Kinematics  # type: ignore
    except Exception:
        _DQ_Kinematics = None

# 项目内依赖
from utils.config_module import ConfigModule
from utils.high_ros_module import HighROSModule

# ---------------------- 简单工具函数 ----------------------

def dq_to_vec8(dq_obj: DQ) -> np.ndarray:
    return np.asarray(vec8(dq_obj), dtype=float).reshape(8)


def _as_vec8_weight(w, default=1.0) -> np.ndarray:
    if w is None:
        return np.full(8, float(default))
    if np.isscalar(w):
        return np.full(8, float(w))
    return np.asarray(w, dtype=float).reshape(8)


# ---------------------- DQ 运动学包装 ----------------------
class DQKino:
    def __init__(self, dual: DQ_CooperativeDualTaskSpace, q1_dof: int, q2_dof: int, prefer_analytic_translation_jac: bool = False):
        self.dual = dual
        self.n1 = int(q1_dof)
        self.n2 = int(q2_dof)
        self.nq = self.n1 + self.n2
        self.prefer_analytic_translation_jac = bool(prefer_analytic_translation_jac)

    # relative (8)
    def rel_vec8(self, q: np.ndarray) -> np.ndarray:
        return dq_to_vec8(self.dual.relative_pose(q))

    def rel_jac(self, q: np.ndarray) -> np.ndarray:
        J = np.asarray(self.dual.relative_pose_jacobian(q), dtype=float)
        assert J.shape == (8, self.nq), f"rel_jac shape {J.shape} != (8,{self.nq})"
        return J

    # absolute (8)
    def abs_vec8(self, q: np.ndarray) -> np.ndarray:
        return dq_to_vec8(self.dual.absolute_pose(q))

    def abs_jac(self, q: np.ndarray) -> np.ndarray:
        J = np.asarray(self.dual.absolute_pose_jacobian(q), dtype=float)
        assert J.shape == (8, self.nq), f"abs_jac shape {J.shape} != (8,{self.nq})"
        return J

    # 绝对平移 p(q) 及其雅可比 Jp(q)
    def abs_translation_and_jac(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = self.dual.absolute_pose(q)
        p4 = np.asarray(vec4(a.translation()), dtype=float).reshape(4)
        p = p4[1:4]
        # 解析 Jacobian（若可用）
        if self.prefer_analytic_translation_jac and _DQ_Kinematics is not None and hasattr(_DQ_Kinematics, "translation_jacobian"):
            try:
                J8 = np.asarray(self.dual.absolute_pose_jacobian(q), dtype=float)
                T4 = np.asarray(_DQ_Kinematics.translation_jacobian(J8, a), dtype=float)
                Jp = T4[1:4, :]
                return p, Jp
            except Exception:
                pass
        # 回退：有限差分
        Jp = np.zeros((3, self.nq), dtype=float)
        eps = 1e-6
        q = np.asarray(q, dtype=float).reshape(self.nq)
        for i in range(self.nq):
            dqv = np.zeros(self.nq)
            dqv[i] = eps
            tp4 = np.asarray(vec4(self.dual.absolute_pose(q + dqv).translation()), dtype=float).reshape(4)
            tm4 = np.asarray(vec4(self.dual.absolute_pose(q - dqv).translation()), dtype=float).reshape(4)
            Jp[:, i] = (tp4[1:4] - tm4[1:4]) / (2.0 * eps)
        return p, Jp


# ---------------------- Crocoddyl 差分动作模型（纯运动学） ----------------------
class DiffKinoDQ(croco.DifferentialActionModelAbstract):
    """
    x=q, u=qdot, xdot=u

    代价：
      0.5*( (g_rel - rel(q))^T W_rel (g_rel - rel(q))
           + (g_abs - abs(q))^T W_abs (g_abs - abs(q)) )

    速度整形：
      r_vabs = J_abs(q) u + beta_abs * (g_abs - abs(q))
      0.5 * r_vabs^T W_vabs r_vabs

    控制正则： 0.5*w_u*||u||^2

    注：避障在 DDP 之后做 VFI 投影；相对位姿恒定由等式投影保证。
    """

    def __init__(self, dq_kino: DQKino, weights: dict, goals: dict):
        state = croco.StateVector(dq_kino.nq)
        super().__init__(state, dq_kino.nq)
        self.kino = dq_kino
        self.w_rel_vec = _as_vec8_weight(weights.get("w_rel", 0.0))
        self.w_abs_vec = _as_vec8_weight(weights.get("w_abs", 1e3))
        self.w_u = float(weights.get("w_u", 5e-2))
        self.w_vabs_vec = _as_vec8_weight(weights.get("w_vabs", 5e2))
        self.beta_abs = float(weights.get("beta_abs", 4.0))
        self.g_rel = np.asarray(goals["g_rel"], dtype=float).reshape(8)
        self.g_abs = np.asarray(goals["g_abs"], dtype=float).reshape(8)

    def createData(self):
        return croco.DifferentialActionDataAbstract(self)

    def set_goals(self, g_rel=None, g_abs=None):
        if g_rel is not None:
            self.g_rel = np.asarray(g_rel, dtype=float).reshape(8)
        if g_abs is not None:
            self.g_abs = np.asarray(g_abs, dtype=float).reshape(8)

    def set_weights(self, w_rel=None, w_abs=None, w_u=None, w_vabs=None, beta_abs=None):
        if w_rel is not None:
            self.w_rel_vec = _as_vec8_weight(w_rel, default=0.0)
        if w_abs is not None:
            self.w_abs_vec = _as_vec8_weight(w_abs, default=1.0)
        if w_u is not None:
            self.w_u = float(w_u)
        if w_vabs is not None:
            self.w_vabs_vec = _as_vec8_weight(w_vabs, default=1.0)
        if beta_abs is not None:
            self.beta_abs = float(beta_abs)

    def calc(self, data, x, u=None):
        q = np.asarray(x, dtype=np.float64)
        if u is None:
            u = np.zeros(self.nu, dtype=np.float64)

        e_rel = self.g_rel - self.kino.rel_vec8(q)
        e_abs = self.g_abs - self.kino.abs_vec8(q)

        Ja = self.kino.abs_jac(q)
        r_vabs = Ja @ u + self.beta_abs * e_abs

        data.cost = 0.5 * (
            float(e_rel @ (self.w_rel_vec * e_rel))
            + float(e_abs @ (self.w_abs_vec * e_abs))
            + float(r_vabs @ (self.w_vabs_vec * r_vabs))
            + self.w_u * float(u @ u)
        )

        data.xout = np.ascontiguousarray(u, dtype=np.float64)

        data._e_rel = e_rel
        data._e_abs = e_abs
        data._Ja = Ja
        data._r_vabs = r_vabs

    def calcDiff(self, data, x, u=None):
        q = np.asarray(x, dtype=np.float64)
        if u is None:
            u = np.zeros(self.nu, dtype=np.float64)

        Jr = self.kino.rel_jac(q)
        Ja = data._Ja

        wr_e = self.w_rel_vec * data._e_rel
        wa_e = self.w_abs_vec * data._e_abs
        wv_r = self.w_vabs_vec * data._r_vabs

        data.Lx = - ( Jr.T @ wr_e + Ja.T @ wa_e )
        data.Lu = self.w_u * u

        data.Lu += Ja.T @ wv_r
        data.Lx += - self.beta_abs * (Ja.T @ wv_r)

        eps = 1e-6
        data.Lxx = Jr.T @ (self.w_rel_vec[:, None] * Jr) \
                 + Ja.T @ (self.w_abs_vec[:, None] * Ja) \
                 + (self.beta_abs ** 2) * (Ja.T @ (self.w_vabs_vec[:, None] * Ja)) \
                 + eps * np.eye(self.state.ndx)

        data.Luu = self.w_u * np.eye(self.nu) \
                 + Ja.T @ (self.w_vabs_vec[:, None] * Ja) \
                 + eps * np.eye(self.nu)

        data.Lxu = np.zeros((self.state.ndx, self.nu), dtype=np.float64)

        data.Fx = np.zeros((self.state.ndx, self.state.ndx), dtype=np.float64)
        data.Fu = np.eye(self.nu, dtype=np.float64)


# ---------------------- 小型 MPC/DDP 包装器 ----------------------
class CrocoMPC:
    def __init__(self, dq_kino: DQKino, dt=0.05, N=30, weights: Optional[dict] = None):
        self.kino = dq_kino
        self.nq = dq_kino.nq
        self.dt = float(dt)
        self.N = int(N)

        if weights is None:
            weights = dict(w_rel=np.zeros(8), w_abs=np.ones(8) * 1e3, w_u=5e-2, w_vabs=5e2, beta_abs=4.0)

        zeros8 = np.zeros(8)
        self.running_models: list[DiffKinoDQ] = []
        self.running: list[croco.IntegratedActionModelEuler] = []

        for _ in range(self.N):
            inner = DiffKinoDQ(
                dq_kino,
                weights,
                dict(g_rel=zeros8, g_abs=zeros8),
            )
            self.running_models.append(inner)
            self.running.append(croco.IntegratedActionModelEuler(inner, self.dt))

        term_weights = dict(
            w_rel=_as_vec8_weight(weights.get("w_rel", 0.0)) * 10.0,
            w_abs=_as_vec8_weight(weights.get("w_abs", 1e3)) * 10.0,
            w_u=5e-2,
            w_vabs=_as_vec8_weight(weights.get("w_vabs", 5e2)) * 10.0,
            beta_abs=float(weights.get("beta_abs", 4.0)),
        )
        self.terminal_model = DiffKinoDQ(
            dq_kino,
            term_weights,
            dict(g_rel=zeros8, g_abs=zeros8),
        )
        self.terminal = croco.IntegratedActionModelEuler(self.terminal_model, 0.0)

        self.problem: Optional[croco.ShootingProblem] = None
        self.solver: Optional[croco.SolverDDP] = None
        self.xs: Optional[list[np.ndarray]] = None
        self.us: Optional[list[np.ndarray]] = None

    def set_goals(self, g_rel: np.ndarray, g_abs: np.ndarray):
        for m in self.running_models:
            m.set_goals(g_rel=g_rel, g_abs=g_abs)
        self.terminal_model.set_goals(g_rel=g_rel, g_abs=g_abs)

    def set_weights(self, w_rel=None, w_abs=None, w_u=None, w_vabs=None, beta_abs=None):
        for m in self.running_models:
            m.set_weights(w_rel, w_abs, w_u, w_vabs, beta_abs)
        self.terminal_model.set_weights(
            w_rel if w_rel is not None else None,
            w_abs if w_abs is not None else None,
            5e-2 if w_u is None else max(1e-6, float(w_u)),
            w_vabs if w_vabs is not None else None,
            beta_abs if beta_abs is not None else None,
        )

    def _build_once(self, q0: np.ndarray):
        if self.problem is None:
            self.problem = croco.ShootingProblem(q0, self.running, self.terminal)
            self.solver = croco.SolverDDP(self.problem)
            q0 = np.asarray(q0, dtype=np.float64).reshape(self.nq)
            self.xs = [q0.copy() for _ in range(self.N + 1)]
            self.us = [np.zeros(self.nq, dtype=np.float64) for _ in range(self.N)]

    def _ensure_vec_list(self, arr_list: Iterable[np.ndarray], n: int) -> list[np.ndarray]:
        out = []
        for a in arr_list:
            a = np.asarray(a, dtype=np.float64).reshape(n)
            out.append(np.ascontiguousarray(a))
        return out

    def _dls_abs_step(self, q: np.ndarray, g_abs: np.ndarray, v_max: Optional[float] = None, lam=1e-3, gain=2.0) -> np.ndarray:
        Ja = self.kino.abs_jac(q)
        e_abs = g_abs - self.kino.abs_vec8(q)
        JJ = Ja @ Ja.T + lam * np.eye(8)
        u = gain * (Ja.T @ np.linalg.solve(JJ, e_abs))
        if v_max is not None:
            u = np.clip(u, -float(v_max), float(v_max))
        return u

    def step(self, q0: np.ndarray, g_rel: np.ndarray, g_abs: np.ndarray, max_iters=80, init_reg=1e-4, v_max: Optional[float] = None) -> np.ndarray:
        self._build_once(q0)
        assert self.problem is not None and self.solver is not None

        self.problem.x0 = np.asarray(q0, dtype=np.float64).reshape(self.nq)
        self.set_goals(g_rel=g_rel, g_abs=g_abs)

        assert self.xs is not None and self.us is not None
        self.xs = self._ensure_vec_list(self.xs, self.nq)
        self.us = self._ensure_vec_list(self.us, self.nq)

        try:
            self.solver.solve(self.xs, self.us, int(max_iters), False, float(init_reg))
        except Exception:
            # 回退：少量迭代 + 更大阻尼
            self.solver.solve(self.xs, self.us, 10, False, 1e-2)

        self.xs, self.us = self.solver.xs, self.solver.us
        u0 = np.array(self.us[0], dtype=float)
        if np.linalg.norm(u0) < 1e-6:
            u0 = self._dls_abs_step(self.problem.x0, g_abs, v_max=v_max, lam=1e-3, gain=2.0)
        return u0


# ---------------------- 等式投影器（拉格朗日乘子） ----------------------
class EqualityProjector:
    """将 u_des 投影到 {u | J u = 0} 等式约束集合。
    用途：冻结相对位姿（速度层）——J_rel(q) u = 0。
    """

    def __init__(self, reg: float = 1e-8):
        self.reg = float(reg)

    def project(self, J: np.ndarray, u_des: np.ndarray) -> np.ndarray:
        J = np.asarray(J, dtype=float)
        u = np.asarray(u_des, dtype=float)
        if J.size == 0:
            return u.copy()
        m, _ = J.shape
        JJt = J @ J.T + self.reg * np.eye(m)
        lam = np.linalg.solve(JJt, J @ u)
        return u - J.T @ lam

    def nullspace_matrix(self, J: np.ndarray, n: int) -> np.ndarray:
        J = np.asarray(J, dtype=float)
        m = J.shape[0]
        JJt = J @ J.T + self.reg * np.eye(m)
        return np.eye(n) - J.T @ np.linalg.solve(JJt, J)


# ---------------------- VFI 投影器（球形障碍） ----------------------
@dataclass
class Sphere:
    center: np.ndarray
    radius: float
    buffer: float = 0.0


class VFIProjector:
    """速度层不等式投影：min ½||u - u_des||²  s.t.  A u >= b。
    若提供 J_eq，则首先将 u 投影到 J_eq u = 0，然后仅在 Null(J_eq) 方向上修正。
    """

    def __init__(self, kino: DQKino, obstacles: Optional[Iterable[dict|Sphere]], gamma: float = 6.0, activate_margin: float = 0.10, max_iters: int = 20):
        self.kino = kino
        self.gamma = float(gamma)
        self.activate_margin = float(activate_margin)
        self.max_iters = int(max_iters)
        self._eq = EqualityProjector(reg=1e-8)
        self.set_obstacles(obstacles)

    def set_obstacles(self, obstacles: Optional[Iterable[dict|Sphere]]):
        self.obstacles: list[dict] = []
        if obstacles is None:
            return
        for o in obstacles:
            if isinstance(o, Sphere):
                self.obstacles.append({"center": np.asarray(o.center, dtype=float).reshape(3), "radius": float(o.radius), "buffer": float(o.buffer)})
            else:
                # 期望键：center(np.array(3,)), radius(float), 可选 buffer(float)
                self.obstacles.append({
                    "center": np.asarray(o.get("center", [0, 0, 0]), dtype=float).reshape(3),
                    "radius": float(o.get("radius", 0.05)),
                    "buffer": float(max(0.0, o.get("buffer", 0.0))),
                })

    def _build_Ab(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
        p, Jp = self.kino.abs_translation_and_jac(q)
        A_rows: list[np.ndarray] = []
        b_vals: list[float] = []
        active_idx: list[int] = []
        for idx, o in enumerate(self.obstacles):
            c = np.asarray(o["center"], dtype=float).reshape(3)
            R = float(o["radius"]) + float(max(0.0, o.get("buffer", 0.0)))
            d = p - c
            dist = float(np.linalg.norm(d))
            if dist < 1e-9:
                continue
            s = dist - R
            if s >= self.activate_margin:
                continue  # 太远，不激活
            n = d / dist
            A_rows.append(n @ Jp)
            b_vals.append(- self.gamma * s)
            active_idx.append(idx)
        if len(A_rows) == 0:
            A = np.zeros((0, self.kino.nq), dtype=float)
            b = np.zeros(0, dtype=float)
        else:
            A = np.vstack(A_rows)
            b = np.asarray(b_vals, dtype=float)
        return A, b, active_idx

    def project(self, q: np.ndarray, u_des: np.ndarray, v_limit: Optional[float] = None, J_eq: Optional[np.ndarray] = None, reg_eq: float = 1e-8) -> tuple[np.ndarray, int]:
        u = np.asarray(u_des, dtype=float).reshape(self.kino.nq).copy()
        if v_limit is not None:
            u = np.clip(u, -float(v_limit), float(v_limit))

        # 先投影到等式集合
        N = None
        if J_eq is not None and J_eq.size > 0:
            self._eq.reg = float(reg_eq)
            u = self._eq.project(J_eq, u)
            N = self._eq.nullspace_matrix(J_eq, self.kino.nq)

        A, b, active_idx = self._build_Ab(q)
        if A.shape[0] == 0:
            return u, 0

        # 交替投影到半空间 {u | aᵢ·u >= bᵢ}
        for _ in range(self.max_iters):
            viol = A @ u < b - 1e-12
            if not np.any(viol):
                break
            for i in np.where(viol)[0]:
                ai = A[i]
                gap = b[i] - float(ai @ u)
                if N is not None:
                    ai_dir = ai @ N
                    denom = float(ai_dir @ ai_dir) + 1e-12
                    if denom < 1e-10:
                        continue  # 在 Null(J_eq) 中不可行，跳过
                    u += (gap / denom) * (N.T @ ai.T)
                else:
                    denom = float(ai @ ai) + 1e-12
                    u += (gap / denom) * ai

        if v_limit is not None:
            u = np.clip(u, -float(v_limit), float(v_limit))
        return u, len(active_idx)


# ---------------------- 控制模块（与 MPPIAdpAnModule 相似接口） ----------------------
class CrocoddylDDPVFIModule:
    """基于 Crocoddyl+VFI 的 ROS2 控制模块（无可视化）。

    - 从 ROS 订阅/读取当前两臂关节位置；
    - 通过 DDP 计算 u_des，并进行等式与 VFI 投影；
    - 通过 HighROSModule 发布速度到 topic（bullet_robot_ros.py 负责显示与积分）。
    """

    def __init__(self, config: ConfigModule, desire_abs_pose: Iterable[float], desire_rel_pose: Iterable[float], obstacles: Optional[Iterable[dict|Sphere]] = None):
        # 目标
        self.desire_abs_pose = DQ(desire_abs_pose).normalize()
        self.desire_rel_pose = DQ(desire_rel_pose).normalize()
        self.g_abs = dq_to_vec8(self.desire_abs_pose)
        self.g_rel = dq_to_vec8(self.desire_rel_pose)

        # 机器人模型（CPU）
        # robot1
        robot1_dh = np.array(config.robot1_dh_mat, dtype=float).T
        self.robot1_q_num = int(config.robot1_q_num)
        r1_base = DQ(config.robot1_base).normalize()
        r1_ee = DQ(config.robot1_effector).normalize()
        if int(getattr(config, "robot1_dh_type", 0)) == 1:
            self.cpu_robot1 = DQ_SerialManipulatorMDH(robot1_dh)
        else:
            self.cpu_robot1 = DQ_SerialManipulatorDH(robot1_dh)
        self.cpu_robot1.set_base_frame(r1_base)
        self.cpu_robot1.set_reference_frame(r1_base)
        self.cpu_robot1.set_effector(r1_ee)
        # robot2
        robot2_dh = np.array(config.robot2_dh_mat, dtype=float).T
        self.robot2_q_num = int(config.robot2_q_num)
        r2_base = DQ(config.robot2_base).normalize()
        r2_ee = DQ(config.robot2_effector).normalize()
        if int(getattr(config, "robot2_dh_type", 0)) == 1:
            self.cpu_robot2 = DQ_SerialManipulatorMDH(robot2_dh)
        else:
            self.cpu_robot2 = DQ_SerialManipulatorDH(robot2_dh)
        self.cpu_robot2.set_base_frame(r2_base)
        self.cpu_robot2.set_reference_frame(r2_base)
        self.cpu_robot2.set_effector(r2_ee)

        # 协作模型 + 运动学包装
        self.dual = DQ_CooperativeDualTaskSpace(self.cpu_robot1, self.cpu_robot2)
        self.kino = DQKino(self.dual, self.robot1_q_num, self.robot2_q_num, prefer_analytic_translation_jac=False)
        self.nq = self.kino.nq

        # DDP 参数
        self.ddp_dt = float(getattr(config, "ddp_dt", 0.15))
        self.ddp_N = int(getattr(config, "ddp_N", 15))
        self.ddp_max_iters = int(getattr(config, "ddp_max_iters", 400))
        self.v_limit = float(getattr(config, "ddp_v_limit",1.0))

        # 代价权重（默认绝对旋转更重，平移在速度整形项里处理）
        w_abs_rot = float(getattr(config, "ddp_w_abs_rot", 5e2))
        w_abs_trans = float(getattr(config, "ddp_w_abs_trans", 0.0))
        self.w_abs_vec = np.array([w_abs_rot, w_abs_rot, w_abs_rot, w_abs_rot, w_abs_trans, w_abs_trans, w_abs_trans, w_abs_trans], dtype=float)
        mpc_weights = {
            "w_rel": np.zeros(8),        # 相对位姿由等式冻结
            "w_abs": self.w_abs_vec,
            "w_vabs": np.zeros(8),      # 速度整形默认关闭（必要时可打开）
            "beta_abs": 0.0,
            "w_u": 0.0,
        }
        self.mpc = CrocoMPC(self.kino, dt=self.ddp_dt, N=self.ddp_N, weights=mpc_weights)

        # 投影器
        self.eq = EqualityProjector(reg=1e-8)
        self.vfi = VFIProjector(self.kino, obstacles, gamma=float(getattr(config, "vfi_gamma", 8.0)), activate_margin=float(getattr(config, "vfi_activate_margin", 0.10)), max_iters=int(getattr(config, "vfi_max_iters", 25)))

        # 状态缓存（来自 ROS）
        self.robot1_q = np.zeros(self.robot1_q_num)
        self.robot2_q = np.zeros(self.robot2_q_num)

        # ROS 高层接口
        self.ros_module = HighROSModule(config)
        self.ros_thread = threading.Thread(target=self.ros_module.run)
        self.ros_thread.daemon = True
        self.ros_thread.start()
        self.start_time = time.time()

    # ---------------------- ROS & 状态 ----------------------
    def update_joint_states(self):
        """从 HighROSModule 读取两臂当前关节角。"""
        q1, q2 = self.ros_module.read_joint_state()
        self.robot1_q = np.asarray(q1, dtype=float).reshape(self.robot1_q_num)
        self.robot2_q = np.asarray(q2, dtype=float).reshape(self.robot2_q_num)

    def get_q_all(self) -> np.ndarray:
        return np.concatenate([self.robot1_q, self.robot2_q])

    # ---------------------- 运行循环 API ----------------------
    def warm_up(self, rounds: int = 5):
        """预热 DDP 与投影器；不发布控制。"""
        for _ in range(max(1, int(rounds))):
            self.update_joint_states()
            q = self.get_q_all()
            _ = self.mpc.step(q, g_rel=self.g_rel, g_abs=self.g_abs, max_iters=min(self.ddp_max_iters, 60), v_max=self.v_limit)
        self.start_time = time.time()

    def play_once(self):
        """执行一次：读取 q -> DDP -> 等式冻结 -> VFI 投影 -> 发布 u。"""
        self.update_joint_states()
        q = self.get_q_all()

        # 1) DDP 生成 u_des
        u_des = self.mpc.step(q, g_rel=self.g_rel, g_abs=self.g_abs, max_iters=self.ddp_max_iters, v_max=self.v_limit)
        u_des = np.clip(u_des, -self.v_limit, self.v_limit)

        # 2) 等式冻结：J_rel(q) u = 0
        J_rel = self.kino.rel_jac(q)
        u_eq = self.eq.project(J_rel, u_des)

        # 3) VFI 避障（限制在 Null(J_rel) 内）
        u_cmd, _ = self.vfi.project(q, u_eq, v_limit=self.v_limit, J_eq=J_rel)

        # 4) 发布到 ROS（由 bullet_robot_ros.py 进行积分与可视化）
        self.ros_module.write_high_u(u_cmd.tolist())

    # ---------------------- 目标与障碍管理 ----------------------
    def set_abs_goal(self, desire_abs_pose: Iterable[float]):
        self.desire_abs_pose = DQ(desire_abs_pose).normalize()
        self.g_abs = dq_to_vec8(self.desire_abs_pose)

    def set_rel_goal(self, desire_rel_pose: Iterable[float]):
        self.desire_rel_pose = DQ(desire_rel_pose).normalize()
        self.g_rel = dq_to_vec8(self.desire_rel_pose)

    def set_obstacles(self, obstacles: Optional[Iterable[dict|Sphere]]):
        self.vfi.set_obstacles(obstacles)


# ---------------------- 可选：最小运行脚本 ----------------------
if __name__ == "__main__":
    # 示例参数（请根据实际修改）
    os.environ['ROS_DOMAIN_ID'] = '16'
    import rclpy
    rclpy.init(args=None)
    desire_abs_pose = [-0.009809, -0.700866, -0.008828, 0.713171, 0.03289, -0.000662, -0.283115, -0.003703]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.001626, -0.002018, 0.28023, 0.00204]

    cfg_path = os.path.join(os.path.dirname(__file__), "ur3_and_ur3e.yaml")
    cfg = ConfigModule(cfg_path)

    # 可选：障碍（球）
    obstacles = [
        {"center": np.array([0.5, 0.00, 0.25]), "radius": 0.04, "buffer": 0.02},
        {"center": np.array([0.45, 0.06, 0.25]), "radius": 0.04, "buffer": 0.02},
    ]

    ctrl = CrocoddylDDPVFIModule(cfg, desire_abs_pose, desire_rel_pose, obstacles)
    ctrl.warm_up()
    print("[CrocoddylDDPVFIModule] running. Press Ctrl+C to stop.")
    try:
        while True:
            ctrl.play_once()
            time.sleep(max(0.01, ctrl.ddp_dt))
    except KeyboardInterrupt:
        pass
