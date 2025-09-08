#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: crocoddyl_ddp_vfi_rect.py
"""
Dual-arm DQ + Crocoddyl demo — Lagrange-multiplier equality for **relative-pose invariance** +
Vector Field Inequalities (VFI) obstacle avoidance with **rectangular boxes** + PyBullet playback (no ROS).

WHAT CHANGED (vs. spherical obstacles)
- 支持 `shape: "box"`（长方体）障碍，基于盒子的**有符号距离函数 SDF**与法向计算：
  - s(p) > 0: 点在盒外，数值为到表面的最短距离。
  - s(p) < 0: 点在盒内，数值为到表面的负距离。
  - ∇s(p) 为指向外侧的单位法线（几乎处处可微）。
- VFI 约束仍为速度层障碍函数： `\dot{s}(q) = (∂s/∂p) J_p(q) u ≥ -γ s(q)`。
- `buffer` 用于安全膨胀： 用 `s_eff = s - buffer` 进入不等式与激活逻辑。
- 仍兼容 `shape:"sphere"`（便于回滚/对比）。

Run: `python crocoddyl_ddp_vfi_rect.py`
Deps: crocoddyl, dqrobotics, numpy, optional pybullet
"""

from __future__ import annotations
import os
import time
import numpy as np
import crocoddyl as croco
from dqrobotics import DQ, vec8, vec4
# Analytic translation jacobian provider (some builds expose it here)
try:
    from dqrobotics.kinematics import DQ_Kinematics as _DQ_Kinematics  # type: ignore
except Exception:
    try:
        from dqrobotics.robot_modeling import DQ_Kinematics as _DQ_Kinematics  # type: ignore
    except Exception:
        _DQ_Kinematics = None
from dqrobotics.robot_modeling import (
    DQ_SerialManipulatorDH,
    DQ_SerialManipulatorMDH,
    DQ_CooperativeDualTaskSpace,
)

from utils.config_module import ConfigModule

# ---------------------- Optional: PyBullet offline viewer ----------------------
try:
    import pybullet as pyb
    import pybullet_data
    _HAVE_BULLET = True
except Exception:
    _HAVE_BULLET = False


class BulletPlayback:
    """Minimal PyBullet viewer to replay a joint trajectory after optimization.

    Uses pyb_utils.Robot if available; otherwise falls back to raw joint indices.
    Draws boxes/spheres based on `shape`.
    """

    def __init__(self, urdf_path="model/dual_arm_model/dual_arm_model.urdf", timestep=1.0 / 60.0, obstacles=None):
        if not _HAVE_BULLET:
            raise RuntimeError("PyBullet not installed. Please `pip install pybullet`.")
        self.client = pyb.connect(pyb.GUI)
        pyb.setTimeStep(timestep, physicsClientId=self.client)
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self._load_world()
        self.robot_id = self._load_robot(urdf_path)
        self._maybe_wrap_robot()
        self._build_joint_index_cache()
        self._obstacle_visual_ids = []
        self._point_visual_ids = []
        pyb.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=51,
            cameraPitch=-32,
            cameraTargetPosition=[0.0, 0.0, 0.0],
        )
        self.obstacles = obstacles if obstacles is not None else []
        if self.obstacles:
            self.draw_obstacles(self.obstacles)

    def _load_world(self):
        pyb.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=self.client)

    def _load_robot(self, urdf_rel):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(base_dir, urdf_rel)
        path = local_path if os.path.exists(local_path) else urdf_rel
        if not os.path.exists(path):
            print(f"[BulletPlayback] URDF not found at '{path}'. Make sure the path is correct.")
        return pyb.loadURDF(path, [0, 0, 0], useFixedBase=True, physicsClientId=self.client)

    def _maybe_wrap_robot(self):
        self._pyb_robot = None
        try:
            import pyb_utils  # noqa: F401
            self._pyb_robot = pyb_utils.Robot(self.robot_id, client_id=self.client)
        except Exception:
            self._pyb_robot = None

    def _build_joint_index_cache(self):
        self._rev_joint_indices = []
        nj = pyb.getNumJoints(self.robot_id, physicsClientId=self.client)
        for j in range(nj):
            ji = pyb.getJointInfo(self.robot_id, j, physicsClientId=self.client)
            jtype = ji[2]
            if jtype in (pyb.JOINT_REVOLUTE, pyb.JOINT_PRISMATIC):
                self._rev_joint_indices.append(j)

    def reset_q(self, q_all):
        q = np.asarray(q_all, dtype=float).ravel()
        if self._pyb_robot is not None:
            self._pyb_robot.reset_joint_configuration(q)
            return
        for j, v in zip(self._rev_joint_indices, q):
            pyb.resetJointState(self.robot_id, j, float(v), physicsClientId=self.client)

    def play(self, q_traj, rate_hz=30, loop=False):
        if len(q_traj) == 0:
            print("[BulletPlayback] Empty trajectory.")
            return
        dt = 1.0 / max(1, int(rate_hz))
        try:
            while True:
                for q in q_traj:
                    self.reset_q(q)
                    pyb.stepSimulation(physicsClientId=self.client)
                    time.sleep(dt)
                if not loop:
                    break
        except KeyboardInterrupt:
            pass

    def draw_obstacles(self, obstacles=None):
        obs = obstacles if obstacles is not None else getattr(self, "obstacles", [])
        for o in obs:
            shape = o.get("shape", "box")
            if shape == "sphere":
                self._draw_sphere(o)
            else:
                self._draw_box(o)

    def _draw_sphere(self, o, rgba=None):
        if rgba is None:
            rgba = [1, 0, 0, 0.35]
        c = np.asarray(o.get("center", [0, 0, 0]), dtype=float).reshape(3)
        R = float(o.get("radius", 0.1))
        color = o.get("color", rgba)
        try:
            vis = pyb.createVisualShape(pyb.GEOM_SPHERE, radius=R, rgbaColor=color)
            bid = pyb.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis, basePosition=c)
            self._obstacle_visual_ids.append((bid, vis))
        except Exception as e:
            print(f"[BulletPlayback] Failed to draw sphere at {c} r={R}: {e}")

    def _draw_box(self, o, rgba=None):
        if rgba is None:
            rgba = [1, 0, 0, 0.35]
        c = np.asarray(o.get("center", [0, 0, 0]), dtype=float).reshape(3)
        half = np.asarray(o.get("half", [0.05, 0.05, 0.05]), dtype=float).reshape(3)
        color = o.get("color", rgba)
        quat = np.asarray(o.get("quat_xyzw", [0, 0, 0, 1]), dtype=float).reshape(4)
        try:
            vis = pyb.createVisualShape(pyb.GEOM_BOX, halfExtents=half.tolist(), rgbaColor=color)
            bid = pyb.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis, basePosition=c, baseOrientation=quat.tolist())
            self._obstacle_visual_ids.append((bid, vis))
        except Exception as e:
            print(f"[BulletPlayback] Failed to draw box at {c} half={half}: {e}")

    def draw_points(self, points, radius=0.01, rgba=None):
        if rgba is None:
            rgba = [1, 0, 0, 1]
        try:
            vis = pyb.createVisualShape(pyb.GEOM_SPHERE, radius=float(radius), rgbaColor=rgba)
            for p in points:
                p = np.asarray(p, dtype=float).reshape(3)
                pyb.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis, basePosition=p)
        except Exception as e:
            print(f"[BulletPlayback] Failed to draw points: {e}")

# ---------------------- Helpers ----------------------

def dq_to_vec8(dq_obj: DQ) -> np.ndarray:
    return np.asarray(vec8(dq_obj), dtype=float).reshape(8)


def _as_vec8_weight(w, default=1.0) -> np.ndarray:
    if w is None:
        return np.full(8, float(default))
    if np.isscalar(w):
        return np.full(8, float(w))
    arr = np.asarray(w, dtype=float).reshape(8)
    return arr

# ---------------------- Geometry & SDF for boxes ----------------------

def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to 3x3 rotation matrix. Assumes q normalized or near-normalized."""
    x, y, z, w = [float(v) for v in q]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)],
    ], dtype=float)
    return R


def _box_localize(p: np.ndarray, box: dict[str, object]):
    c = np.asarray(box.get("center", [0, 0, 0]), dtype=float).reshape(3)
    half = np.asarray(box.get("half", [0.05, 0.05, 0.05]), dtype=float).reshape(3)
    quat = np.asarray(box.get("quat_xyzw", [0, 0, 0, 1]), dtype=float).reshape(4)
    R = quat_xyzw_to_R(quat)
    p_local = R.T @ (np.asarray(p, dtype=float).reshape(3) - c)
    return p_local, R, c, half


def box_sdf_and_grad_world(p: np.ndarray, box: dict[str, object]) -> tuple[float, np.ndarray]:
    """Signed distance and outward unit normal (world) to an oriented box.

    - Outside: distance to clamped point; gradient along vector to that point.
    - Inside: negative min margin to faces; gradient is face normal of nearest face.
    """
    p_l, R, _, half = _box_localize(p, box)
    # Outside case
    clamped = np.minimum(np.maximum(p_l, -half), half)
    delta = p_l - clamped
    nrm = np.linalg.norm(delta)
    if nrm > 1e-12:
        s = float(nrm)
        grad_local = delta / nrm
        grad_world = R @ grad_local
        return s, grad_world
    # Inside case
    m = half - np.abs(p_l)
    i = int(np.argmin(m))  # nearest face axis
    s = -float(m[i])
    grad_local = np.zeros(3, dtype=float)
    grad_local[i] = np.sign(p_l[i]) if np.abs(p_l[i]) > 1e-12 else 1.0
    grad_world = R @ grad_local
    # ensure unit length
    gnorm = np.linalg.norm(grad_world)
    if gnorm > 1e-12:
        grad_world /= gnorm
    return s, grad_world


def signed_distances_to_obstacles(p: np.ndarray, obstacles: list[dict], use_buffer: bool = False) -> np.ndarray:
    """Vector of signed distances to each obstacle at point p (world).
    - For box: SDF(box) - (buffer if use_buffer).
    - For sphere: ||p-c|| - R (R includes buffer if use_buffer=True).
    """
    dists = []
    p = np.asarray(p, dtype=float).reshape(3)
    for o in obstacles:
        shape = o.get("shape", "box")
        if shape == "sphere":
            c = np.asarray(o.get("center", [0, 0, 0]), dtype=float).reshape(3)
            R = float(o.get("radius", 0.0)) + (float(max(0.0, o.get("buffer", 0.0))) if use_buffer else 0.0)
            dists.append(np.linalg.norm(p - c) - R)
        else:
            s, _ = box_sdf_and_grad_world(p, o)
            if use_buffer:
                s -= float(max(0.0, o.get("buffer", 0.0)))
            dists.append(s)
    return np.asarray(dists, dtype=float)


# ---------------------- Low-level module (no ROS) ----------------------
class LowLevelModule:
    """Relative pose via CLIK (legacy) or via equality projection; inject DDP velocity.

    默认仍保留原 CLIK 接口 `get_u`（便于对比/回滚）。
    """

    def __init__(self, config: ConfigModule, desire_abs_pose, desire_rel_pose, nullspace_gain: float = 0.8):
        self.rel_gain = float(config.rel_gain)
        self.abs_gain = float(getattr(config, "abs_gain", 1.0))
        self.nullspace_gain = float(np.clip(nullspace_gain, 0.0, 1.0))

        self.desire_rel_pose = DQ(desire_rel_pose).normalize()
        self.desire_abs_pose = DQ(desire_abs_pose).normalize()

        # robot1
        robot1_dh = np.array(config.robot1_dh_mat, dtype=float).T
        self.robot1_q_num = int(config.robot1_q_num)
        r1_base = DQ(config.robot1_base).normalize()
        r1_ee = DQ(config.robot1_effector).normalize()
        if int(config.robot1_dh_type) == 1:
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
        if int(config.robot2_dh_type) == 1:
            self.cpu_robot2 = DQ_SerialManipulatorMDH(robot2_dh)
        else:
            self.cpu_robot2 = DQ_SerialManipulatorDH(robot2_dh)
        self.cpu_robot2.set_base_frame(r2_base)
        self.cpu_robot2.set_reference_frame(r2_base)
        self.cpu_robot2.set_effector(r2_ee)

        # cooperative model
        self.dual = DQ_CooperativeDualTaskSpace(self.cpu_robot1, self.cpu_robot2)

        # state buffers
        self.robot1_q = np.zeros(self.robot1_q_num)
        self.robot2_q = np.zeros(self.robot2_q_num)
        self.dual_arm_joint_pos = np.concatenate([self.robot1_q, self.robot2_q])
        self.high_level_u = np.zeros_like(self.dual_arm_joint_pos)
        self.dual_arm_joint_vel = np.zeros_like(self.dual_arm_joint_pos)
        self.dual_arm_abs_feedback = self.dual.absolute_pose(self.dual_arm_joint_pos)

    def update_joint_states(self, q1=None, q2=None):
        if q1 is not None:
            self.robot1_q = np.asarray(q1, dtype=float).copy()
        if q2 is not None:
            self.robot2_q = np.asarray(q2, dtype=float).copy()
        self.dual_arm_joint_pos = np.concatenate((self.robot1_q, self.robot2_q))

    # ---- Legacy CLIK path (kept for comparison) ----
    def get_u(self, v_limit=None):
        rel_now = dq_to_vec8(self.dual.relative_pose(self.dual_arm_joint_pos))
        rel_des = dq_to_vec8(self.desire_rel_pose)
        rel_err = rel_des - rel_now

        Jrel = np.asarray(self.dual.relative_pose_jacobian(self.dual_arm_joint_pos))
        lam = 1e-3
        Jpinv = Jrel.T @ np.linalg.pinv(Jrel @ Jrel.T + lam * np.eye(8))
        v_rel = self.rel_gain * (Jpinv @ rel_err)

        N = np.eye(self.robot1_q_num + self.robot2_q_num) - Jpinv @ Jrel
        alpha = self.nullspace_gain
        u = v_rel + alpha * (N @ self.high_level_u)

        if v_limit is not None:
            u = np.clip(u, -float(v_limit), float(v_limit))

        self.dual_arm_joint_vel = u
        self.dual_arm_abs_feedback = self.dual.absolute_pose(self.dual_arm_joint_pos)


# ---------------------- Kinematics wrapper for DDP ----------------------
class DQKino:
    def __init__(self, dual: DQ_CooperativeDualTaskSpace, r1: DQ_SerialManipulatorDH, r2: DQ_SerialManipulatorDH, q1_dof, q2_dof, prefer_analytic_translation_jac=False):
        self.dual = dual
        self.r1 = r1
        self.r2 = r2
        self.n1 = q1_dof
        self.n2 = q2_dof
        self.nq = self.n1 + self.n2
        self.prefer_analytic_translation_jac = bool(prefer_analytic_translation_jac)

    # relative (8)
    def rel_vec8(self, q):
        return dq_to_vec8(self.dual.relative_pose(q))

    def rel_jac(self, q):
        J = np.asarray(self.dual.relative_pose_jacobian(q), dtype=float)
        assert J.shape == (8, self.nq), f"rel_jac shape {J.shape} != (8,{self.nq})"
        return J

    # absolute (8)
    def abs_vec8(self, q):
        return dq_to_vec8(self.dual.absolute_pose(q))

    def abs_jac(self, q):
        J = np.asarray(self.dual.absolute_pose_jacobian(q), dtype=float)
        assert J.shape == (8, self.nq), f"abs_jac shape {J.shape} != (8,{self.nq})"
        return J

    # absolute translation p(q) and Jp(q)
    def abs_translation_and_jac(self, q):
        """Absolute translation p (3,) and its jacobian Jp (3xN).
        Default: robust finite-difference; analytic is optional.
        """
        a = self.dual.absolute_pose(q)
        p4 = np.asarray(vec4(a.translation()), dtype=float).reshape(4)
        p = p4[1:4]
        # Prefer robust finite difference unless explicitly requested
        if self.prefer_analytic_translation_jac and _DQ_Kinematics is not None and hasattr(_DQ_Kinematics, "translation_jacobian"):
            try:
                J8 = np.asarray(self.dual.absolute_pose_jacobian(q), dtype=float)
                T4 = np.asarray(_DQ_Kinematics.translation_jacobian(J8, a), dtype=float)
                Jp = T4[1:4, :]
                return p, Jp
            except Exception:
                pass
        # Fallback: central finite differences with vec4
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


# ---------------------- Differential action model (kinematic) ----------------------
class DiffKinoDQ(croco.DifferentialActionModelAbstract):
    """
    x=q, u=qdot, xdot=u

    State cost:
      0.5*( (g_rel - rel(q))^T W_rel (g_rel - rel(q))
           + (g_abs - abs(q))^T W_abs (g_abs - abs(q)) )

    Velocity shaping:
      r_vabs = J_abs(q) u + beta_abs * (g_abs - abs(q))
      0.5 * r_vabs^T W_vabs r_vabs

    Control regularization: 0.5*w_u*||u||^2

    NOTE: Obstacles via VFI **after** DDP; relative-pose invariance enforced by equality projection.
    """

    def __init__(self, dq_kino: DQKino, weights, goals):
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
        data = croco.DifferentialActionDataAbstract(self)
        return data

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


# ---------------------- A tiny MPC/DDP wrapper ----------------------
class CrocoMPC:
    def __init__(self, dq_kino: DQKino, dt=0.05, N=30, weights=None):
        self.kino = dq_kino
        self.nq = dq_kino.nq
        self.dt = float(dt)
        self.N = int(N)

        if weights is None:
            weights = dict(w_rel=np.zeros(8), w_abs=np.ones(8) * 1e3, w_u=5e-2, w_vabs=5e2, beta_abs=4.0)

        zeros8 = np.zeros(8)
        self.running = []
        self.running_models = []

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

        self.problem = None
        self.solver = None
        self.xs = None
        self.us = None

    def set_goals(self, g_rel, g_abs):
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

    def _build_once(self, q0):
        if self.problem is None:
            self.problem = croco.ShootingProblem(q0, self.running, self.terminal)
            self.solver = croco.SolverDDP(self.problem)
            q0 = np.asarray(q0, dtype=np.float64).reshape(self.nq)
            self.xs = [q0.copy() for _ in range(self.N + 1)]
            self.us = [np.zeros(self.nq, dtype=np.float64) for _ in range(self.N)]

    def _ensure_vec_list(self, arr_list, n):
        out = []
        for a in arr_list:
            a = np.asarray(a, dtype=np.float64).reshape(n)
            out.append(np.ascontiguousarray(a))
        return out

    def _dls_abs_step(self, q, g_abs, v_max=None, lam=1e-3, gain=2.0):
        Ja = self.kino.abs_jac(q)
        e_abs = g_abs - self.kino.abs_vec8(q)
        JJ = Ja @ Ja.T + lam * np.eye(8)
        u = gain * (Ja.T @ np.linalg.solve(JJ, e_abs))
        if v_max is not None:
            u = np.clip(u, -float(v_max), float(v_max))
        return u

    def step(self, q0, g_rel, g_abs, max_iters=80, init_reg=1e-4, v_max=None):
        self._build_once(q0)
        self.problem.x0 = np.asarray(q0, dtype=np.float64).reshape(self.nq)
        self.set_goals(g_rel=g_rel, g_abs=g_abs)

        self.xs = self._ensure_vec_list(self.xs, self.nq)
        self.us = self._ensure_vec_list(self.us, self.nq)

        try:
            self.solver.solve(self.xs, self.us, int(max_iters), False, float(init_reg))
        except Exception:
            self.solver.solve(self.xs, self.us, 10, False, 1e-2)

        self.xs, self.us = self.solver.xs, self.solver.us
        u0 = np.array(self.us[0], dtype=float)

        if np.linalg.norm(u0) < 1e-6:
            u0 = self._dls_abs_step(self.problem.x0, g_abs, v_max=v_max, lam=1e-3, gain=2.0)
        return u0


# ---------------------- Equality projector (Lagrange multipliers) ----------------------
class EqualityProjector:
    """Project `u_des` onto equality set {u | J_eq u = 0} using Lagrange multipliers.

    Why: 保证相对位姿在速度层不变 (J_rel u = 0)。
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


# ---------------------- Vector Field Inequality projector ----------------------
class VFIProjector:
    """Dependency-free velocity-level inequality projector for **boxes & spheres**.

    Solve:  min ½||u - u_des||²  s.t.  A u >= b.
    If `J_eq` provided, restrict updates to Null(J_eq) and pre-project u to satisfy J_eq u = 0.
    """

    def __init__(self, kino: DQKino, obstacles, gamma: float = 6.0, activate_margin: float = 0.10, max_iters: int = 20):
        self.kino = kino
        self.gamma = float(gamma)         # class-K gain
        self.activate_margin = float(activate_margin)  # activate when s_eff < this (meters)
        self.max_iters = int(max_iters)
        self.set_obstacles(obstacles)
        self._eq = EqualityProjector(reg=1e-8)

    def set_obstacles(self, obstacles):
        self.obstacles = [] if obstacles is None else list(obstacles)

    def _row_for_obstacle(self, p: np.ndarray, Jp: np.ndarray, obs: dict) -> tuple[float, np.ndarray] | None:
        shape = obs.get("shape", "box")
        if shape == "sphere":
            c = np.asarray(obs.get("center", [0, 0, 0]), dtype=float).reshape(3)
            R = float(obs.get("radius", 0.05))
            buf = float(max(0.0, obs.get("buffer", 0.0)))
            d = p - c
            dist = float(np.linalg.norm(d))
            if dist < 1e-12:
                return None
            n = d / dist
            s = dist - R
            s_eff = s - buf
            if s_eff >= self.activate_margin:
                return None
            a = n @ Jp
            b = - self.gamma * s_eff
            return b, a
        else:  # box
            s, n = box_sdf_and_grad_world(p, obs)
            s_eff = s - float(max(0.0, obs.get("buffer", 0.0)))
            if s_eff >= self.activate_margin:
                return None
            a = n @ Jp
            b = - self.gamma * s_eff
            return b, a

    def _build_Ab(self, q):
        p, Jp = self.kino.abs_translation_and_jac(q)
        A_rows = []
        b_vals = []
        active_idx = []
        for idx, o in enumerate(self.obstacles):
            out = self._row_for_obstacle(p, Jp, o)
            if out is None:
                continue
            b, a = out
            A_rows.append(a)
            b_vals.append(b)
            active_idx.append(idx)
        if len(A_rows) == 0:
            A = np.zeros((0, self.kino.nq), dtype=float)
            b = np.zeros(0, dtype=float)
        else:
            A = np.vstack(A_rows)
            b = np.asarray(b_vals, dtype=float)
        return A, b, active_idx

    def project(self, q, u_des, v_limit=None, J_eq: np.ndarray | None = None, reg_eq: float = 1e-8):
        u = np.asarray(u_des, dtype=float).reshape(self.kino.nq).copy()
        if v_limit is not None:
            u = np.clip(u, -float(v_limit), float(v_limit))

        # Pre-project onto equality set if provided
        N = None
        if J_eq is not None and J_eq.size > 0:
            self._eq.reg = float(reg_eq)
            u = self._eq.project(J_eq, u)
            N = self._eq.nullspace_matrix(J_eq, self.kino.nq)

        A, b, active_idx = self._build_Ab(q)
        if A.shape[0] == 0:
            return u, 0

        # cyclic projections onto halfspaces {u | aᵢ·u >= bᵢ}
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
                        continue  # infeasible in Null(J_eq); skip
                    u += (gap / denom) * (N.T @ ai.T)
                else:
                    denom = float(ai @ ai) + 1e-12
                    u += (gap / denom) * ai

        if v_limit is not None:
            u = np.clip(u, -float(v_limit), float(v_limit))
        return u, len(active_idx)


# ---------------------- Numeric example (no ROS) ----------------------
def main(show_bullet=True, loop_playback=False):
    # absolute move target; relative keep
    desire_abs_position = np.array([0.5, 0.0, 0.5])
    desire_abs_pose = [- 0.009809, - 0.700866, - 0.008828, 0.713171, 0.03289, - 0.000662, - 0.283115, - 0.003703]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.001626, -0.002018, 0.28023, 0.00204]

    cfg_path = os.path.join(os.path.dirname(__file__), "ur3_and_ur3e.yaml")
    config = ConfigModule(cfg_path)
    llm = LowLevelModule(config, desire_abs_pose, desire_rel_pose, nullspace_gain=0.8)

    # initial joints (can be from config; here given explicitly)
    q1 = np.array([-1.8470081584056457, -2.7298507268179617, -0.6953932972144096, -1.508942496823497,  2.0236098037789576, -0.31532559669045146])
    q2 = np.array([ 1.842840084853423 , -0.48057750070854266,  0.8378998011418625 , -1.7586738880406665, -2.056763439048601 ,  3.415677557660605 ])
    llm.update_joint_states(q1, q2)

    # DQKino + Crocoddyl
    q1_dof = 6
    q2_dof = 6
    kino = DQKino(llm.dual, llm.cpu_robot1, llm.cpu_robot2, q1_dof, q2_dof, prefer_analytic_translation_jac=False)

    # weights (8D): relative small (we freeze it via equality), absolute larger; non-tiny w_u
    w_abs_rot = 5e2
    w_abs_trans = 0
    w_abs_vec = np.array([w_abs_rot, w_abs_rot, w_abs_rot, w_abs_rot, w_abs_trans, w_abs_trans, w_abs_trans, w_abs_trans], dtype=float)

    # --- define BOX obstacles (raw) ---
    # A thin rectangular wall near x≈0.5, spanning y,z with small thickness along x
    obstacles_raw = [
        {"shape": "box", "center": np.array([0.50,  0.0, 0.25]), "half": np.array([0.05, 0.05, 0.02]), "buffer": 0.01, "quat_xyzw": np.array([0, 0, 0, 1]), "color": [1, 0, 0, 0.35]},
        # 示例：带旋转的长方体（绕Z轴旋转约20°）
        # {"shape": "box", "center": np.array([0.45,  0.12, 0.25]), "half": np.array([0.010, 0.080, 0.060]), "buffer": 0.015,
        #  "quat_xyzw": np.array([0, 0, np.sin(np.deg2rad(20)/2), np.cos(np.deg2rad(20)/2)]), "color": [0, 0, 1, 0.35]},
    ]

    # effective list for visualization equals raw half (buffer只在VFI中生效)
    obstacles_eff = obstacles_raw

    mpc = CrocoMPC(
        kino,
        dt=0.15,
        N=15,
        weights={
            "w_rel": np.zeros(8),        # relative handled by equality freeze
            "w_abs": w_abs_vec,
            "w_vabs": np.array([2e1, 2e1, 2e1, 2e1, 2e1, 2e1, 2e1, 2e1], dtype=float),
            "beta_abs": 0.0,
            "w_u": 0,
        },
    )

    # goals (8D)
    g_rel = dq_to_vec8(llm.desire_rel_pose)
    g_abs = dq_to_vec8(llm.desire_abs_pose)

    # VFI projector (γ gain and activation margin are key knobs)
    vfi = VFIProjector(kino, obstacles_raw, gamma=8.0, activate_margin=0.10, max_iters=25)

    # sim params
    dt = 0.15
    T = 60.0
    steps = int(T / dt)
    v_max = 0.6
    print_interval = max(1, steps // 200)

    q_traj = []
    collision_events = []
    eq = EqualityProjector(reg=1e-8)

    print("Start sim (VFI[box] + EqualityFreeze):")
    for k in range(steps):
        q = np.concatenate([q1, q2])

        # 1) high-level from DDP
        u_ddp = mpc.step(q, g_rel=g_rel, g_abs=g_abs, max_iters=200, v_max=v_max)
        u_ddp = np.clip(u_ddp, -v_max, v_max)

        # 2) freeze relative pose: J_rel(q) u = 0 via Lagrange multipliers
        J_rel = kino.rel_jac(q)
        u_eq = eq.project(J_rel, u_ddp)

        # 3) VFI projection under equality: ensure Au>=b and keep J_rel u = 0
        u_cmd, n_active = vfi.project(q, u_eq, v_limit=v_max, J_eq=J_rel)

        # integrate
        q_next = q + dt * u_cmd
        q1, q2 = q_next[: llm.robot1_q_num], q_next[llm.robot1_q_num :]
        llm.update_joint_states(q1, q2)

        q_traj.append(q_next.copy())

        # diagnostics & collision check against boxes (without buffer)
        p_mid, _ = kino.abs_translation_and_jac(q_next)
        sds = signed_distances_to_obstacles(p_mid, obstacles_raw, use_buffer=False)
        tol = 1e-5  # 数值容差，避免边界抖动
        if sds.size > 0 and np.any(sds < -tol):
            for idx in np.where(sds < -tol)[0].tolist():
                collision_events.append({
                    "step": k,
                    "time": k * dt,
                    "obstacle": int(idx),
                    "depth": float(-sds[idx]),
                    "p": p_mid.copy(),
                })
            if (k % max(1, steps // 50)) == 0:
                print(f"[COLLISION] step={k} t={k*dt:.2f}s boxes={np.where(sds < -tol)[0].tolist()} depths={(-sds[sds < -tol]).round(4).tolist()}")

        if (k % print_interval) == 0 or k == steps - 1:
            rel_now = dq_to_vec8(llm.dual.relative_pose(llm.dual_arm_joint_pos))
            abs_now = dq_to_vec8(llm.dual.absolute_pose(llm.dual_arm_joint_pos))
            e_rel = np.linalg.norm(rel_now - g_rel)
            e_abs = np.linalg.norm(abs_now - g_abs)
            try:
                p, _ = kino.abs_translation_and_jac(q)
                d0 = signed_distances_to_obstacles(p, obstacles_eff, use_buffer=True)[0]
                d0_str = f" s_to_box0_eff={d0:+.3f}m"
            except Exception:
                d0_str = ""
            print(
                f"[{k:4d}/{steps}]  ||rel_err||={e_rel: .3e}   ||abs_err||={e_abs: .3e}   "
                f"|u_ddp|={np.linalg.norm(u_ddp):.3f}   |u_cmd|={np.linalg.norm(u_cmd):.3f}   "
                f"active_vfi={n_active}{d0_str}"
            )

    print("Done.")

    # summary for collisions
    if collision_events:
        worst = max(collision_events, key=lambda e: e["depth"])
        print(f"[SUMMARY] Collisions: {len(collision_events)} events; worst depth={worst['depth']:.4f}m at step={worst['step']} t={worst['time']:.2f}s")
        # optional CSV dump for analysis
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, "collision_events.csv")
            import csv
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "time", "obstacle", "depth", "px", "py", "pz"])
                for e in collision_events:
                    w.writerow([e["step"], e["time"], e["obstacle"], e["depth"], e["p"][0], e["p"][1], e["p"][2]])
            print(f"[SUMMARY] Collision log saved to {csv_path}")
        except Exception as _:
            pass

    if show_bullet:
        if not _HAVE_BULLET:
            print("[WARN] PyBullet not installed. `pip install pybullet` to enable playback.")
            return
        # Draw boxes (visual only); buffer 在VFI中处理，不在可视化中膨胀
        obstacles_draw = [
            {k: v for k, v in o.items() if k in ("shape", "center", "half", "quat_xyzw", "color")}
            for o in obstacles_raw
        ]
        viewer = BulletPlayback(urdf_path="model/dual_arm_model/dual_arm_model.urdf", timestep=1.0 / 60.0, obstacles=obstacles_draw)
        if collision_events:
            viewer.draw_points([e["p"] for e in collision_events], radius=0.01, rgba=[1, 0, 0, 1])
        rate_hz = int(max(1.0 / dt, 5))
        print(f"[BulletPlayback] Replaying {len(q_traj)} frames at ~{rate_hz} FPS. Press Ctrl+C to exit.")
        viewer.play(q_traj, rate_hz=rate_hz, loop=loop_playback)


if __name__ == "__main__":
    main(show_bullet=True, loop_playback=False)
