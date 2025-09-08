
import os
import math
import time
import numpy as np
import pybullet as p
import pybullet_data

# -------------------- User-configurable knobs --------------------
SIM_TOTAL_TIME = 12.0      # total simulated time [s]
GUI = True                 # True = p.GUI, False = p.DIRECT
BULLET_TIMESTEP = 1.0/240. # Bullet internal dt [s]

HL_DT = 0.030              # "CEM" high-level plan refresh period [s]  (30ms)
LL_DT = 0.008              # low-level update/integration period [s]   (8ms)

# joint limits (fallback if URDF has no limits; tune as needed)
Q_MIN = np.array([-3.0]*6 + [-3.0]*6, dtype=np.float32)
Q_MAX = np.array([ 3.0]*6 + [ 3.0]*6, dtype=np.float32)
DQ_MIN = np.array([-0.6]*6 + [-0.6]*6, dtype=np.float32)  # rad/s
DQ_MAX = np.array([ 0.6]*6 + [ 0.6]*6, dtype=np.float32)
DDQ_MIN = np.array([-0.6]*6 + [-0.6]*6, dtype=np.float32) # rad/s^2
DDQ_MAX = np.array([ 0.6]*6 + [ 0.6]*6, dtype=np.float32)

# initial posture from your ROS-based script (UR3 + UR3e)
Q0_LEFT  = np.array([-1.8470081584056457, -2.7298507268179617, -0.6953932972144096, -1.508942496823497,  2.0236098037789576, -0.31532559669045146], dtype=np.float32)
Q0_RIGHT = np.array([ 1.842840084853423 , -0.48057750070854266,  0.8378998011418625, -1.7586738880406665, -2.056763439048601 ,  3.415677557660605 ], dtype=np.float32)
Q0 = np.concatenate([Q0_LEFT, Q0_RIGHT]).astype(np.float32)

# planner horizon (seconds) and resolution: we store velocities per LL_DT step
PLAN_HORIZON_SEC = 0.5
PLAN_STEPS = int(round(PLAN_HORIZON_SEC / LL_DT))

# simple moving target for demo (per-joint sinusoids)
TARGET_AMPL = np.deg2rad(10.0) # +/- 10 deg
TARGET_FREQ = 0.10             # Hz

# ---------------------- Helper functions -------------------------
def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def find_urdf_path():
    """Try a robust sequence of URDF paths."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "model", "dual_arm_model", "dual_arm_model.urdf"),
        "model/dual_arm_model/dual_arm_model.urdf",  # fallback to repo-relative
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # still return last candidate; Bullet will error visibly if missing
    return candidates[-1]

def get_revolute_joint_indices(body_id):
    idx = []
    for j in range(p.getNumJoints(body_id)):
        ji = p.getJointInfo(body_id, j)
        joint_type = ji[2]
        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            idx.append(j)
    return idx

# ---------------------- Dummy "CEM" planner ----------------------
class DummyCEMPlanner:
    """
    A lightweight, stable stand-in:
    - Plans joint velocities to move q towards a time-varying target q_target(t).
    - Adds clamps on velocity and acceleration (slew limits).
    - Returns an array of shape [PLAN_STEPS, nq].
    """
    def __init__(self, nq, ll_dt, dq_min, dq_max, ddq_min, ddq_max):
        self.nq = nq
        self.ll_dt = float(ll_dt)
        self.dq_min = dq_min.astype(np.float32)
        self.dq_max = dq_max.astype(np.float32)
        self.ddq_min = ddq_min.astype(np.float32)
        self.ddq_max = ddq_max.astype(np.float32)
        self._last_u = np.zeros(nq, dtype=np.float32)  # for slew limit

        # gains for a safe PD-to-target
        self.kp = 2.0   # rad/s per rad error
        self.kd = 0.1

    def target(self, q, t):
        # simple sinusoidal target around initial q for both arms
        phase = 2.0 * math.pi * TARGET_FREQ * t
        offset = TARGET_AMPL * np.sin(phase + np.arange(self.nq)*0.3)  # phase stagger across joints
        return q*0.0 + offset  # relative target from zero; you can swap to absolute target if desired

    def compute_plan(self, q_now, dq_now, t_now):
        plan = np.zeros((PLAN_STEPS, self.nq), dtype=np.float32)
        u_prev = self._last_u.copy()

        q_sim = q_now.copy()
        dq_sim = dq_now.copy()

        for k in range(PLAN_STEPS):
            tk = t_now + k*self.ll_dt
            q_ref = self.target(Q0, tk)  # trajectory relative to origin
            err = (q_ref - q_sim)  # position error
            # PD in velocity space
            u_des = self.kp * err - self.kd * dq_sim

            # velocity clamp
            u_des = clamp(u_des, self.dq_min, self.dq_max)

            # slew (acc) limit relative to last planned u
            du = clamp(u_des - u_prev, self.ddq_min * self.ll_dt, self.ddq_max * self.ll_dt)
            u = u_prev + du

            plan[k, :] = u

            # internal rollout (Euler)
            q_sim = clamp(q_sim + u * self.ll_dt, Q_MIN, Q_MAX)
            dq_sim = u.copy()
            u_prev = u.copy()

        # store last
        self._last_u = plan[-1].copy()
        return plan

# -------------------------- Main sim -----------------------------
def main():
    # Bullet setup
    cid = p.connect(p.GUI if GUI else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(BULLET_TIMESTEP)
    plane_id = p.loadURDF("plane.urdf")

    urdf_path = find_urdf_path()
    robot_id = p.loadURDF(urdf_path, basePosition=[0,0,0], useFixedBase=True)
    joint_ids = get_revolute_joint_indices(robot_id)
    if len(joint_ids) < 12:
        print(f"[Warn] Found {len(joint_ids)} actuated joints, expected 12. The script will control whatever it finds.")

    nq = len(joint_ids)
    # initialize state
    q = Q0.copy()
    if nq != len(q):
        # resize if URDF joint count differs (robustness)
        q = np.resize(q, nq).astype(np.float32)
    dq = np.zeros(nq, dtype=np.float32)

    # write initial pose
    for j, qi in zip(joint_ids, q):
        p.resetJointState(robot_id, j, float(qi))

    # Disable default motors to avoid interference
    for j in joint_ids:
        p.setJointMotorControl2(robot_id, j, controlMode=p.VELOCITY_CONTROL, force=0.0)

    # planner and buffers
    planner = DummyCEMPlanner(nq, LL_DT, DQ_MIN[:nq], DQ_MAX[:nq], DDQ_MIN[:nq], DDQ_MAX[:nq])
    plan = np.zeros((PLAN_STEPS, nq), dtype=np.float32)
    plan_step_idx = PLAN_STEPS  # trigger immediate replan

    # simulation loop
    t = 0.0
    sim_steps = 0
    substeps_per_ll = max(1, int(round(LL_DT / BULLET_TIMESTEP)))

    TOL = 1e-9
    T_END = SIM_TOTAL_TIME

    last_hl_time = -1e9
    while t < T_END - TOL:
        # high-level refresh each HL_DT
        if (t - last_hl_time) >= HL_DT - 1e-12:
            plan = planner.compute_plan(q_now=q, dq_now=dq, t_now=t)
            plan_step_idx = 0
            last_hl_time = t

        # low-level apply first element of plan
        if plan_step_idx >= PLAN_STEPS:
            # plan exhausted (corner case if HL_DT > PLAN_HORIZON_SEC); hold zeros
            u = np.zeros(nq, dtype=np.float32)
        else:
            u = plan[plan_step_idx]
            plan_step_idx += 1

        # integrate (Euler) with clamps
        dq_des = clamp(u, DQ_MIN[:nq], DQ_MAX[:nq])
        q_next = clamp(q + dq_des * LL_DT, Q_MIN[:nq], Q_MAX[:nq])

        # write to Bullet
        for j, qj in zip(joint_ids, q_next):
            p.resetJointState(robot_id, j, float(qj))

        # advance Bullet deterministically: substeps for the LL_DT window
        for _ in range(substeps_per_ll):
            p.stepSimulation()
        sim_steps += substeps_per_ll

        # commit state
        dq = dq_des
        q = q_next
        t += LL_DT

    print(f"Done. Simulated {T_END:.3f}s using {sim_steps} Bullet substeps.")

if __name__ == "__main__":
    main()