import torch
from dq_torch import  norm, haminus8, conj,dq_mult, rel_abs_pose_rel_jac, rel_dir_rect, project_q_by_constraint_jacobian
import numpy as np
import cProfile
from dqrobotics.robot_modeling import DQ_SerialManipulatorMDH, DQ_CooperativeDualTaskSpace
from dqrobotics import DQ
import time



########################################################### 
################### CPU MODEL Define #####################
##########################################################    
# cpu robot1
@torch.jit.script
def get_rel_jacobian_null(jac_batch: torch.Tensor):
    # 形状：jac_batch [B, m, n]，n = robot1_q_num + robot2_q_num
    B, m, n = jac_batch.shape  # 运行时检查可保留/删除
    # assert B == batch_size and n == robot1_q_num + robot2_q_num

    # 保持你原来的 float64 计算逻辑，但避免多余拷贝
    J = jac_batch.to(torch.float64, copy=False)
    device = J.device

    # 单位阵用 expand（零拷贝），避免 repeat 的真实复制
    batch_i  = torch.eye(n, dtype=torch.float64, device=device).unsqueeze(0).expand(B, n, n)
    batch_eps = (1e-16) * torch.eye(m, dtype=torch.float32, device=device).unsqueeze(0).expand(B, m, m)

    # 按原逻辑计算
    J_t   = J.transpose(-2, -1).contiguous()  # 保证后续 matmul 的内存布局更友好
    J_J_t = torch.matmul(J, J_t)
    J_pinv = torch.matmul(J_t, torch.inverse(J_J_t + batch_eps))
    J_pinv_J = torch.matmul(J_pinv, J)
    rel_jacobian_null = batch_i - J_pinv_J
    return rel_jacobian_null.to(torch.float32)

robot1_config_dh_mat = np.array([[0.0, 0.333,   0.0,        0.0, 0],
                                [0.0, 0.0,     0.0,    -1.5708, 0],
                                [0.0, 0.316,   0.0,     1.5708, 0],
                                [0.0, 0.0,     0.0825,  1.5708, 0],
                                [0.0, 0.384,  -0.0825, -1.5708, 0],
                                [0.0, 0.0,     0.0,     1.5708, 0],
                                [0.0, 0.0,   0.088,   1.5708, 0]])
robot1_dh_mat =  robot1_config_dh_mat.T

cpu_robot1 = DQ_SerialManipulatorMDH(robot1_dh_mat)
cpu_robot1.set_base_frame(DQ([1.0, 0, 0, 0, 0.0, 0.0, -0.175, 0.0]))
cpu_robot1.set_effector(DQ([1.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0]))
cpu_robot1.set_reference_frame(DQ([1.0, 0, 0, 0, 0.0, 0.0, -0.175, 0.0]))
# cpu robot2
robot2_config_dh_mat = np.array([[0.0, 0.333,   0.0,        0.0, 0],
                                [0.0, 0.0,     0.0,    -1.5708, 0],
                                [0.0, 0.316,   0.0,     1.5708, 0],
                                [0.0, 0.0,     0.0825,  1.5708, 0],
                                [0.0, 0.384,  -0.0825, -1.5708, 0],
                                [0.0, 0.0,     0.0,     1.5708, 0],
                                [0.0, 0.0,   0.088,   1.5708, 0]])
robot2_dh_mat =  robot2_config_dh_mat.T
cpu_robot2 = DQ_SerialManipulatorMDH(robot2_dh_mat)
cpu_robot2.set_base_frame(DQ([1.0, 0, 0, 0, 0.0, 0.0, 0.175, 0.0]))
cpu_robot2.set_effector(DQ([1.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0]))
cpu_robot2.set_reference_frame(DQ([1.0, 0, 0, 0, 0.0, 0.0, 0.175, 0.0]))
cpu_dq_dual_arm_model = DQ_CooperativeDualTaskSpace(cpu_robot1, cpu_robot2)
a = cpu_dq_dual_arm_model.relative_pose([0, -1.0471, 0, -2.6178,  1.5707,  1.5707, 0.7853, 0, -1.0471, 0, -2.6178, -1.5707, 1.5707, 0.7853])
q = [0, -1.0471, 0, -2.6178,  1.5707,  1.5707, 0.7853, 0, -1.0471, 0, -2.6178, -1.5707, 1.5707, 0.7853]

########################################################### 
################### GPU MODEL Define #####################
##########################################################    
batch_size = 1
dh_matrix1 = torch.tensor([
    [0.0, 0.333,   0.0,        0.0, 0],
    [0.0, 0.0,     0.0,    -1.5708, 0],
    [0.0, 0.316,   0.0,     1.5708, 0],
    [0.0, 0.0,     0.0825,  1.5708, 0],
    [0.0, 0.384,  -0.0825, -1.5708, 0],
    [0.0, 0.0,     0.0,     1.5708, 0],
    [0.0, 0.0,   0.088,   1.5708, 0]
], dtype=torch.float32, device= "cuda:0")
dh_matrix2 = torch.tensor([
    [0.0, 0.333,   0.0,        0.0, 0],
    [0.0, 0.0,     0.0,    -1.5708, 0],
    [0.0, 0.316,   0.0,     1.5708, 0],
    [0.0, 0.0,     0.0825,  1.5708, 0],
    [0.0, 0.384,  -0.0825, -1.5708, 0],
    [0.0, 0.0,     0.0,     1.5708, 0],
    [0.0, 0.0,   0.088,   1.5708, 0]
], dtype=torch.float32, device= "cuda:0") 
dual_arm_joint_pos = [0, -1.0471, 0, -2.6178,  1.5707,  1.5707, 0.7853, 0, -1.0471, 0, -2.6178, -1.5707, 1.5707, 0.7853]
batch_robot1_base = torch.tensor([1.0, 0, 0, 0, 0.0, 0.0, -0.175, 0.0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_robot2_base = torch.tensor([1.0, 0, 0, 0, 0.0, 0.0, 0.175, 0.0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_robot1_effector = torch.tensor([1.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_robot2_effector = torch.tensor([1.0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
q_vec1 = torch.tensor([0, -1.0471, 0, -2.6178,  1.5707,  1.5707, 0.7853], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1) 
q_vec2 = torch.tensor([0, -1.0471, 0, -2.6178, -1.5707, 1.5707, 0.7853], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
q_vec3 = torch.tensor([0, -1, 0, -2, -1.7, 1.4707, 0.7853], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
dh_matrix1 = dh_matrix1.reshape(-1)
dh_matrix2 = dh_matrix2.reshape(-1)
desire_line_d = [0,0,0,1]
desire_quat_line_ref = [0, -0.011682, 0.003006, -0.999927]
batch_line_d = torch.tensor(desire_line_d, dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_quat_line_ref = torch.tensor(desire_quat_line_ref, dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
start_time = time.time()
first_bacth_rel_pos, bacth_abs_pos, bacth_rel_jacobian,  batch_abs_position, batch_angle = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q_vec1, q_vec2,
                         batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)
first_bacth_rel_pos = torch.tensor([9.63267947e-05,  7.07173586e-01, -7.07039957e-01, -9.63267808e-05, 3.51225857e-06, 2.47457994e-01, 2.47504763e-01, -9.62904111e-07], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
end_time = time.time()
print("Time taken for rel_abs_pose_rel_abs_jac: ", end_time - start_time)
batch_ident_dq = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
temp_dq = dq_mult(conj(first_bacth_rel_pos), batch_ident_dq)
error_dq = batch_ident_dq - temp_dq 
error_dq = error_dq.reshape(batch_size, 8, 1)
bacth_rel_pos, bacth_abs_pos, bacth_rel_jacobian,  batch_abs_position, batch_angle = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q_vec1, q_vec2,
                         batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)
print(bacth_rel_pos)
q1,q2 = project_q_by_constraint_jacobian(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q_vec1, q_vec3, first_bacth_rel_pos,
                        7, 7, 1, 1)
print(first_bacth_rel_pos)
bacth_rel_pos, bacth_abs_pos, bacth_rel_jacobian,  batch_abs_position, batch_angle = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q1, q2,
                         batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)


print(bacth_rel_pos)
torch.cuda.synchronize()
start_time = time.time()
q1,q2 = project_q_by_constraint_jacobian(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q_vec1, q_vec3, first_bacth_rel_pos,
                         7, 7, 1, 1)
torch.cuda.synchronize()
end_time = time.time()
print("Time taken for project_q_by_constraint_jacobian: ", end_time - start_time)


# start_time = time.time()
# bacth_rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
#                          batch_robot1_base,  batch_robot2_base, 
#                          batch_robot1_effector, batch_robot2_effector, 
#                          q_vec1, q_vec2,
#                          batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)
# end_time = time.time()
# print("Time taken for rel_abs_pose_rel_abs_jac: ", end_time - start_time)
# start_time = time.time()
# bacth_rel_pos, bacth_abs_pos, batch_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
#                          batch_robot1_base,  batch_robot2_base, 
#                          batch_robot1_effector, batch_robot2_effector, 
#                          q_vec1, q_vec2,
#                          batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)
# end_time = time.time()
print("Time taken for rel_abs_pose_rel_abs_jac: ", end_time - start_time)
get_rel_jacobian_null(bacth_rel_jacobian)

get_rel_jacobian_null(bacth_rel_jacobian)
# #d8mat= batch *8 *8 
# # d8=eye[-1,1,1,1,-1,1,1,1]
# start_time = time.time()
# bacth_rel_pos_vec = bacth_rel_pos.reshape(batch_size, 8, 1)
# batch_d8_mat = torch.diag(torch.tensor([1, -1, -1, -1, 1, -1, -1, -1], dtype=torch.float32, device="cuda:0")).unsqueeze(0).repeat(batch_size, 1, 1)
# batch_hm8_mat = haminus8(batch_ident_dq)
# N = batch_hm8_mat@batch_d8_mat@batch_rel_jacobian
# N_T = N.transpose(1, 2)
# a = N_T@error_dq
# end_time = time.time()
# print("Time taken forN_T@bacth_rel_pos_vec ", end_time - start_time)
# print("N_T@bacth_rel_pos_vec:", a)
# start_time = time.time()
# a = rel_dir_rect(batch_ident_dq, bacth_rel_pos, batch_rel_jacobian, 7, 7)
# end_time = time.time()
# print("Time taken for rel_dir_rect: ", end_time - start_time)
# print("rel_dir_rect result:", a)

# start_time = time.time()
# a = rel_dir_rect(batch_ident_dq, batch_ident_dq, batch_rel_jacobian, 7, 7)
# end_time = time.time()
# print("Time taken for rel_dir_rect: ", end_time - start_time)

bacth_rel_pos, bacth_abs_pos, bacth_rel_jacobian,  batch_abs_position, batch_angle = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q1, q_vec3,
                         batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)
torch.cuda.synchronize()
start_time = time.time()
get_rel_jacobian_null(bacth_rel_jacobian)
torch.cuda.synchronize()
end_time = time.time()
print("Time taken for get_rel_jacobian_null: ", end_time - start_time)
# # - 0.241632 + 0.900844i + 0.323735j + 0.15903k + E*( - 0.162267 - 0.178889i + 0.441044j - 0.13104k)
# ########################################################### 
# ######################  WARM UP ###########################
# ###########################################################   
# for i in range(10):
#     rel_abs_pose_rel_abs_jac(dh_matrix1, dh_matrix2,
#                          batch_robot1_base,  batch_robot2_base, 
#                          batch_robot1_effector, batch_robot2_effector, 
#                          q_vec1, q_vec2,
#                          batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)

# for i in range(10):
#     print(cpu_dq_dual_arm_model.relative_pose(dual_arm_joint_pos))
#     cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos)
#     cpu_dq_dual_arm_model.relative_pose_jacobian(dual_arm_joint_pos)
#     cpu_dq_dual_arm_model.absolute_pose_jacobian(dual_arm_joint_pos)

# ########################################################### 
# ###################### START Test #########################
# ########################################################### 
# for i in range(100):
#     start_time = time.time()
#     rel_abs_pose_rel_abs_jac(dh_matrix1, dh_matrix2,
#                          batch_robot1_base,  batch_robot2_base, 
#                          batch_robot1_effector, batch_robot2_effector, 
#                          q_vec1, q_vec2,
#                          batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)
#     end_time = time.time()
#     print("GPU Time taken: ", end_time - start_time)
# for i in range(100):
#     start_time = time.time()
#     for i in range(20000):
#         cpu_dq_dual_arm_model.relative_pose(dual_arm_joint_pos)
#         cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos)
#         cpu_dq_dual_arm_model.relative_pose_jacobian(dual_arm_joint_pos)
#         cpu_dq_dual_arm_model.absolute_pose_jacobian(dual_arm_joint_pos)
#     end_time = time.time()
#     print("CPU Time taken: ", end_time - start_time)
import math

def _quat_conj(q):  # q: [...,4]
    return torch.stack([q[...,0], -q[...,1], -q[...,2], -q[...,3]], dim=-1)

def _quat_mul(a, b):  # a,b: [...,4] -> [...,4]
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], dim=-1)

@torch.no_grad()
def evaluate_projection_precision(
    dh_matrix1, dh_matrix2,
    base1, base2, eff1, eff2,
    q1, q2,
    target_rel_dq,              # 形状 [B,8]
    line_d, quat_line_ref,      # 你现有的任务参数
    n1=7, n2=7, b1=1, b2=1,
    verbose=True
):
    """
    评测 current(q1,q2) 相对 target_rel_dq 的投影精度。
    返回一个 dict（每个 batch 一行）。
    """
    # 当前相对位姿与雅可比
    cur_rel, _, cur_rel_J, _, _ = rel_abs_pose_rel_jac(
        dh_matrix1, dh_matrix2,
        base1, base2, eff1, eff2,
        q1, q2, line_d, quat_line_ref,
        n1, n2, b1, b2
    )  # cur_rel: [B,8], cur_rel_J: [B,8,n1+n2]

    B = cur_rel.shape[0]
    device = cur_rel.device
    ident = torch.tensor([1,0,0,0,0,0,0,0], dtype=cur_rel.dtype, device=device).expand(B, -1)

    # Δ = conj(target) ⊗ current
    delta = dq_mult(conj(target_rel_dq), cur_rel)  # [B,8]

    # 处理双四元数“符号模二”问题：若 Δ 的实部为负，翻转号以保持与 I 同半球
    sign = torch.where(delta[:, 0:1] < 0, -torch.ones_like(delta[:, 0:1]), torch.ones_like(delta[:, 0:1]))
    delta = delta * sign

    # 角度误差：r = Δ的旋转四元数部分
    r = delta[:, :4]
    w = torch.clamp(r[:, 0], -1.0, 1.0)
    ang_rad = 2.0 * torch.arccos(w)                       # [B]
    ang_deg = ang_rad * (180.0 / math.pi)

    # 平移误差：t = 2 * (d * conj(r)).vec
    d = delta[:, 4:]
    t_quat = _quat_mul(d, _quat_conj(r))                  # [B,4]
    t_vec = 2.0 * t_quat[:, 1:4]                          # [B,3]，单位与 DH 一致（米）
    trans_m = torch.linalg.norm(t_vec, dim=-1)            # [B]
    trans_mm = trans_m * 1000.0

    # 双四元数残差范数 ‖Δ - I‖₂
    dq_residual = torch.linalg.vector_norm(delta - ident, dim=-1)  # [B]

    # 线性化约束残差：a = (H^- D8 J)^T * e，e = I - Δ
    batch_hm8 = haminus8(ident)                           # [B,8,8]
    d8 = torch.tensor([1,-1,-1,-1, 1,-1,-1,-1], dtype=cur_rel.dtype, device=device)
    batch_d8 = torch.diag(d8).unsqueeze(0).expand(B, -1, -1)  # [B,8,8]
    N = batch_hm8 @ batch_d8 @ cur_rel_J                  # [B,8,nq]
    e = (ident - delta).unsqueeze(-1)                     # [B,8,1]
    a = torch.transpose(N, 1, 2) @ e                      # [B,nq,1]
    lin_residual = torch.linalg.vector_norm(a.squeeze(-1), dim=-1)  # [B]

    metrics = {
        "rot_err_deg": ang_deg,
        "trans_err_mm": trans_mm,
        "dq_residual_norm": dq_residual,
        "linearized_residual_norm": lin_residual
    }

    if verbose:
        for b in range(B):
            print(f"[Projection metrics | batch {b}] "
                  f"rot_err = {metrics['rot_err_deg'][b].item():.6f} deg | "
                  f"trans_err = {metrics['trans_err_mm'][b].item():.6f} mm | "
                  f"dq_residual = {metrics['dq_residual_norm'][b].item():.3e} | "
                  f"linearized_residual = {metrics['linearized_residual_norm'][b].item():.3e}")
    return metrics
# 评测一次投影精度（针对 q1, q2 相对 first_bacth_rel_pos）
_ = evaluate_projection_precision(
    dh_matrix1, dh_matrix2,
    batch_robot1_base, batch_robot2_base,
    batch_robot1_effector, batch_robot2_effector,
    q1, q2,
    first_bacth_rel_pos,                 # 作为 target
    batch_line_d, batch_quat_line_ref,   # 你的任务参数
    7, 7, 1, 1,
    verbose=True
)


N = 20
rot_list, trans_list = [], []
for i in range(N):
    q2_rand = (q_vec3 + 0.5*torch.randn_like(q_vec3)).clamp(-3.0, 3.0)  # 简单扰动
    qq1, qq2 = project_q_by_constraint_jacobian(
        dh_matrix1, dh_matrix2,
        batch_robot1_base, batch_robot2_base,
        batch_robot1_effector, batch_robot2_effector,
        q_vec1, q2_rand, first_bacth_rel_pos,
        7, 7, 1, 1
    )
    m = evaluate_projection_precision(
        dh_matrix1, dh_matrix2,
        batch_robot1_base, batch_robot2_base,
        batch_robot1_effector, batch_robot2_effector,
        qq1, qq2,
        first_bacth_rel_pos,
        batch_line_d, batch_quat_line_ref,
        7, 7, 1, 1,
        verbose=False
    )
    rot_list.append(m["rot_err_deg"])
    trans_list.append(m["trans_err_mm"])

rot_err = torch.cat(rot_list).mean().item()
trans_err = torch.cat(trans_list).mean().item()
print(f"[Projection metrics | sweep N={N}] mean_rot_err = {rot_err:.6f} deg | mean_trans_err = {trans_err:.6f} mm")