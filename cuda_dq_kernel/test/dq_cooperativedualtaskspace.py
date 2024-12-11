import torch
from torch import nn
import numpy as np
from dq_mult_extension import dq_mult, dq_sqrt, Ad, conj, hamiplus8, haminus8
from dq_torch import C8
from time import time

@torch.jit.script
def _dh2dq(dh1:torch.Tensor, dh2:torch.Tensor, theta1:torch.Tensor, theta2:torch.Tensor, ith:int):
    half_theta1 = dh1[:, 0, ith] / 2.0 + theta1 / 2.0
    d1 = dh1[:, 1, ith]
    a1 = dh1[:, 2, ith]
    half_alpha1 = dh1[:, 3, ith] / 2.0  # 优化点：只计算一次除法
    sin_half_theta1, cos_half_theta1 = torch.sin(half_theta1), torch.cos(half_theta1)
    sin_half_alpha1, cos_half_alpha1 = torch.sin(half_alpha1), torch.cos(half_alpha1)
    # 优化：预先计算乘积
    a_cos_half_alpha1 = a1 * cos_half_alpha1
    d_sin_half_alpha1 = d1 * sin_half_alpha1
    a_sin_half_alpha1 = a1 * sin_half_alpha1
    d_cos_half_alpha1 = d1 * cos_half_alpha1
    # 计算四元数的各个部分
    q11 = cos_half_alpha1 * cos_half_theta1
    q21 = sin_half_alpha1 * cos_half_theta1
    q31 = sin_half_alpha1 * sin_half_theta1
    q41 = cos_half_alpha1 * sin_half_theta1
    q51 = -(a_sin_half_alpha1 * cos_half_theta1 + d_cos_half_alpha1 * sin_half_theta1) / 2.0
    q61 =  (a_cos_half_alpha1 * cos_half_theta1 - d_sin_half_alpha1 * sin_half_theta1) / 2.0
    q71 =  (a_cos_half_alpha1 * sin_half_theta1 + d_sin_half_alpha1 * cos_half_theta1) / 2.0
    q81 =  (d_cos_half_alpha1 * cos_half_theta1 - a_sin_half_alpha1 * sin_half_theta1) / 2.0
    # 利用堆叠（stack）来组合结果
    combined2 = torch.stack((q11, q21, q31, q41, q51, q61, q71, q81), dim=1)
    half_theta2 = dh2[:, 0, ith] / 2.0 + theta2 / 2.0
    d2 = dh2[:, 1, ith]
    a2 = dh2[:, 2, ith]
    half_alpha2 = dh2[:, 3, ith] / 2.0  # 优化点：只计算一次除法
    sin_half_theta2, cos_half_theta2 = torch.sin(half_theta2), torch.cos(half_theta2)
    sin_half_alpha2, cos_half_alpha2 = torch.sin(half_alpha2), torch.cos(half_alpha2)
    # 优化：预先计算乘积
    a_cos_half_alpha2 = a2 * cos_half_alpha2
    d_sin_half_alpha2 = d2 * sin_half_alpha2
    a_sin_half_alpha2 = a2 * sin_half_alpha2
    d_cos_half_alpha2 = d2 * cos_half_alpha2
    # 计算四元数的各个部分
    q12 = cos_half_alpha2 * cos_half_theta2
    q22 = sin_half_alpha2 * cos_half_theta2
    q32 = sin_half_alpha2 * sin_half_theta2
    q42 = cos_half_alpha2 * sin_half_theta2
    q52 = -(a_sin_half_alpha2 * cos_half_theta2 + d_cos_half_alpha2 * sin_half_theta2) / 2.0
    q62 =  (a_cos_half_alpha2 * cos_half_theta2 - d_sin_half_alpha2 * sin_half_theta2) / 2.0
    q72 =  (a_cos_half_alpha2 * sin_half_theta2 + d_sin_half_alpha2 * cos_half_theta2) / 2.0
    q82 =  (d_cos_half_alpha2 * cos_half_theta2 - a_sin_half_alpha2 * sin_half_theta2) / 2.0
    # 利用堆叠（stack）来组合结果
    combined1 = torch.stack((q12, q22, q32, q42, q52, q62, q72, q82), dim=1)
    return combined1, combined2

#@torch.jit.script
def raw_fkm(dh1:torch.Tensor, dh2:torch.Tensor, theta1:torch.Tensor, theta2:torch.Tensor, to_ith_link:int, batchsize:int):
    q_init1 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], device=dh1.device, dtype=dh1.dtype).repeat(batchsize, 1)
    q_init2 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], device=dh1.device, dtype=dh1.dtype).repeat(batchsize, 1)
    for i in range(to_ith_link + 1):
        a1, a2 = _dh2dq(dh1, dh2, theta1[:,i], theta2[:,i], i)
        q_init1 = dq_mult(q_init1, a1)
        q_init2 = dq_mult(q_init2, a2)
    return q_init1, q_init2

#@torch.jit.script
def rel_abs_pose(dh1:torch.Tensor, dh2:torch.Tensor, theta1:torch.Tensor, theta2:torch.Tensor, batchsize:int, joint_num:int):
    q_init1 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], device=dh1.device, dtype=dh1.dtype).repeat(batchsize, 1)
    q_init2 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], device=dh1.device, dtype=dh1.dtype).repeat(batchsize, 1)
    for i in range(joint_num+1):
        a1, a2 = _dh2dq(dh1, dh2, theta1[:,i], theta2[:,i], i)
        q_init1 = dq_mult(q_init1, a1)
        q_init2 = dq_mult(q_init2, a2)
    return dq_mult(conj(q_init2), q_init1), dq_mult(q_init2, dq_sqrt(q_init1))

# 这里与dqrobotics不一致 dqrobotics w是通过dh矩阵计算的，这里直接给定 默认旋转   
#@torch.jit.script
def rel_abs_pose_rel_jac(dh1:torch.Tensor, dh2:torch.Tensor,
                         base1:torch.Tensor, base2:torch.Tensor,
                         effector1:torch.Tensor, effector2:torch.Tensor,
                         theta1:torch.Tensor, theta2:torch.Tensor, 
                         batchsize:int, joint_num:int):
    # start_time = time()
    identity_tensor = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], device=dh1.device, dtype=dh1.dtype).repeat(batchsize, 1)
    x_effector1, x_effector2, x1, x2 = identity_tensor.clone(), identity_tensor.clone(), identity_tensor.clone(), identity_tensor.clone()
    zero_tensor = torch.zeros((batchsize, 8, joint_num + 1), device=dh1.device, dtype=dh1.dtype)
    A1, A2, J1, J2 = zero_tensor.clone(),zero_tensor.clone(),zero_tensor.clone(),zero_tensor.clone()
    w = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], device=dh2.device, dtype=dh2.dtype).repeat(batchsize, 1)
    # print("Time taken jacobian: ", time() - start_time)
    # start_time = time()
    # compute the forward kinematics
    for i in range(joint_num+1):
        a1, a2 = _dh2dq(dh1, dh2, theta1[:,i], theta2[:,i], i)
        x_effector1 = dq_mult(x_effector1, a1)
        x_effector2 = dq_mult(x_effector2, a2)
        A1[:,:,i] = a1
        A2[:,:,i] = a2
        # print("a1 ", a1)
        # print("a2 ", a2)
    # print("dh ", time() - start_time)
    # # compute the jacobian
    # start_time = time()
    for i in range(joint_num+1):
        z1 = 0.5*Ad(x1, w)
        z2 = 0.5*Ad(x2, w)
        x1 = dq_mult(x1, A1[:,:,i].contiguous())
        x2 = dq_mult(x2, A2[:,:,i].contiguous())
        j1 =  dq_mult(z1 ,x_effector1)
        j2 =  dq_mult(z2 ,x_effector2)
        J1[:,:,i] = j1
        J2[:,:,i] = j2
        
    J1 = hamiplus8(base1)@haminus8(effector1)@J1
    J2 = hamiplus8(base2)@haminus8(effector2)@J2
    x_effector1 = dq_mult(dq_mult(base1, x_effector1), effector1)
    x_effector2 = dq_mult(dq_mult(base2, x_effector2), effector2)
    rel_pose = dq_mult(conj(x_effector2), x_effector1)
    abs_pose = dq_mult(x_effector2, dq_sqrt(rel_pose))
    c8 = C8(dh1)
    Jxr = torch.cat((hamiplus8(conj(x_effector2)) @ J1, haminus8(x_effector1) @ c8 @ J2), dim=2)
    eps_batch = torch.eye(8, device=dh1.device, dtype=dh1.dtype).repeat(batchsize, 1, 1)
    # inv_Jxr = torch.inverse(Jxr@Jxr.permute(0, 2, 1)+eps_batch)
    return rel_pose, abs_pose, Jxr

