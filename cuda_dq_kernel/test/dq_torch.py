import torch
dq_threshold = 1e-7
# v: torch.Tensor [batch_size, n_dof]

@torch.jit.script
def dq_mult(q1:torch.Tensor, q2:torch.Tensor):
    # 预计算dq2_p和dq2_d的组件，以避免重复计算
    dq1_p = q1[:, :4]
    dq1_d = q1[:, 4:]
    dq2_p = q2[:, :4]
    dq2_d = q2[:, 4:]
    a2, b2, c2, d2 = dq2_p[:, 0], dq2_p[:, 1], dq2_p[:, 2], dq2_p[:, 3]
    ad2, bd2, cd2, dd2 = dq2_d[:, 0], dq2_d[:, 1], dq2_d[:, 2], dq2_d[:, 3]
    # 实部四元数乘法
    q_real_a = dq1_p[:, 0] * a2 - dq1_p[:, 1] * b2 - dq1_p[:, 2] * c2 - dq1_p[:, 3] * d2
    q_real_b = dq1_p[:, 0] * b2 + dq1_p[:, 1] * a2 + dq1_p[:, 2] * d2 - dq1_p[:, 3] * c2
    q_real_c = dq1_p[:, 0] * c2 - dq1_p[:, 1] * d2 + dq1_p[:, 2] * a2 + dq1_p[:, 3] * b2
    q_real_d = dq1_p[:, 0] * d2 + dq1_p[:, 1] * c2 - dq1_p[:, 2] * b2 + dq1_p[:, 3] * a2
    q_real = torch.stack([q_real_a, q_real_b, q_real_c, q_real_d], dim=1)
    # 双部四元数乘法
    q_dual_a = dq1_d[:, 0] * a2 + dq1_p[:, 0] * ad2 - dq1_d[:, 1] * b2 - dq1_p[:, 1] * bd2 - dq1_d[:, 2] * c2 - dq1_p[:, 2] * cd2 - dq1_d[:, 3] * d2 - dq1_p[:, 3] * dd2
    q_dual_b = dq1_d[:, 0] * b2 + dq1_p[:, 0] * bd2 + dq1_d[:, 1] * a2 + dq1_p[:, 1] * ad2 + dq1_d[:, 2] * d2 + dq1_p[:, 2] * dd2 - dq1_d[:, 3] * c2 - dq1_p[:, 3] * cd2
    q_dual_c = dq1_d[:, 0] * c2 + dq1_p[:, 0] * cd2 - dq1_d[:, 1] * d2 - dq1_p[:, 1] * dd2 + dq1_d[:, 2] * a2 + dq1_p[:, 2] * ad2 + dq1_d[:, 3] * b2 + dq1_p[:, 3] * bd2
    q_dual_d = dq1_d[:, 0] * d2 + dq1_p[:, 0] * dd2 + dq1_d[:, 1] * c2 + dq1_p[:, 1] * cd2 - dq1_d[:, 2] * b2 - dq1_p[:, 2] * bd2 + dq1_d[:, 3] * a2 + dq1_p[:, 3] * ad2
    q_dual = torch.stack([q_dual_a, q_dual_b, q_dual_c, q_dual_d], dim=1)
    # 合并实部和双部
    result = torch.cat([q_real, q_dual], dim=1)
    return result

@torch.jit.script
def P(v:torch.Tensor)->torch.Tensor:
    # 直接操作原始张量，避免创建新的零张量
    new_q = v.clone()
    new_q[:, 4:] = 0  # 只将后四位清零
    return new_q

@torch.jit.script
def D(v:torch.Tensor):
    # 直接操作原始张量，避免创建新的零张量
    new_q = torch.zeros_like(v, device=v.device, dtype=v.dtype)
    new_q[:, :4] = v[:, 4:]
    return new_q

@torch.jit.script
def Re(v:torch.Tensor):
    new_q = torch.zeros_like(v, device=v.device, dtype=v.dtype)
    # 只保留每行的第一个和第五个元素
    new_q[:, 0] = v[:, 0]  # 第一个元素
    new_q[:, 4] = v[:, 4]  # 第五个元素
    return new_q

@torch.jit.script
def Im(v:torch.Tensor):
    # 创建一个新的全零张量与self.q形状相同
    im_q = torch.zeros_like(v, device=v.device, dtype=v.dtype)
    # 保留除第一个和第五个元素外的所有元素
    im_q[:, 1:4] = v[:, 1:4]  # 第二到第四元素
    im_q[:, 5:] = v[:, 5:]    # 第六到第八元素
    return im_q

@torch.jit.script
def conj(v:torch.Tensor):
    # 创建一个新的张量，实部符号不变，虚部符号反转
    conj_q = v.clone()
    conj_q[:, 1:] = -v[:, 1:]  # 反转第二到第四元素的符号
    return conj_q

@torch.jit.script
def norm(v:torch.Tensor):
    aux = v.clone()
    norm = torch.zeros_like(v, device=v.device, dtype=v.dtype)
    if torch.all(P(aux) == 0):  # Primary == 0
        return norm  # norm = 0
    else:
        # norm calculation
        norm = dq_mult(conj(aux), aux)
        norm[:,0] = torch.sqrt(norm[:,0])
        norm[:,4] = norm[:,4] / (2 * norm[:,0])
        return norm
    
@torch.jit.script
def translation(v:torch.Tensor):
    # translation part calculation
    translation = P(v)
    translation = 2.0 * dq_mult(D(v),conj(translation))
    return translation

@torch.jit.script
def rotation_angle(v:torch.Tensor):
    # rotation angle calculation
    angle = 2 * torch.acos(v[:, 0])
    return angle

@torch.jit.script
def rotation_axis(v:torch.Tensor):
    # rotation axis calculation
    phi = rotation_angle(v)/2.0
    phi_tensor = torch.zeros_like(v, device=v.device, dtype=v.dtype)
    phi_tensor[:, 0] = 1/torch.sin(phi)
    rot_axis = P(v)
    rot_axis = dq_mult(phi_tensor,Im(rot_axis))
    return rot_axis
 
@torch.jit.script
def dq_log(v:torch.Tensor):
    phi_tensor = torch.zeros_like(v, device=v.device, dtype=v.dtype)
    phi_tensor[:, 0] = 0.5*rotation_angle(v)
    p =dq_mult(phi_tensor,rotation_axis(v))
    d = 0.5*translation(v)
    return torch.stack((p[:,0], p[:,1], p[:,2], p[:,3], d[:,0], d[:,1], d[:,2], d[:,3]), dim=1)

# @torch.jit.script
def dq_exp(v:torch.Tensor):
    batch_size = v.shape[0]
    E_ = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=v.dtype, device=v.device).repeat(batch_size,1)
    prim = P(v)
    phi = prim.norm(p=2, dim=[1])
    phi_tensor = torch.zeros(batch_size,8, device=v.device, dtype=v.dtype)
    phi_tensor[:, 0] = (torch.sin(phi)/(phi+ 1e-8))
    prim = dq_mult(phi_tensor,P(v))
    prim[:, 0] += torch.cos(phi)
    exp = prim + dq_mult(dq_mult(E_,D(v)),prim)
    return exp

# @torch.jit.script
def dq_pow(v:torch.Tensor, n:float):
    return dq_exp(n * dq_log(v))

# @torch.jit.script
def inv(v:torch.Tensor):
    aux = v.clone()
    aux_conj = conj(v)
    aux2 = dq_mult(aux, aux_conj)
    inv_dq = torch.zeros_like(v, device=v.device, dtype=v.dtype)
    inv_dq[:, 0] = 1 / aux2[:, 0]  # 实部
    inv_dq[:, 4] = -aux2[:, 4] / (aux2[:, 0] ** 2)  # 对偶部分的实部，假设对偶部分的虚部为0
    inv_final = dq_mult(aux_conj, inv_dq)
    return inv_final

# @torch.jit.script
def normalize(v:torch.Tensor):
    return dq_mult(v, (inv(norm(v))))

# @torch.jit.script
def hamiplus4(v:torch.Tensor):
    H = torch.stack([
        torch.stack([v[:, 0], -v[:, 1], -v[:, 2], -v[:, 3]], dim=-1),
        torch.stack([v[:, 1],  v[:, 0], -v[:, 3],  v[:, 2]], dim=-1),
        torch.stack([v[:, 2],  v[:, 3],  v[:, 0], -v[:, 1]], dim=-1),
        torch.stack([v[:, 3], -v[:, 2],  v[:, 1],  v[:, 0]], dim=-1)
    ], dim=-2)  # dim=-2 保证矩阵的形状是正确的
    return H

# @torch.jit.script
def haminus4(v:torch.Tensor):
    H = torch.stack([
        torch.stack([v[:, 0], -v[:, 1], -v[:, 2], -v[:, 3]], dim=-1),
        torch.stack([v[:, 1],  v[:, 0],  v[:, 3], -v[:, 2]], dim=-1),
        torch.stack([v[:, 2], -v[:, 3],  v[:, 0],  v[:, 1]], dim=-1),
        torch.stack([v[:, 3],  v[:, 2], -v[:, 1],  v[:, 0]], dim=-1)
    ], dim=-2)
    return H

# @torch.jit.script
def hamiplus8(v:torch.Tensor):
    batch_size = v.shape[0]
    h8 = torch.zeros((batch_size, 8, 8), device=v.device, dtype=v.dtype)
     # First 4x4 block
    h8[:, 0, 0] =  v[:, 0]; h8[:, 0, 1] = -v[:, 1]; h8[:, 0, 2] = -v[:, 2]; h8[:, 0, 3] = -v[:, 3]
    h8[:, 1, 0] =  v[:, 1]; h8[:, 1, 1] =  v[:, 0]; h8[:, 1, 2] = -v[:, 3]; h8[:, 1, 3] =  v[:, 2]
    h8[:, 2, 0] =  v[:, 2]; h8[:, 2, 1] =  v[:, 3]; h8[:, 2, 2] =  v[:, 0]; h8[:, 2, 3] = -v[:, 1]
    h8[:, 3, 0] =  v[:, 3]; h8[:, 3, 1] = -v[:, 2]; h8[:, 3, 2] =  v[:, 1]; h8[:, 3, 3] =  v[:, 0]
    # Second 4x4 block (top right) remains zeros
    # Third 4x4 block (bottom left)
    h8[:, 4, 0] =  v[:, 4]; h8[:, 4, 1] = -v[:, 5]; h8[:, 4, 2] = -v[:, 6]; h8[:, 4, 3] = -v[:, 7]
    h8[:, 5, 0] =  v[:, 5]; h8[:, 5, 1] =  v[:, 4]; h8[:, 5, 2] = -v[:, 7]; h8[:, 5, 3] =  v[:, 6]
    h8[:, 6, 0] =  v[:, 6]; h8[:, 6, 1] =  v[:, 7]; h8[:, 6, 2] =  v[:, 4]; h8[:, 6, 3] = -v[:, 5]
    h8[:, 7, 0] =  v[:, 7]; h8[:, 7, 1] = -v[:, 6]; h8[:, 7, 2] =  v[:, 5]; h8[:, 7, 3] =  v[:, 4]
    # Fourth 4x4 block (bottom right)
    h8[:, 4, 4] =  v[:, 0]; h8[:, 4, 5] = -v[:, 1]; h8[:, 4, 6] = -v[:, 2]; h8[:, 4, 7] = -v[:, 3]
    h8[:, 5, 4] =  v[:, 1]; h8[:, 5, 5] =  v[:, 0]; h8[:, 5, 6] = -v[:, 3]; h8[:, 5, 7] =  v[:, 2]
    h8[:, 6, 4] =  v[:, 2]; h8[:, 6, 5] =  v[:, 3]; h8[:, 6, 6] =  v[:, 0]; h8[:, 6, 7] = -v[:, 1]
    h8[:, 7, 4] =  v[:, 3]; h8[:, 7, 5] = -v[:, 2]; h8[:, 7, 6] =  v[:, 1]; h8[:, 7, 7] =  v[:, 0]
    return h8

# @torch.jit.script
def haminus8(v:torch.Tensor):
    batch_size = v.shape[0]
    h8 = torch.zeros((batch_size, 8, 8), device=v.device, dtype=v.dtype)
    h8[:, 0, 0] =  v[:, 0]; h8[:, 0, 1] = -v[:, 1]; h8[:, 0, 2] = -v[:, 2]; h8[:, 0, 3] = -v[:, 3]
    h8[:, 1, 0] =  v[:, 1]; h8[:, 1, 1] =  v[:, 0]; h8[:, 1, 2] =  v[:, 3]; h8[:, 1, 3] = -v[:, 2]
    h8[:, 2, 0] =  v[:, 2]; h8[:, 2, 1] = -v[:, 3]; h8[:, 2, 2] =  v[:, 0]; h8[:, 2, 3] =  v[:, 1]
    h8[:, 3, 0] =  v[:, 3]; h8[:, 3, 1] =  v[:, 2]; h8[:, 3, 2] = -v[:, 1]; h8[:, 3, 3] =  v[:, 0]
    
    h8[:, 4, 0] =  v[:, 4]; h8[:, 4, 1] = -v[:, 5]; h8[:, 4, 2] = -v[:, 6]; h8[:, 4, 3] = -v[:, 7]
    h8[:, 5, 0] =  v[:, 5]; h8[:, 5, 1] =  v[:, 4]; h8[:, 5, 2] =  v[:, 7]; h8[:, 5, 3] = -v[:, 6]
    h8[:, 6, 0] =  v[:, 6]; h8[:, 6, 1] = -v[:, 7]; h8[:, 6, 2] =  v[:, 4]; h8[:, 6, 3] =  v[:, 5]
    h8[:, 7, 0] =  v[:, 7]; h8[:, 7, 1] =  v[:, 6]; h8[:, 7, 2] = -v[:, 5]; h8[:, 7, 3] =  v[:, 4]

    h8[:, 4, 4] =  v[:, 0]; h8[:, 4, 5] = -v[:, 1]; h8[:, 4, 6] = -v[:, 2]; h8[:, 4, 7] = -v[:, 3]
    h8[:, 5, 4] =  v[:, 1]; h8[:, 5, 5] =  v[:, 0]; h8[:, 5, 6] =  v[:, 3]; h8[:, 5, 7] = -v[:, 2]
    h8[:, 6, 4] =  v[:, 2]; h8[:, 6, 5] = -v[:, 3]; h8[:, 6, 6] =  v[:, 0]; h8[:, 6, 7] =  v[:, 1]
    h8[:, 7, 4] =  v[:, 3]; h8[:, 7, 5] =  v[:, 2]; h8[:, 7, 6] = -v[:, 1]; h8[:, 7, 7] =  v[:, 0]
    return h8

# @torch.jit.script
def Ad(q1:torch.Tensor,q2:torch.Tensor):
    conj_q1 = q1.clone()
    conj_q1[:, 1:] = -q1[:, 1:]  # 反转第二到第四元素的符号
    #
    dq1_p = q1[:, :4]
    dq1_d = q1[:, 4:]
    dq2_p = q2[:, :4]
    dq2_d = q2[:, 4:]
    a2, b2, c2, d2 = dq2_p[:, 0], dq2_p[:, 1], dq2_p[:, 2], dq2_p[:, 3]
    ad2, bd2, cd2, dd2 = dq2_d[:, 0], dq2_d[:, 1], dq2_d[:, 2], dq2_d[:, 3]
    # 实部四元数乘法
    q_real_a = dq1_p[:, 0] * a2 - dq1_p[:, 1] * b2 - dq1_p[:, 2] * c2 - dq1_p[:, 3] * d2
    q_real_b = dq1_p[:, 0] * b2 + dq1_p[:, 1] * a2 + dq1_p[:, 2] * d2 - dq1_p[:, 3] * c2
    q_real_c = dq1_p[:, 0] * c2 - dq1_p[:, 1] * d2 + dq1_p[:, 2] * a2 + dq1_p[:, 3] * b2
    q_real_d = dq1_p[:, 0] * d2 + dq1_p[:, 1] * c2 - dq1_p[:, 2] * b2 + dq1_p[:, 3] * a2
    q_real = torch.stack([q_real_a, q_real_b, q_real_c, q_real_d], dim=1)
    # 双部四元数乘法
    q_dual_a = dq1_d[:, 0] * a2 + dq1_p[:, 0] * ad2 - dq1_d[:, 1] * b2 - dq1_p[:, 1] * bd2 - dq1_d[:, 2] * c2 - dq1_p[:, 2] * cd2 - dq1_d[:, 3] * d2 - dq1_p[:, 3] * dd2
    q_dual_b = dq1_d[:, 0] * b2 + dq1_p[:, 0] * bd2 + dq1_d[:, 1] * a2 + dq1_p[:, 1] * ad2 + dq1_d[:, 2] * d2 + dq1_p[:, 2] * dd2 - dq1_d[:, 3] * c2 - dq1_p[:, 3] * cd2
    q_dual_c = dq1_d[:, 0] * c2 + dq1_p[:, 0] * cd2 - dq1_d[:, 1] * d2 - dq1_p[:, 1] * dd2 + dq1_d[:, 2] * a2 + dq1_p[:, 2] * ad2 + dq1_d[:, 3] * b2 + dq1_p[:, 3] * bd2
    q_dual_d = dq1_d[:, 0] * d2 + dq1_p[:, 0] * dd2 + dq1_d[:, 1] * c2 + dq1_p[:, 1] * cd2 - dq1_d[:, 2] * b2 - dq1_p[:, 2] * bd2 + dq1_d[:, 3] * a2 + dq1_p[:, 3] * ad2
    q_dual = torch.stack([q_dual_a, q_dual_b, q_dual_c, q_dual_d], dim=1)
    # 合并实部和双部
    q1q2 = torch.cat([q_real, q_dual], dim=1)
    ad = dq_mult(q1q2, conj_q1)
    
    return ad

# @torch.jit.script
def vec3(q:torch.Tensor):
    # 返回批处理的第2到第4个元素（1:4），对应四元数的虚部
    return q[:, 1:4]

# @torch.jit.script
def vec6(q:torch.Tensor):
    # 返回批处理的第2到第4个元素以及第6到第8个元素
    return torch.cat((q[:, 1:4], q[:, 5:8]), dim=1)

# @torch.jit.script
def vec4(q:torch.Tensor):
    # 返回批处理的前4个元素
    return q[:, :4]

# @torch.jit.script
def vec8(q:torch.Tensor):
    # 返回整个批处理数据
    return q

# @torch.jit.script
def C8(q:torch.Tensor):
    diag_c8 = torch.zeros((q.shape[0], 8, 8), device=q.device, dtype=q.dtype)
    diag_c8[:, 0, 0] =  1
    diag_c8[:, 1, 1] = -1
    diag_c8[:, 2, 2] = -1
    diag_c8[:, 3, 3] = -1
    diag_c8[:, 4, 4] =  1
    diag_c8[:, 5, 5] = -1
    diag_c8[:, 6, 6] = -1
    diag_c8[:, 7, 7] = -1
    return diag_c8

# @torch.jit.script
def hp8_j_mul(q:torch.Tensor, J:torch.Tensor):
    return hamiplus8(q) @ J

# @torch.jit.script    
def hp8_hm8_j_mul(q1:torch.Tensor, q2:torch.Tensor, J:torch.Tensor):
    batch_size = q1.shape[0]
    h8p = torch.zeros((batch_size, 8, 8),device=q1.device, dtype=q1.dtype)
    h8m = torch.zeros((batch_size, 8, 8),device=q1.device, dtype=q1.dtype)
    # hamiplus8
    h8p[:, 0, 0] =  q1[:, 0]; h8p[:, 0, 1] = -q1[:, 1]; h8p[:, 0, 2] = -q1[:, 2]; h8p[:, 0, 3] = -q1[:, 3]
    h8p[:, 1, 0] =  q1[:, 1]; h8p[:, 1, 1] =  q1[:, 0]; h8p[:, 1, 2] = -q1[:, 3]; h8p[:, 1, 3] =  q1[:, 2]
    h8p[:, 2, 0] =  q1[:, 2]; h8p[:, 2, 1] =  q1[:, 3]; h8p[:, 2, 2] =  q1[:, 0]; h8p[:, 2, 3] = -q1[:, 1]
    h8p[:, 3, 0] =  q1[:, 3]; h8p[:, 3, 1] = -q1[:, 2]; h8p[:, 3, 2] =  q1[:, 1]; h8p[:, 3, 3] =  q1[:, 0]
    h8p[:, 4, 0] =  q1[:, 4]; h8p[:, 4, 1] = -q1[:, 5]; h8p[:, 4, 2] = -q1[:, 6]; h8p[:, 4, 3] = -q1[:, 7]
    h8p[:, 5, 0] =  q1[:, 5]; h8p[:, 5, 1] =  q1[:, 4]; h8p[:, 5, 2] = -q1[:, 7]; h8p[:, 5, 3] =  q1[:, 6]
    h8p[:, 6, 0] =  q1[:, 6]; h8p[:, 6, 1] =  q1[:, 7]; h8p[:, 6, 2] =  q1[:, 4]; h8p[:, 6, 3] = -q1[:, 5]
    h8p[:, 7, 0] =  q1[:, 7]; h8p[:, 7, 1] = -q1[:, 6]; h8p[:, 7, 2] =  q1[:, 5]; h8p[:, 7, 3] =  q1[:, 4]
    h8p[:, 4, 4] =  q1[:, 0]; h8p[:, 4, 5] = -q1[:, 1]; h8p[:, 4, 6] = -q1[:, 2]; h8p[:, 4, 7] = -q1[:, 3]
    h8p[:, 5, 4] =  q1[:, 1]; h8p[:, 5, 5] =  q1[:, 0]; h8p[:, 5, 6] = -q1[:, 3]; h8p[:, 5, 7] =  q1[:, 2]
    h8p[:, 6, 4] =  q1[:, 2]; h8p[:, 6, 5] =  q1[:, 3]; h8p[:, 6, 6] =  q1[:, 0]; h8p[:, 6, 7] = -q1[:, 1]
    h8p[:, 7, 4] =  q1[:, 3]; h8p[:, 7, 5] = -q1[:, 2]; h8p[:, 7, 6] =  q1[:, 1]; h8p[:, 7, 7] =  q1[:, 0]
    # haminus8
    h8m[:, 0, 0] =  q2[:, 0]; h8m[:, 0, 1] = -q2[:, 1]; h8m[:, 0, 2] = -q2[:, 2]; h8m[:, 0, 3] = -q2[:, 3]
    h8m[:, 1, 0] =  q2[:, 1]; h8m[:, 1, 1] =  q2[:, 0]; h8m[:, 1, 2] =  q2[:, 3]; h8m[:, 1, 3] = -q2[:, 2]
    h8m[:, 2, 0] =  q2[:, 2]; h8m[:, 2, 1] = -q2[:, 3]; h8m[:, 2, 2] =  q2[:, 0]; h8m[:, 2, 3] =  q2[:, 1]
    h8m[:, 3, 0] =  q2[:, 3]; h8m[:, 3, 1] =  q2[:, 2]; h8m[:, 3, 2] = -q2[:, 1]; h8m[:, 3, 3] =  q2[:, 0]
    h8m[:, 4, 0] =  q2[:, 4]; h8m[:, 4, 1] = -q2[:, 5]; h8m[:, 4, 2] = -q2[:, 6]; h8m[:, 4, 3] = -q2[:, 7]
    h8m[:, 5, 0] =  q2[:, 5]; h8m[:, 5, 1] =  q2[:, 4]; h8m[:, 5, 2] =  q2[:, 7]; h8m[:, 5, 3] = -q2[:, 6]
    h8m[:, 6, 0] =  q2[:, 6]; h8m[:, 6, 1] = -q2[:, 7]; h8m[:, 6, 2] =  q2[:, 4]; h8m[:, 6, 3] =  q2[:, 5]
    h8m[:, 7, 0] =  q2[:, 7]; h8m[:, 7, 1] =  q2[:, 6]; h8m[:, 7, 2] = -q2[:, 5]; h8m[:, 7, 3] =  q2[:, 4]
    h8m[:, 4, 4] =  q2[:, 0]; h8m[:, 4, 5] = -q2[:, 1]; h8m[:, 4, 6] = -q2[:, 2]; h8m[:, 4, 7] = -q2[:, 3]
    h8m[:, 5, 4] =  q2[:, 1]; h8m[:, 5, 5] =  q2[:, 0]; h8m[:, 5, 6] =  q2[:, 3]; h8m[:, 5, 7] = -q2[:, 2]
    h8m[:, 6, 4] =  q2[:, 2]; h8m[:, 6, 5] = -q2[:, 3]; h8m[:, 6, 6] =  q2[:, 0]; h8m[:, 6, 7] =  q2[:, 1]
    h8m[:, 7, 4] =  q2[:, 3]; h8m[:, 7, 5] =  q2[:, 2]; h8m[:, 7, 6] = -q2[:, 1]; h8m[:, 7, 7] =  q2[:, 0]
    return h8p@h8m@J
