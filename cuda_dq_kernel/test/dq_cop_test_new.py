import torch
from dq_torch import normalize, norm
from torch.autograd import profiler
from dq_mult_extension import rel_abs_pose_rel_jac
import time
import cProfile
    
# 步骤 2: 创建DH参数矩阵
# 假设有一个2关节机械臂，每个关节都是旋转关节
# 参数顺序：θ（半角），d，a，α（半角），关节类型（0=旋转，1=移动）
start_time = time.time()
batch_size = 20000
dh_matrix1 = torch.tensor([
    [0.5, 0.0, 0.1, 0.0, 0],  
    [0.5, 0.0, 0.1, 0.0, 0],
    [0.5, 0.0, 0.1, 0.0, 0],
    [0.5, 0.0, 0.1, 0.0, 0],
    [0.5, 0.0, 0.1, 0.0, 0],
    [0.5, 0.0, 0.1, 0.0, 0] 
], dtype=torch.float32, device= "cuda:0") # .t() 转置矩阵，以匹配类的期望格式
dh_matrix2 = torch.tensor([
    [0.5, 0.0, 0.1, 0.0, 0],  
    [0.5, 0.0, 0.1, 0.0, 0],
    [0.5, 0.0, 0.1, 0.0, 0],
    [0.5, 0.0, 0.1, 0.0, 0],
    [0.5, 0.0, 0.1, 0.0, 0],
    [0.5, 0.0, 0.1, 0.0, 0] 
], dtype=torch.float32, device= "cuda:0")  # .t() 转置矩阵，以匹配类的期望格式
# 步骤 3: 实例化DQ_SerialManipulatorDH
base1 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
base2 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
effector1 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
effector2 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
q_vec1 = torch.tensor([0.5, 0.5, 0.5 ,0.5, 0.5, 0.5], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1) 
q_vec2 = torch.tensor([0.5, 0.5, 0.5 ,0.5, 0.5, 0.5], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
dh_matrix1 = dh_matrix1.reshape(-1)
dh_matrix2 = dh_matrix2.reshape(-1)
end_time = time.time()
# a,b,c = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2, base1, base2, effector1, effector2, q_vec1, q_vec2, 5) 
# a,b,c = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2, base1, base2, effector1, effector2, q_vec1, q_vec2, 5) 
print("Time taken: ", end_time - start_time)

q_vec1 = torch.tensor([0.5, 0.5, 0.5 ,0.5, 0.5, 0.5], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1) 
q_vec2 = torch.tensor([0.5, 0.5, 0.5 ,0.5, 0.5, 0.5], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)

# a,b,c = rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2, base1, base2, effector1, effector2, q_vec1, q_vec2, batch_size, 5)
a,b,c = rel_abs_pose_rel_jac(dh_matrix1.contiguous(), dh_matrix2.contiguous(), base1, base2, effector1, effector2, q_vec1, q_vec2, 5) 
a,b,c = rel_abs_pose_rel_jac(dh_matrix1.contiguous(), dh_matrix2.contiguous(), base1, base2, effector1, effector2, q_vec1, q_vec2, 5)
eps_batch = 1e-8*torch.eye(8, device=base1.device, dtype=base1.dtype).repeat(batch_size, 1, 1)
new_c = torch.inverse(c@c.permute(0, 2, 1)+eps_batch)
new_c = torch.inverse(c@c.permute(0, 2, 1)+eps_batch)
start_time = time.time()
for i in range(10):
    a,b,c = rel_abs_pose_rel_jac(dh_matrix1.contiguous(), dh_matrix2.contiguous(), base1, base2, effector1, effector2, q_vec1, q_vec2, 5)
end_time = time.time()
print("Time taken: ", end_time - start_time)
# print(c)
# print(c)
# print(a)
# print(rel_result1)
# print(b)
# print(rel_result2)
# print(c)
# print("Time taken: ", end_time - start_time)
# # # 步骤 6: 打印结果
# print("Forward Kinematics Result1:", a)
# # 步骤 6: 打印结果