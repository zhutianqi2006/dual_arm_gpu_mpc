import torch
from dq_torch import  norm, rel_abs_pose_rel_jac
import numpy as np
import cProfile
from dqrobotics.robot_modeling import DQ_SerialManipulatorMDH, DQ_CooperativeDualTaskSpace
import time
########################################################### 
################### CPU MODEL Define #####################
##########################################################    
# cpu robot1
robot1_config_dh_mat = np.array([[0.0, 0.333,   0.0,        0.0, 0],
                                [0.0, 0.0,     0.0,    -1.5708, 0],
                                [0.0, 0.316,   0.0,     1.5708, 0],
                                [0.0, 0.0,     0.0825,  1.5708, 0],
                                [0.0, 0.384,  -0.0825, -1.5708, 0],
                                [0.0, 0.0,     0.0,     1.5708, 0],
                                [0.0, 0.0,   0.088,   1.5708, 0]])
robot1_dh_mat =  robot1_config_dh_mat.T
cpu_robot1 = DQ_SerialManipulatorMDH(robot1_dh_mat)
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
cpu_dq_dual_arm_model = DQ_CooperativeDualTaskSpace(cpu_robot1, cpu_robot2)
########################################################### 
################### GPU MODEL Define #####################
##########################################################    
batch_size = 2000
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
dual_arm_joint_pos = [0.5, 0.5, 0.5 ,0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 ,0.5, 0.5, 0.5, 0.5]
batch_robot1_base = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_robot2_base = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_robot1_effector = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_robot2_effector = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
q_vec1 = torch.tensor([0.5, 0.5, 0.5 ,0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1) 
q_vec2 = torch.tensor([0.5, 0.5, 0.5 ,0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
dh_matrix1 = dh_matrix1.reshape(-1)
dh_matrix2 = dh_matrix2.reshape(-1)
desire_line_d = [0,0,0,1]
desire_quat_line_ref = [0, -0.011682, 0.003006, -0.999927]
batch_line_d = torch.tensor(desire_line_d, dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
batch_quat_line_ref = torch.tensor(desire_quat_line_ref, dtype=torch.float32, device= "cuda:0").repeat(batch_size, 1)
########################################################### 
######################  WARM UP ###########################
###########################################################   
for i in range(10):
    rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q_vec1, q_vec2,
                         batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)

for i in range(10):
    cpu_dq_dual_arm_model.relative_pose(dual_arm_joint_pos)
    cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos)
    cpu_dq_dual_arm_model.relative_pose_jacobian(dual_arm_joint_pos)

########################################################### 
###################### START Test #########################
########################################################### 
start_time = time.time()
for i in range(10):
    rel_abs_pose_rel_jac(dh_matrix1, dh_matrix2,
                         batch_robot1_base,  batch_robot2_base, 
                         batch_robot1_effector, batch_robot2_effector, 
                         q_vec1, q_vec2,
                         batch_line_d, batch_quat_line_ref, 7, 7, 1, 1)
end_time = time.time()
print("Time taken: ", end_time - start_time)
start_time = time.time()
for i in range(20000):
    cpu_dq_dual_arm_model.relative_pose(dual_arm_joint_pos)
    cpu_dq_dual_arm_model.absolute_pose(dual_arm_joint_pos)
    cpu_dq_dual_arm_model.relative_pose_jacobian(dual_arm_joint_pos)
end_time = time.time()
print("Time taken: ", end_time - start_time)
