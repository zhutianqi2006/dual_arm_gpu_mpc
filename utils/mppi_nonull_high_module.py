#!/usr/bin/env python
# system library
import os
import time
import numpy as np
import math
from math import pi
import threading
import torch
# curobo for collision detection
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
# DQ Robotics cpu
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH, DQ_SerialManipulatorMDH, DQ_CooperativeDualTaskSpace
# DQ Robotics used in cuda
from dq_torch import rel_abs_pose_rel_jac
from utils.config_module import ConfigModule
from utils.high_ros_module import HighROSModule
# 
import rclpy
import array

class MPPINoNullHighModule():
    def __init__(self, config: ConfigModule , desire_abs_pose: torch.Tensor, desire_abs_position: torch.Tensor,
                 desire_rel_pose: torch.Tensor, desire_line_d: torch.Tensor,  desire_quat_line_ref: torch.Tensor):
        # torch type
        self.dtype = torch.float32
        self.device = "cuda:0"
        # mppi parameters
        self.mppi_T = config.mppi_T
        self.mppi_dt = config.mppi_dt
        self.mppi_seed = config.mppi_seed
        self.batch_size = config.batch_size
        self.min_collision_distance = config.min_collision_distance
        self.min_self_collision_distance = config.min_self_collision_distance
        self.mean = config.mean
        self.std = config.std
        self.log_std = config.log_std
        self.batch_eps = 1e-8*torch.eye(8, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1, 1)
        self.param_lambda = 0.5
        self.max_acc_abs_value = config.max_acc_abs_value
        self.warm_up_flag = False
        self.max_abs_tilt_angle = config.max_abs_tilt_angle 
        # mppi weight
        self.collision_constraint_weight = config.collision_constraint_weight
        self.q_limit_constraint_weight = config.q_limit_constraint_weight
        self.q_vel_constraint_weight = config.q_vel_constraint_weight
        self.tilt_constraint_weight = config.tilt_constraint_weight
        self.abs_weight = config.abs_weight
        self.abs_position_weight = config.abs_position_weight
        self.terminal_abs_weight = config.terminal_abs_weight
        self.stagnation_weight = config.stagnation_weight
        self.terminal_abs_position_weight = config.terminal_abs_position_weight
        self.q_acc_weight = config.q_acc_weight
        self.q_vel_weight = config.q_vel_weight
        # robot 1 
        self.robot1_dh_mat = config.robot1_dh_mat
        self.robot1_base = config.robot1_base 
        self.robot1_effector = config.robot1_effector
        self.robot1_q_num = config.robot1_q_num
        self.robot1_dh_type = config.robot1_dh_type
        # robot 2
        self.robot2_dh_mat = config.robot2_dh_mat
        self.robot2_base = config.robot2_base
        self.robot2_effector = config.robot2_effector
        self.robot2_q_num = config.robot2_q_num
        self.robot2_dh_type = config.robot2_dh_type
        # joint position limits
        self.robot1_q_min = config.robot1_q_min
        self.robot1_q_max = config.robot1_q_max
        self.robot2_q_min = config.robot2_q_min
        self.robot2_q_max = config.robot2_q_max
        # joint velocity limits
        self.robot1_dq_min = config.robot1_dq_min
        self.robot1_dq_max = config.robot1_dq_max
        self.robot2_dq_min = config.robot2_dq_min
        self.robot2_dq_max = config.robot2_dq_max
        # joint acc limits
        self.robot1_ddq_min = config.robot1_ddq_min
        self.robot1_ddq_max = config.robot1_ddq_max
        self.robot2_ddq_min = config.robot2_ddq_min
        self.robot2_ddq_max = config.robot2_ddq_max
        # other parameters
        self.desire_abs_pose = desire_abs_pose
        self.desire_abs_position = desire_abs_position
        self.desire_rel_pose = desire_rel_pose
        self.desire_line_d = desire_line_d
        self.desire_quat_line_ref = desire_quat_line_ref
        self.high_rel_gain = config.high_rel_gain
        self.high_abs_gain = config.high_abs_gain
        self.c_abs_max = config.c_abs_max
        self.c_eta = config.c_eta
        self.c = 0
        self.init_cpu_dq_model()
        # make them to tensor
        self.init_tensor()
        # curobo collision model
        self.curobo_world_file = config.curobo_world_file
        self.curobo_robot_file = config.curobo_robot_file
        self.init_collision_model()
        self.ros_module = HighROSModule(config)
        self.ros_thread = threading.Thread(target=self.ros_module.run)
        self.ros_thread.start()

    def init_cpu_dq_model(self):
        self.cpu_desire_abs_pose = DQ(self.desire_abs_pose)
        self.cpu_desire_abs_pose = self.cpu_desire_abs_pose.normalize()
        self.cpu_desire_abs_pose = vec8(self.cpu_desire_abs_pose)
        self.cpu_desire_rel_pose = DQ(self.desire_rel_pose)
        self.cpu_desire_rel_pose = self.cpu_desire_rel_pose.normalize()
        self.cpu_desire_rel_pose = vec8(self.cpu_desire_rel_pose)
        self.cpu_desire_line_d = DQ(self.desire_line_d)
        self.cpu_desire_quat_line_ref = DQ(self.desire_quat_line_ref)
        
        # robot1
        robot1_config_dh_mat = np.array(self.robot1_dh_mat)
        self.cpu_robot1_dh_mat =  robot1_config_dh_mat.T
        self.cpu_robot1_base = DQ(self.robot1_base)
        self.cpu_robot1_base = self.cpu_robot1_base.normalize() 
        self.cpu_robot1_effector = DQ(self.robot1_effector)
        self.cpu_robot1_effector = self.cpu_robot1_effector.normalize()
        if self.robot1_dh_type == 1:
            self.cpu_robot1 = DQ_SerialManipulatorMDH(self.cpu_robot1_dh_mat)
        else:
            self.cpu_robot1 = DQ_SerialManipulatorDH(self.cpu_robot1_dh_mat)
        self.cpu_robot1.set_base_frame(self.cpu_robot1_base)
        self.cpu_robot1.set_reference_frame(self.cpu_robot1_base)
        self.cpu_robot1.set_effector(self.cpu_robot1_effector)
        # robot2
        robot2_config_dh_mat = np.array(self.robot2_dh_mat)
        self.cpu_robot2_dh_mat =  robot2_config_dh_mat.T
        self.cpu_robot2_base = DQ(self.robot2_base)
        self.cpu_robot2_base = self.cpu_robot2_base.normalize()
        self.cpu_robot2_effector = DQ(self.robot2_effector)
        self.cpu_robot2_effector = self.cpu_robot2_effector.normalize()
        if self.robot2_dh_type == 1:
            self.cpu_robot2 = DQ_SerialManipulatorMDH(self.cpu_robot2_dh_mat)
        else:
            self.cpu_robot2 = DQ_SerialManipulatorDH(self.cpu_robot2_dh_mat)
        self.cpu_robot2.set_base_frame(self.cpu_robot2_base)
        self.cpu_robot2.set_reference_frame(self.cpu_robot2_base)
        self.cpu_robot2.set_effector(self.cpu_robot2_effector)
        # robot2 and robot1
        self.cpu_dq_dual_arm_model = DQ_CooperativeDualTaskSpace(self.cpu_robot1, self.cpu_robot2)

    def init_ros(self):
        self.ros_thread = threading.Thread(target=self.ros_module.run)
        self.ros_thread.start()

    def update_joint_states(self):
        self.robot1_q, self.robot2_q = self.ros_module.read_joint_state()
        self.batch_fake_robot1_q = torch.tensor(self.robot1_q, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_fake_robot2_q = torch.tensor(self.robot2_q, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)

    def init_tensor(self):
        self.gpu_desire_abs_pose = torch.tensor(self.desire_abs_pose, device=self.device, dtype=self.dtype)
        self.gpu_desire_abs_position = torch.tensor(self.desire_abs_position, device=self.device, dtype=self.dtype)
        self.gpu_desire_rel_pose = torch.tensor(self.desire_rel_pose, device=self.device, dtype=self.dtype)
        self.batch_line_d = torch.tensor(self.desire_line_d, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_quat_line_ref = torch.tensor(self.desire_quat_line_ref, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        # robot q min max
        self.gpu_robot1_q_min = torch.tensor(self.robot1_q_min, device=self.device, dtype=self.dtype)
        self.gpu_robot1_q_max = torch.tensor(self.robot1_q_max, device=self.device, dtype=self.dtype)
        self.gpu_robot2_q_min = torch.tensor(self.robot2_q_min, device=self.device, dtype=self.dtype)
        self.gpu_robot2_q_max = torch.tensor(self.robot2_q_max, device=self.device, dtype=self.dtype)
        # robot dq min max
        self.gpu_robot1_dq_min = torch.tensor(self.robot1_dq_min, device=self.device, dtype=self.dtype)
        self.gpu_robot1_dq_max = torch.tensor(self.robot1_dq_max, device=self.device, dtype=self.dtype)
        self.gpu_robot2_dq_min = torch.tensor(self.robot2_dq_min, device=self.device, dtype=self.dtype)
        self.gpu_robot2_dq_max = torch.tensor(self.robot2_dq_max, device=self.device, dtype=self.dtype)
        # robot ddq min max
        self.gpu_robot1_ddq_min = torch.tensor(self.robot1_ddq_min, device=self.device, dtype=self.dtype)
        self.gpu_robot1_ddq_max = torch.tensor(self.robot1_ddq_max, device=self.device, dtype=self.dtype)
        self.gpu_robot2_ddq_min = torch.tensor(self.robot2_ddq_min, device=self.device, dtype=self.dtype)
        self.gpu_robot2_ddq_max = torch.tensor(self.robot2_ddq_max, device=self.device, dtype=self.dtype)
        # init desire pose tensor
        self.batch_desire_abs_position = self.gpu_desire_abs_position.repeat(self.batch_size, 1)
        self.batch_desire_abs_pose = self.gpu_desire_abs_pose.repeat(self.batch_size, 1)
        self.batch_desire_rel_pose = self.gpu_desire_rel_pose.repeat(self.batch_size, 1)
        # robot 1 
        self.gpu_robot1_dh_mat = torch.tensor(self.robot1_dh_mat, device=self.device, dtype= torch.float32)
        self.gpu_robot1_dh_mat = self.gpu_robot1_dh_mat.reshape(-1).contiguous()
        self.gpu_robot1_base  = torch.tensor(self.robot1_base, device=self.device, dtype=self.dtype)
        self.batch_robot1_base = self.gpu_robot1_base.repeat(self.batch_size, 1)
        self.gpu_robot1_effector  = torch.tensor(self.robot1_effector, device=self.device, dtype=self.dtype)
        self.batch_robot1_effector  = self.gpu_robot1_effector.repeat(self.batch_size, 1)
        # robot 2
        self.gpu_robot2_dh_mat = torch.tensor(self.robot2_dh_mat, device=self.device, dtype= torch.float32)
        self.gpu_robot2_dh_mat = self.gpu_robot2_dh_mat.reshape(-1).contiguous()
        self.gpu_robot2_base  = torch.tensor(self.robot2_base, device=self.device, dtype=self.dtype)
        self.batch_robot2_base = self.gpu_robot2_base.repeat(self.batch_size, 1)
        self.gpu_robot2_effector  = torch.tensor(self.robot2_effector, device=self.device, dtype=self.dtype)
        self.batch_robot2_effector  = self.gpu_robot2_effector.repeat(self.batch_size, 1)
        # MPPI
        self.last_mppi_result = torch.zeros(self.mppi_T, (self.robot1_q_num+self.robot2_q_num), device=self.device, dtype=self.dtype)
        self.current_mppi_result = torch.zeros(self.mppi_T, (self.robot1_q_num+self.robot2_q_num), device=self.device, dtype=self.dtype)
        self.first_element_mppi_result = torch.zeros((self.robot1_q_num+self.robot2_q_num), device=self.device, dtype=self.dtype)
        self.batch_max_abs_tilt_angle = torch.tensor(self.max_abs_tilt_angle, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        # joint position limits
        self.batch_robot1_q_min = torch.tensor(self.robot1_q_min, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_robot1_q_max = torch.tensor(self.robot1_q_max, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_robot2_q_min = torch.tensor(self.robot2_q_min, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_robot2_q_max = torch.tensor(self.robot2_q_max, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        # joint velocity limits
        self.batch_robot1_dq_min = torch.tensor(self.robot1_dq_min, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_robot1_dq_max = torch.tensor(self.robot1_dq_max, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_robot2_dq_min = torch.tensor(self.robot2_dq_min, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)
        self.batch_robot2_dq_max = torch.tensor(self.robot2_dq_max, device=self.device, dtype=self.dtype).repeat(self.batch_size, 1)

    def init_collision_model(self):

        self.tensor_args = TensorDeviceType()
        self.curobo_config = RobotWorldConfig.load_from_config(self.curobo_robot_file, self.curobo_world_file, 
                                                               collision_activation_distance=self.min_collision_distance,
                                                               self_collision_activation_distance=self.min_self_collision_distance)
        self.curobo_fn = RobotWorld(self.curobo_config)
        self.curobo_fn2 = RobotWorld(self.curobo_config)

    def update_curobo_world_model(self, time_elapsed:float):
        self.last_obstacle_x = self.current_obstacle_x
        self.curobo_config.world_model.world_model.cuboid[2].dims[1] = self.init_obstacle_x_dim
        self.curobo_config.world_model.world_model.cuboid[2].pose[1] = self.init_obstacle_x + math.sin(0.15*time_elapsed)
        self.ros_module.write_obstacle(self.curobo_config.world_model.world_model.cuboid[2].pose[0:3], 1)
        self.curobo_fn.update_world(self.curobo_config.world_model.world_model)
        self.curobo_fn2.update_world(self.curobo_config.world_model.world_model)
        self.current_obstacle_x = self.curobo_config.world_model.world_model.cuboid[2].pose[1]
        self.fake_obstacle_x = self.curobo_config.world_model.world_model.cuboid[2].pose[1]
       
    def update_fake_curobo_world_model(self, vel:float):
        fake_obstacle_x_dim = 1.4*abs(vel)*self.mppi_dt*self.mppi_T
        self.fake_obstacle_x += 0.5*vel*self.mppi_dt*self.mppi_T
        self.curobo_config.world_model.world_model.cuboid[2].pose[1] = self.fake_obstacle_x
        self.curobo_config.world_model.world_model.cuboid[2].dims[1] += fake_obstacle_x_dim
        self.curobo_fn.update_world(self.curobo_config.world_model.world_model)
        self.curobo_fn2.update_world(self.curobo_config.world_model.world_model)

    def update_obstacle_velocity_estimate(self):
        self.current_obstacle_x_velocity = (self.current_obstacle_x - self.last_obstacle_x)/0.05
        
    def get_collision_cost(self, weight:float):
        q = torch.cat((self.batch_fake_robot1_q, self.batch_fake_robot2_q), dim=1)
        q_mid = torch.cat(((self.last_batch_fake_robot1_q+self.batch_fake_robot1_q)/2.0, (self.last_batch_fake_robot2_q+self.batch_fake_robot2_q)/2.0), dim=1)
        d_world1, d_self1 = self.curobo_fn.get_world_self_collision_distance_from_joints(q)
        d_world2, d_self2 = self.curobo_fn.get_world_self_collision_distance_from_joints(q_mid)
        d_new = d_world1 + d_world2 + d_self1 + d_self2
        d_new[d_new!=0] = weight
        num_samples = d_new.size(0)
        return d_new.view(num_samples, 1)

    def moving_average_filter(self, xx: torch.Tensor, window_size: int) -> torch.Tensor:
        """Apply moving average filter for smoothing input sequence, using numpy internally."""
        # Convert to numpy
        xx_np = xx.cpu().numpy()
        b = np.ones(window_size) / window_size
        num_steps, num_controls = xx_np.shape
        xx_mean_np = np.zeros_like(xx_np)
        
        # Apply moving average filter
        for d in range(num_controls):
            # Convolve and handle edges by using 'valid' mode or 'full' and slicing
            convolved = np.convolve(xx_np[:, d], b, mode='same')
            # Handle the edge normalization if needed
            half_window = window_size // 2
            xx_mean_np[:, d] = convolved
        
        # Convert back to torch tensor on the same device as input
        xx_mean = torch.from_numpy(xx_mean_np).to(xx.device)
        return xx_mean
    
    def mppi_worker(self):
        # init ur3 dual quaternion modelwa
        batch_last_mppi_result = self.last_mppi_result.repeat(self.batch_size, 1, 1)
        robot1_acc_explore_seq, robot2_acc_explore_seq = epsilon_generator_log(int(self.batch_size), self.robot1_q_num, self.robot2_q_num, self.mppi_T,
                                                                               self.mean, self.std, self.log_std, self.mppi_dt*self.max_acc_abs_value, self.mppi_seed)
        batch_last_robot1_mppi_result = batch_last_mppi_result[:,:,:self.robot1_q_num]
        batch_last_robot2_mppi_result = batch_last_mppi_result[:,:, self.robot1_q_num:]
        batch_robot1_dq_seq = robot1_acc_explore_seq + batch_last_mppi_result[:,:,:self.robot1_q_num]
        batch_robot2_dq_seq = robot2_acc_explore_seq + batch_last_mppi_result[:,:, self.robot1_q_num:]
        self.last_batch_fake_robot1_q = self.batch_fake_robot1_q.clone()
        self.last_batch_fake_robot2_q = self.batch_fake_robot2_q.clone()
        self.first_batch_fake_robot1_q = self.batch_fake_robot1_q.clone()
        self.first_batch_fake_robot2_q = self.batch_fake_robot2_q.clone()
        self.stage_cost = torch.zeros(self.batch_size, 1, dtype = self.dtype, device = self.device)
        for i in range(self.mppi_T):
            # init to get first nullspace matrix
            if i == 0:
                rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(self.gpu_robot1_dh_mat, self.gpu_robot2_dh_mat,
                                                     self.batch_robot1_base,  self.batch_robot2_base, 
                                                     self.batch_robot1_effector, self.batch_robot2_effector, 
                                                     self.batch_fake_robot1_q, self.batch_fake_robot2_q,
                                                     self.batch_line_d, self.batch_quat_line_ref, self.robot1_q_num, self.robot2_q_num, self.robot1_dh_type, self.robot2_dh_type)
                bacth_rel_jacobian_null =  get_rel_jacobian_null(bacth_rel_jacobian, self.robot1_q_num, self.robot2_q_num, self.batch_size)

            batch_robot1_ith_dq, batch_robot2_ith_dq = get_current_vel(batch_robot1_dq_seq, batch_robot2_dq_seq, i)
            batch_robot1_ith_proj_dq = torch.clamp(batch_robot1_ith_dq, self.gpu_robot1_dq_min, self.gpu_robot1_dq_max)
            batch_robot2_ith_proj_dq = torch.clamp(batch_robot2_ith_dq, self.gpu_robot2_dq_min, self.gpu_robot2_dq_max)
            
            self.last_batch_fake_robot1_q = self.batch_fake_robot1_q
            self.last_batch_fake_robot2_q = self.batch_fake_robot2_q
            # update fake joint position
            self.batch_fake_robot1_q, self.batch_fake_robot2_q, batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq=update_joint_position_with_limits(self.batch_fake_robot1_q, self.batch_fake_robot2_q, 
                                        batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq,
                                        self.gpu_robot1_q_min, self.gpu_robot1_q_max, 
                                        self.gpu_robot2_q_min, self.gpu_robot2_q_max, self.mppi_dt)
            robot1_acc_explore_seq[:,i,:self.robot1_q_num] = torch.clamp(robot1_acc_explore_seq[:,i,:self.robot1_q_num], self.mppi_dt*self.gpu_robot1_ddq_min, self.mppi_dt*self.gpu_robot1_ddq_max)
            robot2_acc_explore_seq[:,i,:self.robot2_q_num] = torch.clamp(robot2_acc_explore_seq[:,i,:self.robot2_q_num], self.mppi_dt*self.gpu_robot2_ddq_min, self.mppi_dt*self.gpu_robot2_ddq_max)
            
            rel_pos, bacth_abs_pos, bacth_rel_jacobian, batch_abs_position, batch_angle = rel_abs_pose_rel_jac(self.gpu_robot1_dh_mat, self.gpu_robot2_dh_mat,
                                                     self.batch_robot1_base,  self.batch_robot2_base, 
                                                     self.batch_robot1_effector, self.batch_robot2_effector, 
                                                     self.batch_fake_robot1_q, self.batch_fake_robot2_q,
                                                     self.batch_line_d, self.batch_quat_line_ref, self.robot1_q_num, self.robot2_q_num, self.robot1_dh_type, self.robot2_dh_type)
            abs_cost = get_abs_cost(self.batch_desire_abs_pose, bacth_abs_pos, self.batch_desire_abs_position, batch_abs_position, self.abs_weight, self.abs_position_weight)
            vel_cost = get_vel_cost(batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq, self.q_vel_weight)
            tilt_constraint_cost = get_tilt_constraint_cost(batch_angle, self.batch_max_abs_tilt_angle, self.tilt_constraint_weight)
            acc_cost = get_acc_cost(batch_robot1_ith_proj_dq, batch_robot2_ith_proj_dq, batch_last_mppi_result, self.robot1_q_num, self.robot2_q_num, i, self.q_acc_weight)
            collision_cost = self.get_collision_cost(self.collision_constraint_weight)
            rel_cost = get_rel_error(self.gpu_desire_rel_pose, rel_pos, 1000)
            self.stage_cost += (abs_cost + rel_cost + vel_cost+ collision_cost + acc_cost + tilt_constraint_cost)
            
        joint_change = abs(self.first_batch_fake_robot1_q-self.batch_fake_robot1_q).sum(dim=1, keepdim=True)+abs(self.first_batch_fake_robot2_q-self.batch_fake_robot2_q).sum(dim=1, keepdim=True)
        min_joint_change = 0.02
        joint_change = torch.clamp(joint_change, min=min_joint_change)
        stagnation_cost = self.stagnation_weight*joint_change
        abs_terminal_cost = get_abs_cost(self.batch_desire_abs_pose, bacth_abs_pos, self.batch_desire_abs_position, batch_abs_position, self.terminal_abs_weight, self.terminal_abs_position_weight)
        self.stage_cost += abs_terminal_cost + abs_terminal_cost/stagnation_cost
        min_energy = self.stage_cost.min()
        epsilon = get_all_dq_seq(robot1_acc_explore_seq, robot2_acc_explore_seq)
        w_epsilon = compute_weights(epsilon, self.stage_cost, self.batch_size, self.param_lambda)
        w_epsilon = self.moving_average_filter(w_epsilon, int(self.mppi_T))
        self.current_mppi_result = w_epsilon + self.last_mppi_result
        self.current_mppi_result = torch.clamp(self.current_mppi_result, torch.cat((self.gpu_robot1_dq_min, self.gpu_robot2_dq_min)), torch.cat((self.gpu_robot1_dq_max, self.gpu_robot2_dq_max)))
        self.last_mppi_result[:-1,:] = self.current_mppi_result[1:,:]
        self.last_mppi_result[-1,:] = self.current_mppi_result[-1,:]
        return self.current_mppi_result[0], min_energy
    
    
    def warm_up(self):
        for i in range(10):
            self.update_joint_states()
            mppi_u0, mppi_energy = self.mppi_worker()
            mppi_u0 = mppi_u0.cpu().numpy()
        self.last_mppi_result = torch.zeros(self.mppi_T, (self.robot1_q_num+self.robot2_q_num), device=self.device, dtype=self.dtype)
        self.current_mppi_result = torch.zeros(self.mppi_T, (self.robot1_q_num+self.robot2_q_num), device=self.device, dtype=self.dtype)
        self.first_element_mppi_result = torch.zeros((self.robot1_q_num+self.robot2_q_num), device=self.device, dtype=self.dtype)
        self.c = 0.0
        self.start_time = time.time()

    def play_once(self):
        self.update_joint_states()
        mppi_u0, mppi_energy = self.mppi_worker()
        print("mppi_energy: ", mppi_energy)
        u0 =mppi_u0
        u0 = u0.tolist()
        self.ros_module.write_high_u(u0)

# batch_robot_q size is batch_size x joint_num
# batch_robot_dq size is batch_size x joint_num
# robot_joint_min size is joint_num
# robot_joint_min size is joint_num
@torch.jit.script
def update_joint_position_with_limits(batch_robot1_q:torch.Tensor, batch_robot2_q:torch.Tensor, 
                          batch_robot1_dq:torch.Tensor, batch_robot2_dq:torch.Tensor,
                          robot1_joint_min:torch.Tensor, robot1_joint_max:torch.Tensor,
                          robot2_joint_min:torch.Tensor, robot2_joint_max:torch.Tensor, dt:float):
    # 计算期望的新位置
    updated_robot1_q = batch_robot1_q + batch_robot1_dq * dt
    updated_robot2_q = batch_robot2_q + batch_robot2_dq * dt

    # 限制新位置在关节限制范围内
    clamped_robot1_q = torch.clamp(updated_robot1_q, robot1_joint_min, robot1_joint_max)
    clamped_robot2_q = torch.clamp(updated_robot2_q, robot2_joint_min, robot2_joint_max)

    # 计算允许的速度
    allowed_robot1_dq = (clamped_robot1_q - batch_robot1_q) / dt
    allowed_robot2_dq = (clamped_robot2_q - batch_robot2_q) / dt

    return clamped_robot1_q, clamped_robot2_q, allowed_robot1_dq, allowed_robot2_dq


@torch.jit.script   
def update_fake_joint_pos(batch_fake_robot1_q:torch.Tensor, batch_fake_robot2_q:torch.Tensor, 
                          batch_robot1_qd:torch.Tensor, batch_robot2_qd:torch.Tensor, dt:float):
    last_batch_fake_robot1_q = batch_fake_robot1_q.clone()
    last_batch_fake_robot2_q = batch_fake_robot2_q.clone()
    batch_fake_robot1_q = dt * batch_robot1_qd + batch_fake_robot1_q
    batch_fake_robot2_q = dt * batch_robot2_qd + batch_fake_robot2_q
    return last_batch_fake_robot1_q, last_batch_fake_robot2_q, batch_fake_robot1_q, batch_fake_robot2_q

# get acc    
@torch.jit.script    
def compute_weights(epsilon: torch.Tensor, S: torch.Tensor, batch_size:int, p_lamada: float) -> torch.Tensor:
    """compute weights for each sample"""
    # prepare buffer
    w = torch.zeros(batch_size, device=S.device, dtype= S.dtype)
    # calculate rho
    rho = S.min()
    # calculate eta
    eta = torch.sum(torch.exp((-1.0 / p_lamada) * (S - rho)))
    while eta < 22 or eta > 30:
        if eta > 30:
            p_lamada *= 0.8
        else:
            p_lamada *= 1.2
        eta = torch.sum(torch.exp((-1.0 / p_lamada) * (S - rho)))
    w = (1.0 / eta) * torch.exp((-1.0 / p_lamada) * (S - rho))
    w_epsilon = torch.sum(w.view(batch_size, 1, 1) * epsilon, dim=0)
    return w_epsilon

# return u through null space
@torch.jit.script
def get_proj_qd(batch_robot1_qd:torch.Tensor, batch_robot2_qd:torch.Tensor, 
                robot1_q_num:int, robot2_q_num:int, batch_null_space_mat:torch.Tensor):
    batch_two_robot_qd = torch.cat((batch_robot1_qd.unsqueeze(2),  batch_robot2_qd.unsqueeze(2)), dim=1)
    # 切片Jacobian矩阵以适应维度
    batch_null_space_mat_first = batch_null_space_mat[:, :robot1_q_num, :]  # shape is batch_size x robot1_q_num x (robot1_q_num+robot2_q_num)
    batch_null_space_mat_last = batch_null_space_mat[:,  robot1_q_num:, :]
    # 执行矩阵乘法并更新位置
    batch_robot1_proj_qd = torch.matmul(batch_null_space_mat_first, batch_two_robot_qd).squeeze(2)
    batch_robot2_proj_qd  = torch.matmul(batch_null_space_mat_last, batch_two_robot_qd).squeeze(2)  
    return batch_robot1_proj_qd, batch_robot2_proj_qd

# get i-th/MPPI_T joint velocity
@torch.jit.script
def get_current_vel(robot1_dq_seq: torch.Tensor, robot2_dq_seq: torch.Tensor, i: int):
    robot1_dq = robot1_dq_seq[:, i, :]
    robot2_dq = robot2_dq_seq[:, i, :]
    return robot1_dq, robot2_dq

# combine two robot joint position
@torch.jit.script
def get_all_q(batch_robot1_q:torch.Tensor, batch_robot2_q:torch.Tensor):
    return torch.cat((batch_robot1_q, batch_robot2_q), dim=1)

# combine two mppi exploration robot joint velocity
@torch.jit.script
def get_all_dq_seq(batch_robot1_dq_rollout_seq:torch.Tensor, batch_robot2_dq_rollout_seq:torch.Tensor):
    return torch.cat((batch_robot1_dq_rollout_seq, batch_robot2_dq_rollout_seq), dim=2)

# desire_abs_pos 为 batch_size*8
# get abs error 1 norm when expolration
@torch.jit.script
def get_abs_cost(desire_abs_pos:torch.Tensor, abs_pos:torch.Tensor, desire_abs_position:torch.Tensor, abs_position:torch.Tensor, rot_weight:float, position_weight:float):
    quat_difference = rot_weight*torch.abs(desire_abs_pos - abs_pos)
    position_difference = position_weight*torch.abs(desire_abs_position -  abs_position)
    result = quat_difference.sum(dim=1, keepdim=True) + position_difference.sum(dim=1, keepdim=True)
    return result

# get vel 1 norm when expolration
@torch.jit.script
def get_vel_cost(batch_robot1_dq:torch.Tensor, batch_robot2_dq:torch.Tensor, weight:float):
    difference = torch.abs(batch_robot1_dq) + torch.abs(batch_robot2_dq)
    result = weight * difference.sum(dim=1, keepdim=True)    
    return result

@torch.jit.script
def get_rel_error(desire_rel_pos:torch.Tensor, rel_pos:torch.Tensor, weight:float):
    difference = torch.abs(desire_rel_pos - rel_pos)
    result = weight * difference.sum(dim=1, keepdim=True)    
    return result

# when vel over limit, return cost
@torch.jit.script
def get_vel_constraint_cost(batch_robot1_dq:torch.Tensor, batch_robot2_dq:torch.Tensor,
                            batch_robot1_dq_min:torch.Tensor, batch_robot1_dq_max:torch.Tensor,
                            batch_robot2_dq_min:torch.Tensor, batch_robot2_dq_max:torch.Tensor,
                            weight:float):
    # Detect violations for robot1
    robot1_violation_min = (batch_robot1_dq < batch_robot1_dq_min).float()
    robot1_violation_max = (batch_robot1_dq > batch_robot1_dq_max).float()
    
    # Detect violations for robot2
    robot2_violation_min = (batch_robot2_dq < batch_robot2_dq_min).float()
    robot2_violation_max = (batch_robot2_dq > batch_robot2_dq_max).float()
    
    # Total violations per batch
    total_violations = robot1_violation_min + robot1_violation_max + robot2_violation_min + robot2_violation_max
    
    # Calculate total penalty per batch
    total_cost = weight * total_violations.sum(dim=1, keepdim=True)
    
    return total_cost          
             
# when pos over limit, return cost
@torch.jit.script
def get_pos_constraint_cost(batch_robot1_q:torch.Tensor, batch_robot2_q:torch.Tensor,
                            batch_robot1_q_min:torch.Tensor, batch_robot1_q_max:torch.Tensor,
                            batch_robot2_q_min:torch.Tensor, batch_robot2_q_max:torch.Tensor,
                            weight:float):
    # Detect violations for robot1
    robot1_violation_min = (batch_robot1_q < batch_robot1_q_min).float()
    robot1_violation_max = (batch_robot1_q > batch_robot1_q_max).float()
    
    # Detect violations for robot2
    robot2_violation_min = (batch_robot2_q < batch_robot2_q_min).float()
    robot2_violation_max = (batch_robot2_q > batch_robot2_q_max).float()
    
    # Total violations per batch
    total_violations = robot1_violation_min + robot1_violation_max + robot2_violation_min + robot2_violation_max
    
    # Calculate total penalty per batch
    total_cost = weight * total_violations.sum(dim=1, keepdim=True)
    return total_cost

@torch.jit.script
def get_tilt_constraint_cost(angle:torch.Tensor, batch_max_abs_tilt_angle:torch.Tensor, weight:float):
    # Detect violations for robot1
    robot1_violation_min = (angle < -batch_max_abs_tilt_angle).float()
    robot1_violation_max = (angle > batch_max_abs_tilt_angle).float()
    
    # Total violations per batch
    total_violations = robot1_violation_min + robot1_violation_max
    
    # Calculate total penalty per batch
    total_cost = weight * total_violations.sum(dim=1, keepdim=True)
    return total_cost

@torch.jit.script
def get_acc_cost(batch_robot1_dq:torch.Tensor, batch_robot2_dq:torch.Tensor, batch_last_mppi_result:torch.Tensor, 
                 robot1_q_num:int, robot2_q_num:int, i:int, weight:float):
    last_robot1_dq = batch_last_mppi_result[:, i, :robot1_q_num]
    last_robot2_dq = batch_last_mppi_result[:, i, robot1_q_num:]
    difference = torch.abs(batch_robot1_dq - last_robot1_dq) + torch.abs(batch_robot2_dq - last_robot2_dq)
    result = weight * difference.sum(dim=1, keepdim=True)    
    return result

# get nullspace matrix
@torch.jit.script
def get_rel_jacobian_null(jac_batch:torch.Tensor, robot1_q_num:int, robot2_q_num:int, batch_size: int):
    batch_i = torch.eye(robot1_q_num+robot2_q_num, dtype=torch.float64, device= "cuda:0").repeat(batch_size, 1, 1)
    batch_eps = 1e-16*torch.eye(8, dtype=torch.float64, device= "cuda:0").repeat(batch_size, 1, 1)
    jac_batch = jac_batch.to(torch.float64)
    jac_batch_t = jac_batch.transpose(-2, -1)
    J_J_t  =  torch.matmul(jac_batch, jac_batch_t)
    J_pinv = jac_batch_t@torch.inverse(J_J_t + batch_eps)
    J_pinv_J = torch.matmul(J_pinv, jac_batch)
    rel_jacobian_null = batch_i - J_pinv_J
    return rel_jacobian_null.to(torch.float32)

# get random acceleration 1
@torch.jit.script
def epsilon_generator(batch_size: int, mppi_T: int, mean: float, std: float, max_abs_value: float):
    # 生成标准正态分布随机数
    data = torch.randn(batch_size, mppi_T, 12, dtype=torch.float32, device="cuda:0")
    # 调整随机数的均值和标准差
    data = data * std + mean
    # 限制数据的绝对值不超过 max_abs_value
    data = torch.clamp(data, min=-max_abs_value, max=max_abs_value)
    part1, part2 = data.split(6, dim=2)
    return part1, part2

# get random acceleration 2
@torch.jit.script
def epsilon_generator_log(batch_size: int, robot1_q_num: int, robot2_q_num: int, 
                          mppi_T: int, mean: float, std: float, log_std: float, max_abs_value: float, mppi_seed:int):
    total_q_num = robot1_q_num + robot2_q_num
    torch.manual_seed(mppi_seed)
    data = torch.randn(batch_size, mppi_T, total_q_num, dtype=torch.float32, device="cuda:0")
    torch.manual_seed(mppi_seed)
    log_normal_data = torch.exp(torch.randn(batch_size, mppi_T, total_q_num, dtype=torch.float32, device="cuda:0") * log_std)
    combined_data = data * log_normal_data
    combined_data = combined_data * std + mean
    combined_data = torch.clamp(combined_data, min=-max_abs_value, max=max_abs_value)
    part1, part2 = combined_data.split([robot1_q_num, robot2_q_num], dim=2)
    return part1, part2