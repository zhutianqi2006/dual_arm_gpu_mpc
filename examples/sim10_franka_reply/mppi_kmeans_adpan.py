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
from utils.mppi_kmeans_adpan_module import MPPIKmeansAdpAnModule
# 
import rclpy
import array


class MPPIKmeansAdpAnModuleDynamic(MPPIKmeansAdpAnModule):
    def init_collision_model(self):
        self.tensor_args = TensorDeviceType()
        self.curobo_config = RobotWorldConfig.load_from_config(self.curobo_robot_file, self.curobo_world_file, 
                                                               collision_activation_distance=self.min_collision_distance,
                                                               self_collision_activation_distance=self.min_self_collision_distance)
        self.curobo_fn = RobotWorld(self.curobo_config)
        self.curobo_fn2 = RobotWorld(self.curobo_config)
        # obstacle 1
        self.init_obstacle_x = self.curobo_config.world_model.world_model.cuboid[0].pose[0]
        self.init_obstacle_x_dim = self.curobo_config.world_model.world_model.cuboid[0].dims[0]
        self.current_obstacle_x = self.init_obstacle_x
        self.last_obstacle_x = self.init_obstacle_x
        # obstacle 2
        self.init_obstacle_y = self.curobo_config.world_model.world_model.cuboid[1].pose[1]
        self.init_obstacle_y_dim = self.curobo_config.world_model.world_model.cuboid[1].dims[1]
        self.current_obstacle_y = self.init_obstacle_y
        self.last_obstacle_y = self.init_obstacle_y

    def update_curobo_world_model(self, time_elapsed:float):
       # obstacle 1
        self.last_obstacle_x = self.current_obstacle_x
        self.curobo_config.world_model.world_model.cuboid[0].dims[0] = self.init_obstacle_x_dim
        self.curobo_config.world_model.world_model.cuboid[0].pose[0] = self.init_obstacle_x + 0.1*time_elapsed
        # obstacle 2
        self.last_obstacle_y = self.current_obstacle_y
        self.curobo_config.world_model.world_model.cuboid[1].dims[1] = self.init_obstacle_y_dim
        self.curobo_config.world_model.world_model.cuboid[1].pose[1] = self.init_obstacle_y + 0.1*time_elapsed
        self.ros_module.write_obstacle(self.curobo_config.world_model.world_model.cuboid[0].pose[0:3]+self.curobo_config.world_model.world_model.cuboid[1].pose[0:3], 2)
        self.curobo_fn.update_world(self.curobo_config.world_model.world_model)
        self.curobo_fn2.update_world(self.curobo_config.world_model.world_model)
        # update obstacle position
        self.current_obstacle_x = self.curobo_config.world_model.world_model.cuboid[0].pose[0]
        self.fake_obstacle_x = self.curobo_config.world_model.world_model.cuboid[0].pose[0]
        self.current_obstacle_y = self.curobo_config.world_model.world_model.cuboid[1].pose[1]
        self.fake_obstacle_y = self.curobo_config.world_model.world_model.cuboid[1].pose[1]

    def update_fake_curobo_world_model(self, vel1:float, vel2:float):
        # obstacle 1
        fake_obstacle_x_dim= 1.4*self.mppi_T*abs(vel1)*self.mppi_dt
        self.fake_obstacle_x += 0.5*vel1*self.mppi_dt*self.mppi_T
        self.curobo_config.world_model.world_model.cuboid[0].pose[0] += fake_obstacle_x_dim
        self.curobo_config.world_model.world_model.cuboid[0].pose[0] = self.fake_obstacle_x
        # obstacle 2
        fake_obstacle_y_dim = 1.4*self.mppi_T*abs(vel2)*self.mppi_dt
        self.fake_obstacle_y += 0.5*vel2*self.mppi_dt*self.mppi_T
        self.curobo_config.world_model.world_model.cuboid[1].pose[1] += fake_obstacle_y_dim
        self.curobo_config.world_model.world_model.cuboid[1].pose[1] = self.fake_obstacle_y
        self.curobo_fn.update_world(self.curobo_config.world_model.world_model)
        self.curobo_fn2.update_world(self.curobo_config.world_model.world_model)

    def update_obstacle_velocity_estimate(self):
        self.current_obstacle_x_velocity = (self.current_obstacle_x - self.last_obstacle_x)/0.1
        self.current_obstacle_y_velocity = (self.current_obstacle_y - self.last_obstacle_y)/0.1
        
    def play_once(self):
        self.update_curobo_world_model(time.time() - self.start_time)
        self.update_joint_states()
        self.update_obstacle_velocity_estimate()
        self.update_fake_curobo_world_model(self.current_obstacle_x_velocity, self.current_obstacle_y_velocity)
        _, mppi_energy = self.mppi_worker()
        mppi_u0, _ = self.mppi_worker2()
        p_u0, p_energy = self.traditional_control_result()
        print("mppi_energy: ", mppi_energy)
        print("p_energy: ", p_energy)
        mppi_u0 = mppi_u0[0].cpu().numpy()
        flag = self.update_c(mppi_energy, p_energy)
        print(self.c)
        if(flag ==True):
            u0 = p_u0
            self.last_mppi_result = torch.zeros(self.mppi_T, (self.robot1_q_num+self.robot2_q_num), device=self.device, dtype=self.dtype)
            self.current_mppi_result = torch.zeros(self.mppi_T, (self.robot1_q_num+self.robot2_q_num), device=self.device, dtype=self.dtype)
        else:
            u0 =mppi_u0
        u0 = u0.tolist()
        self.ros_module.write_high_u(u0)

def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    # 步骤 3: 实例化DQ_SerialManipulatorDH
    desire_abs_pose = [-0.743697, 0.002919, 0.668503, 0.003245, 0.001775, -0.011929, 0.003932, -0.392559]
    desire_abs_position = [-0.507146, -0.006007, 0.599849]
    desire_rel_pose = [-0.161469, 0.982784, -0.000002, 0.089794, 0.000098, -0.000579, 0.420438, 0.006521]
    desire_line_d = [0, 0, 0, 1]
    desire_quat_line_ref = [0, -0.994308, 0.00868, 0.106191]
    config_path = os.path.join(os.path.dirname(__file__), 'two_franka_r9.yaml')
    config = ConfigModule(config_path)
    mppi_module =  MPPIKmeansAdpAnModuleDynamic(config, desire_abs_pose, desire_abs_position, desire_rel_pose, desire_line_d, desire_quat_line_ref)
    mppi_module.warm_up()
    while True:
        mppi_module.play_once()


if __name__ == "__main__":
    main()