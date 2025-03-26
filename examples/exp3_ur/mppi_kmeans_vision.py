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
        self.init_obstacle_x = self.curobo_config.world_model.world_model.cuboid[0].pose[0]
        self.init_obstacle_x_dim = self.curobo_config.world_model.world_model.cuboid[0].dims[0]

    def update_curobo_world_model(self):
        self.curobo_config.world_model.world_model.cuboid[0].pose[0] = self.ros_module.dynamic_obstacle[0]
        self.curobo_config.world_model.world_model.cuboid[0].pose[1] = self.ros_module.dynamic_obstacle[1]
        self.curobo_config.world_model.world_model.cuboid[0].pose[2] = self.ros_module.dynamic_obstacle[2]
        self.ros_module.write_obstacle(self.curobo_config.world_model.world_model.cuboid[0].pose[0:3], 1)
        self.curobo_fn.update_world(self.curobo_config.world_model.world_model)
        self.curobo_fn2.update_world(self.curobo_config.world_model.world_model)


    def play_once(self):
        self.update_curobo_world_model()
        self.update_joint_states()
        _, mppi_energy = self.mppi_worker()
        mppi_u, _ = self.mppi_worker2()
        mppi_u0 = mppi_u[0].cpu().numpy()
        p_u0, p_energy = self.traditional_control_result()
        print("mppi_energy: ", mppi_energy)
        print("p_energy: ", p_energy)
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
    desire_abs_pose = [0.005744, - 0.683663, 0.020305, 0.729493, - 0.033974, - 0.094329, - 0.254169, - 0.08106]
    desire_abs_position = [0.32, -0.25, 0.40]
    # desire_abs_pose = [0.005744, - 0.683663, 0.020305, 0.729493,  0.06289854,  0.07243161, -0.17836156,  0.07235046]
    # desire_abs_position = [0.35, 0.20, 0.15]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.054285, - 0.000927, - 0.262089, - 0.003409]
    desire_line_d = [0,0,0,1]
    desire_quat_line_ref = [0,-0.9995,-0.026341,0.017418]
    config_path = os.path.join(os.path.dirname(__file__), 'ur3_and_ur3e.yaml')
    config = ConfigModule(config_path)
    mppi_module = MPPIKmeansAdpAnModuleDynamic(config, desire_abs_pose, desire_abs_position, desire_rel_pose, desire_line_d, desire_quat_line_ref)
    mppi_module.warm_up()
    while True:
        mppi_module.play_once()



if __name__ == "__main__":
    main()