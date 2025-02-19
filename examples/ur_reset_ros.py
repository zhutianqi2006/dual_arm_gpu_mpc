# Python standard lib
import os
import rtde_control
import rtde_receive
import numpy as np
# DQ Robotics
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH, DQ_CooperativeDualTaskSpace
from math import pi
import time
# ROS2
import rclpy
import rclpy.logging
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class DualArmDQController(Node):
    def __init__(self):
        super().__init__('dual_arm_dq_controller')
        self.init_dq_robot()
        self.setup_ros2()

    def init_dq_robot(self):
        ur3_dh_theta = np.array([0, 0, 0, 0, 0, 0]) # DH Joint angles (theta)
        ur3_dh_a = np.array([0, -0.24365, -0.21325, 0, 0, 0]) # DH Link lengths (a)
        ur3_dh_d = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819]) # DH Link offsets (d)
        ur3_dh_alpha = np.array([pi/2, 0, 0, pi/2, -pi/2, 0]) # DH Link twists (alpha)
        ur3_dh_type = np.array([0, 0, 0, 0, 0, 0]) # Joint types (0 for revolute)
        ur3_dh_matrix = np.array([ur3_dh_theta, ur3_dh_d, ur3_dh_a, ur3_dh_alpha, ur3_dh_type])
        # ur3 base frame change
        ur3_base_quat = 0.6125 + 0.3536*i_ - 0.3536*j_ + 0.6125*k_
        ur3_base = ur3_base_quat + E_ * 0.5 * DQ([0, 0.0, -0.09, 0.235]) * ur3_base_quat
        ur3_base = ur3_base.normalize()
        self.ur3_robot = DQ_SerialManipulatorDH(ur3_dh_matrix)
        self.ur3_robot.set_base_frame(ur3_base)
        self.ur3_robot.set_reference_frame(ur3_base)
        # init ur3e dual quaternion model
        ur3e_dh_theta = np.array([0, 0, 0, 0, 0, 0]) # DH Joint angles (theta)
        ur3e_dh_a = np.array([0, -0.24355, -0.2132, 0, 0, 0]) # DH Link lengths (a)
        ur3e_dh_d = np.array([0.15185, 0, 0, 0.13105, 0.08535, 0.0921]) # DH Link offsets (d)
        ur3e_dh_alpha = np.array([pi/2, 0, 0, pi/2, -pi/2, 0]) # DH Link twists (alpha)
        ur3e_dh_type = np.array([0, 0, 0, 0, 0, 0]) # Joint types (0 for revolute)
        ur3e_dh_matrix = np.array([ur3e_dh_theta, ur3e_dh_d, ur3e_dh_a, ur3e_dh_alpha, ur3e_dh_type])
        # ur3e base frame change
        ur3e_base_quat = -0.6125 + 0.3536*i_ - 0.3536*j_ - 0.6125*k_
        ur3e_base = ur3e_base_quat + E_ * 0.5 * DQ([0.0, 0.0, 0.09, 0.235]) * ur3e_base_quat
        ur3e_base = ur3e_base.normalize()
        self.ur3e_robot = DQ_SerialManipulatorDH(ur3e_dh_matrix)
        self.ur3e_robot.set_base_frame(ur3e_base)
        self.ur3e_robot.set_reference_frame(ur3e_base)
        # comibine dual arm model
        self.dq_dual_arm_model = DQ_CooperativeDualTaskSpace(self.ur3_robot, self.ur3e_robot)
        # i
        self.i = 0

    def setup_ros2(self):
        # init ros2 message for control
        self.ur3_vel_msg = JointState()
        self.ur3e_vel_msg = JointState()
        self.ur3_vel_msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ur3e_vel_msg.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                                  'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.ur3_vel_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ur3e_vel_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ur3_current_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ur3e_current_joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ur3_last_quat = [0.0, 0.0, 0.0, 1.0]
        self.ur3e_last_quat = [0.0, 0.0, 0.0, 1.0]
        self.ur3_init_flag = False
        self.ur3e_init_flag = False
        self.refer_init_flag = False
        # init ros2 message for record
        self.record_data_msg = Float64MultiArray()
        
        # setup ros2 publishers and subscribers
        self.subscription_ur3 = self.create_subscription(
            JointState,
            'ur3_joint_states',
            self.ur3_joint_state_callback,
            0)
        self.subscription_ur3e = self.create_subscription(
            JointState,
            'ur3e_joint_states',
            self.ur3e_joint_state_callback,
            0)
        self.publisher_ur3_velocity = self.create_publisher(
            JointState,
            'ur3_joint_command',
            1)
        self.publisher_ur3e_velocity = self.create_publisher(
            JointState,
            'ur3e_joint_command',
            1)
        self.publisher_record_data = self.create_publisher(
            Float64MultiArray,
            'dual_arm_record_data',
            1)
        self.timer = self.create_timer(0.008, self.velocity_publisher_callback)

    def ur3_joint_state_callback(self, msg):
        self.ur3_current_joint_positions = list(msg.position[:6])
        self.ur3_init_flag = True
        
    def ur3e_joint_state_callback(self, msg):
        self.ur3e_current_joint_positions = list(msg.position[:6])
        self.ur3e_init_flag = True

    def velocity_publisher_callback(self):
        if  (self.ur3_init_flag and self.ur3e_init_flag)==False:
            return
        self.i += 1
        dual_arm_rel_refer = DQ([0.043815, 0.998793, 0.006783, 0.021159, 0.001626, - 0.002018, 0.28023, 0.00204])
        dual_arm_abs_refer = DQ([- 0.009809, - 0.700866, - 0.008828, 0.713171, 0.097394, - 0.000512, - 0.184699, - 0.00145])
        if self.i < 2000:
            dual_arm_abs_refer = DQ([- 0.009809, - 0.700866, - 0.008828, 0.713171, 0.097394, - 0.000512, - 0.184699, - 0.00145])
        dual_arm_joint_angle = np.concatenate((self.ur3_current_joint_positions, self.ur3e_current_joint_positions))
        # print("dual arm relative_pose", self.dq_dual_arm_model.relative_pose(dual_arm_joint_angle))
        # print("dual arm absolute_pose", self.dq_dual_arm_model.absolute_pose(dual_arm_joint_angle))
        # print("dual arm absolute_position", self.get_dq_position(self.dq_dual_arm_model.absolute_pose(dual_arm_joint_angle)))
        [dual_arm_joint_vel, dual_arm_jacobian] = self.dual_arm_relative_pose_control(dual_arm_joint_angle, dual_arm_rel_refer, gain=0.5)
        [dual_arm_joint_vel_add, dual_arm_jacobian_add] =self.dual_arm_absolute_pose_control(dual_arm_joint_angle, dual_arm_abs_refer, gain=0.5)
        dual_arm_joint_vel = dual_arm_joint_vel +np.matmul(np.eye(12)-np.linalg.pinv(dual_arm_jacobian)@(dual_arm_jacobian),dual_arm_joint_vel_add)
        ur3_joint_vel = dual_arm_joint_vel[:6]
        ur3e_joint_vel = dual_arm_joint_vel[6:]
        if self.i > 2000:
            ur3_joint_vel = np.array([0.0]*6)
            ur3e_joint_vel = np.array([0.0]*6)
        self.ur3_vel_msg.velocity = ur3_joint_vel.tolist()
        self.ur3e_vel_msg.velocity = ur3e_joint_vel.tolist()
        # print("ur3",self.ur3_vel_msg.velocity)
        # print("ur3e",self.ur3e_vel_msg.velocity)
        self.publisher_ur3_velocity.publish(self.ur3_vel_msg)
        self.publisher_ur3e_velocity.publish(self.ur3e_vel_msg)
        self.ur3_init_flag = False
        self.ur3e_init_flag = False

    def dual_arm_relative_pose_control(self, dual_arm_joint_pos:np.ndarray, dual_arm_rel_refer:DQ, gain:float=0.5) -> list:
        dual_arm_rel_jacobian = self.dq_dual_arm_model.relative_pose_jacobian(dual_arm_joint_pos)
        dual_arm_rel_refer = dual_arm_rel_refer.normalize()
        dual_arm_rel_refer = vec8(dual_arm_rel_refer)
        dual_arm_rel_feedback = vec8(self.dq_dual_arm_model.relative_pose(dual_arm_joint_pos))
        dual_arm_rel_error = dual_arm_rel_refer - dual_arm_rel_feedback
        dual_arm_rel_jacobian_roboust_inv = dual_arm_rel_jacobian.T @ np.linalg.pinv(np.matmul(dual_arm_rel_jacobian, dual_arm_rel_jacobian.T) + 1e-8 * np.eye(8))
        dual_arm_rel_joint_vel = gain * np.matmul(dual_arm_rel_jacobian_roboust_inv, (dual_arm_rel_error))
        return [dual_arm_rel_joint_vel, dual_arm_rel_jacobian]

    def dual_arm_absolute_pose_control(self, dual_arm_joint_pos:np.ndarray, dual_arm_abs_refer:np.array, gain:float=0.5) -> list:
        dual_arm_abs_jacobian = self.dq_dual_arm_model.absolute_pose_jacobian(dual_arm_joint_pos)
        dual_arm_abs_feedback = vec8(self.dq_dual_arm_model.absolute_pose(dual_arm_joint_pos))
        dual_arm_abs_refer = vec8(dual_arm_abs_refer)
        dual_arm_abs_error = dual_arm_abs_refer - dual_arm_abs_feedback
        dual_arm_abs_dis_joint_vel = gain * np.linalg.pinv(dual_arm_abs_jacobian) @ dual_arm_abs_error
        return [dual_arm_abs_dis_joint_vel, dual_arm_abs_jacobian]

    def get_dq_position(self,dq:DQ):
        dq_p = dq.P()
        dq_d = dq.D()
        position =2*dq_d*dq_p.inv()
        return position

    def from_position2dq(self,dq:DQ):
        dq_p = dq.P()
        position = 0.335385*i_+ 0.013657*j_ + 0.31*k_
        return dq_p+E_*0.5*position*dq_p

def main(args=None):
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)
    dual_arm_model= DualArmDQController()
    rclpy.spin(dual_arm_model)

if __name__ == "__main__":
    main()


