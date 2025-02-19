
# Python standard lib
import os
import torch
import yaml

class ConfigModule():
    def __init__(self, config_path):
        # Load the YAML file
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        # Set attributes based on the loaded YAML data
        self.ros_node_name = config_data.get('ros_node_name', 'default_node')
        self.ros_update_rate = config_data.get('ros_update_rate', 60)
        
        # Robot 1 configuration
        self.robot1_q_num = config_data.get('robot1_q_num', 0)
        self.robot1_q_name_list = config_data.get('robot1_q_name_list', [])
        self.robot1_q_sub_topic = config_data.get('robot1_q_sub_topic', '/robot1/joint_states')
        self.robot1_dq_pub_topic = config_data.get('robot1_dq_pub_topic', '/robot1/joint_commands')
        self.robot1_dh_type = config_data.get('robot1_dh_type', 0)
        self.robot1_dh_mat = config_data.get('robot1_dh_parameters', [])
        self.robot1_base = config_data.get('robot1_base', [1, 0, 0, 0, 0, 0, 0, 0])
        self.robot1_effector = config_data.get('robot1_effector', [1, 0, 0, 0, 0, 0, 0, 0])
        # Robot 2 configuration
        self.robot2_q_num = config_data.get('robot2_q_num', 0)
        self.robot2_q_name_list = config_data.get('robot2_q_name_list', [])
        self.robot2_q_sub_topic = config_data.get('robot2_q_sub_topic', '/robot2/joint_states')
        self.robot2_dq_pub_topic = config_data.get('robot2_dq_pub_topic', '/robot2/joint_commands')
        self.robot2_dh_type = config_data.get('robot2_dh_type', 0)
        self.robot2_dh_mat = config_data.get('robot2_dh_parameters', [])
        self.robot2_base = config_data.get('robot2_base', [1, 0, 0, 0, 0, 0, 0, 0])
        self.robot2_effector = config_data.get('robot2_effector', [1, 0, 0, 0, 0, 0, 0, 0])
        # MPPI configuration
        self.mppi_dt = config_data.get('mppi_dt', 0.1)
        self.mppi_T = config_data.get('mppi_T', 10)
        self.mppi_seed = config_data.get('mppi_seed', 0)
        self.batch_size = config_data.get('batch_size', 1000)
        self.mean = config_data.get('mean', 0.0)
        self.std = config_data.get('std', 0.1)
        self.gamma = config_data.get('gamma',0.1)
        self.an_std = config_data.get('an_std', 0.1)
        self.log_std = config_data.get('log_std', 0.2)
        self.max_acc_abs_value = config_data.get('max_acc_abs_value', 0.6)
        self.max_abs_tilt_angle = config_data.get('max_abs_tilt_angle',5)
        self.min_collision_distance = config_data.get('min_collision_distance', 0.0)
        self.min_self_collision_distance = config_data.get('min_self_collision_distance', 0.0)
        self.high_level_u_topic = config_data.get('high_level_u_topic', '/high_level_u')
        # MPPI weights load
        self.collision_constraint_weight = config_data.get('collision_constraint_weight', 0.1)
        self.q_limit_constraint_weight = config_data.get('q_limit_constraint_weight', 0.1)
        self.q_vel_constraint_weight = config_data.get('q_vel_constraint_weight', 0.1)
        self.tilt_constraint_weight = config_data.get('tilt_constraint_weight', 0.1)
        self.abs_weight = config_data.get('abs_weight', 0.1)
        self.abs_position_weight = config_data.get('abs_position_weight', 0.1)
        self.terminal_abs_weight = config_data.get('terminal_abs_weight', 0.1)
        self.terminal_abs_position_weight = config_data.get('terminal_abs_position_weight', 0.1)
        self.q_acc_weight = config_data.get('q_acc_weight', 0.1)
        self.q_vel_weight = config_data.get('q_vel_weight', 0.1)
        self.stagnation_weight = config_data.get('stagnation_weight', 0.1)
        self.curobo_world_file = config_data.get('curobo_world_file', 'franka_dynamic_exp3_env.yml')
        self.curobo_robot_file = config_data.get('curobo_robot_file', 'dual_panda_curobo_obstacle.yml')
        # traditional control
        self.high_rel_gain = config_data.get('high_rel_gain', 0.1)
        self.high_abs_gain = config_data.get('high_abs_gain', 0.1)
        self.rel_gain = config_data.get('rel_gain', 0.1)
        self.abs_gain = config_data.get('abs_gain', 0.1)
        # function c to change conrtoller
        self.c_eta = config_data.get('c_eta', 0.1)
        self.c_abs_max = config_data.get('c_abs_max', 0.1)
        # joint position limits
        self.robot1_q_min = config_data.get('robot1_q_min', [-3.0, -3.0, -3.0, -3.0, -3.0, -6.0])
        self.robot1_q_max = config_data.get('robot1_q_max', [3.0, 3.0, 3.0, 3.0, 3.0, 6.0])
        self.robot2_q_min = config_data.get('robot2_q_min', [-3.0, -3.0, -3.0, -3.0, -3.0, -6.0])
        self.robot2_q_max = config_data.get('robot2_q_max', [3.0, 3.0, 3.0, 3.0, 3.0, 6.0])
        # joint velocity limits
        self.robot1_dq_min = config_data.get('robot1_dq_min', [-0.6, -0.6, -0.6, -0.6, -0.6, -0.6])
        self.robot1_dq_max = config_data.get('robot1_dq_max', [0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        self.robot2_dq_min = config_data.get('robot2_dq_min', [-0.6, -0.6, -0.6, -0.6, -0.6, -0.6])
        self.robot2_dq_max = config_data.get('robot2_dq_max', [0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        # joint acc limits
        self.robot1_ddq_min = config_data.get('robot1_ddq_min', [-0.6, -0.6, -0.6, -0.6, -0.6, -0.6])
        self.robot1_ddq_max = config_data.get('robot1_ddq_max', [0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        self.robot2_ddq_min = config_data.get('robot2_ddq_min', [-0.6, -0.6, -0.6, -0.6, -0.6, -0.6])
        self.robot2_ddq_max = config_data.get('robot2_ddq_max', [0.6, 0.6, 0.6, 0.6, 0.6, 0.6])

        
if __name__ == "__main__":
    # Assuming your YAML file is in the same directory as this script
    config_path = os.path.join(os.path.dirname(__file__), 'ur3_and_ur3e.yaml')
    config_module = ConfigModule(config_path)
    
    # Access the DH parameters tensor
    print("DH Parameters as PyTorch Tensor:")
    print(config_module.robot1_dh_mat)
    print(config_module.robot2_dh_mat)
    print(config_module.robot1_base)
