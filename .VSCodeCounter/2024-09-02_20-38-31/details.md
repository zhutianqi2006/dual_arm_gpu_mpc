# Details

Date : 2024-09-02 20:38:31

Directory /home/echoz/upload_code/dual_arm_dq_controller

Total : 111 files,  984805 codes, 1846 comments, 1726 blanks, all 988377 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [111.py](/111.py) | Python | 35 | 4 | 1 | 40 |
| [b_urdf_structs_example.py](/b_urdf_structs_example.py) | Python | 33 | 4 | 1 | 38 |
| [common/math_extension.py](/common/math_extension.py) | Python | 4 | 0 | 0 | 4 |
| [cuda_dq_kernel/dq._kerner.cu](/cuda_dq_kernel/dq._kerner.cu) | CUDA C++ | 1,143 | 86 | 134 | 1,363 |
| [cuda_dq_kernel/dq.cpp](/cuda_dq_kernel/dq.cpp) | C++ | 116 | 0 | 4 | 120 |
| [cuda_dq_kernel/dq_cooperativedualtaskspace.py](/cuda_dq_kernel/dq_cooperativedualtaskspace.py) | Python | 140 | 16 | 6 | 162 |
| [cuda_dq_kernel/dq_cop_test.py](/cuda_dq_kernel/dq_cop_test.py) | Python | 44 | 15 | 3 | 62 |
| [cuda_dq_kernel/dq_torch.py](/cuda_dq_kernel/dq_torch.py) | Python | 228 | 45 | 31 | 304 |
| [cuda_dq_kernel/include/dq_utils.h](/cuda_dq_kernel/include/dq_utils.h) | C++ | 27 | 0 | 2 | 29 |
| [cuda_dq_kernel/install.log](/cuda_dq_kernel/install.log) | Log | 4,591 | 0 | 34 | 4,625 |
| [cuda_dq_kernel/setup.py](/cuda_dq_kernel/setup.py) | Python | 22 | 0 | 0 | 22 |
| [cuda_dq_kernel/test/cpu_dq_cop.py](/cuda_dq_kernel/test/cpu_dq_cop.py) | Python | 38 | 7 | 2 | 47 |
| [cuda_dq_kernel/test/dq_cooperativedualtaskspace.py](/cuda_dq_kernel/test/dq_cooperativedualtaskspace.py) | Python | 98 | 20 | 7 | 125 |
| [cuda_dq_kernel/test/dq_cop_test.py](/cuda_dq_kernel/test/dq_cop_test.py) | Python | 43 | 15 | 3 | 61 |
| [cuda_dq_kernel/test/dq_cop_test_new.py](/cuda_dq_kernel/test/dq_cop_test_new.py) | Python | 46 | 18 | 3 | 67 |
| [cuda_dq_kernel/test/dq_torch.py](/cuda_dq_kernel/test/dq_torch.py) | Python | 228 | 45 | 31 | 304 |
| [cuda_dq_kernel/test/test_ad.py](/cuda_dq_kernel/test/test_ad.py) | Python | 27 | 2 | 2 | 31 |
| [cuda_dq_kernel/test/test_conj.py](/cuda_dq_kernel/test/test_conj.py) | Python | 20 | 2 | 2 | 24 |
| [cuda_dq_kernel/test/test_d.py](/cuda_dq_kernel/test/test_d.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_dq_exp.py](/cuda_dq_kernel/test/test_dq_exp.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_dq_inv.py](/cuda_dq_kernel/test/test_dq_inv.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_dq_log.py](/cuda_dq_kernel/test/test_dq_log.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_dq_normalize.py](/cuda_dq_kernel/test/test_dq_normalize.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_dq_sqrt.py](/cuda_dq_kernel/test/test_dq_sqrt.py) | Python | 33 | 2 | 1 | 36 |
| [cuda_dq_kernel/test/test_haminus8.py](/cuda_dq_kernel/test/test_haminus8.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_hamiplus8.py](/cuda_dq_kernel/test/test_hamiplus8.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_im.py](/cuda_dq_kernel/test/test_im.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_multi.py](/cuda_dq_kernel/test/test_multi.py) | Python | 27 | 0 | 3 | 30 |
| [cuda_dq_kernel/test/test_norm.py](/cuda_dq_kernel/test/test_norm.py) | Python | 23 | 2 | 2 | 27 |
| [cuda_dq_kernel/test/test_p.py](/cuda_dq_kernel/test/test_p.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_re.py](/cuda_dq_kernel/test/test_re.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_rotation_angle.py](/cuda_dq_kernel/test/test_rotation_angle.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_rotation_axis.py](/cuda_dq_kernel/test/test_rotation_axis.py) | Python | 24 | 2 | 2 | 28 |
| [cuda_dq_kernel/test/test_translation.py](/cuda_dq_kernel/test/test_translation.py) | Python | 27 | 2 | 2 | 31 |
| [cuda_dq_kernel/torch_inv_and_cupy_inv_test.py](/cuda_dq_kernel/torch_inv_and_cupy_inv_test.py) | Python | 22 | 2 | 5 | 29 |
| [cuda_dq_test/dq._kerner.cu](/cuda_dq_test/dq._kerner.cu) | CUDA C++ | 45 | 36 | 10 | 91 |
| [cuda_dq_test/dq.cpp](/cuda_dq_test/dq.cpp) | C++ | 13 | 0 | 2 | 15 |
| [cuda_dq_test/include/utils.h](/cuda_dq_test/include/utils.h) | C++ | 5 | 0 | 2 | 7 |
| [cuda_dq_test/setup.py](/cuda_dq_test/setup.py) | Python | 20 | 0 | 0 | 20 |
| [cuda_dq_test/test.py](/cuda_dq_test/test.py) | Python | 6 | 0 | 1 | 7 |
| [cuda_dqmppi/cpu_dq_robotics.py](/cuda_dqmppi/cpu_dq_robotics.py) | Python | 20 | 0 | 1 | 21 |
| [cuda_dqmppi/cpu_dq_test.py](/cuda_dqmppi/cpu_dq_test.py) | Python | 13 | 0 | 1 | 14 |
| [cuda_dqmppi/dq.py](/cuda_dqmppi/dq.py) | Python | 180 | 61 | 44 | 285 |
| [cuda_dqmppi/dq_cooperativedualtaskspace.py](/cuda_dqmppi/dq_cooperativedualtaskspace.py) | Python | 40 | 0 | 13 | 53 |
| [cuda_dqmppi/dq_kinematics.py](/cuda_dqmppi/dq_kinematics.py) | Python | 27 | 6 | 8 | 41 |
| [cuda_dqmppi/dq_serialmanipulator.py](/cuda_dqmppi/dq_serialmanipulator.py) | Python | 77 | 9 | 23 | 109 |
| [cuda_dqmppi/dq_serialmanipulatordh.py](/cuda_dqmppi/dq_serialmanipulatordh.py) | Python | 80 | 6 | 15 | 101 |
| [cuda_dqmppi/dq_serialmanipulatordh_test.py](/cuda_dqmppi/dq_serialmanipulatordh_test.py) | Python | 35 | 11 | 1 | 47 |
| [cuda_dqmppi/dq_test.py](/cuda_dqmppi/dq_test.py) | Python | 14 | 3 | 2 | 19 |
| [cuda_dqmppi/dq_test_new.py](/cuda_dqmppi/dq_test_new.py) | Python | 20 | 0 | 4 | 24 |
| [cuda_dqmppi/dq_utils.py](/cuda_dqmppi/dq_utils.py) | Python | 192 | 60 | 29 | 281 |
| [cuda_dqmppi/test.py](/cuda_dqmppi/test.py) | Python | 12 | 1 | 2 | 15 |
| [cuda_dqmppi_new/cpu_dq_cop.py](/cuda_dqmppi_new/cpu_dq_cop.py) | Python | 37 | 0 | 2 | 39 |
| [cuda_dqmppi_new/cpu_dq_robotics.py](/cuda_dqmppi_new/cpu_dq_robotics.py) | Python | 14 | 2 | 1 | 17 |
| [cuda_dqmppi_new/cpu_dq_test.py](/cuda_dqmppi_new/cpu_dq_test.py) | Python | 13 | 0 | 1 | 14 |
| [cuda_dqmppi_new/dq_cooperativedualtaskspace.py](/cuda_dqmppi_new/dq_cooperativedualtaskspace.py) | Python | 140 | 16 | 6 | 162 |
| [cuda_dqmppi_new/dq_cop_test.py](/cuda_dqmppi_new/dq_cop_test.py) | Python | 43 | 15 | 2 | 60 |
| [cuda_dqmppi_new/dq_kinematics.py](/cuda_dqmppi_new/dq_kinematics.py) | Python | 23 | 0 | 8 | 31 |
| [cuda_dqmppi_new/dq_serialmanipulator.py](/cuda_dqmppi_new/dq_serialmanipulator.py) | Python | 59 | 17 | 24 | 100 |
| [cuda_dqmppi_new/dq_serialmanipulatordh.py](/cuda_dqmppi_new/dq_serialmanipulatordh.py) | Python | 91 | 14 | 18 | 123 |
| [cuda_dqmppi_new/dq_serialmanipulatordh_test.py](/cuda_dqmppi_new/dq_serialmanipulatordh_test.py) | Python | 38 | 12 | 3 | 53 |
| [cuda_dqmppi_new/dq_test.py](/cuda_dqmppi_new/dq_test.py) | Python | 21 | 5 | 2 | 28 |
| [cuda_dqmppi_new/dq_test_new.py](/cuda_dqmppi_new/dq_test_new.py) | Python | 20 | 0 | 4 | 24 |
| [cuda_dqmppi_new/dq_torch.py](/cuda_dqmppi_new/dq_torch.py) | Python | 218 | 55 | 32 | 305 |
| [cuda_dqmppi_new/m.py](/cuda_dqmppi_new/m.py) | Python | 2 | 0 | 0 | 2 |
| [cuda_dqmppi_new/test.py](/cuda_dqmppi_new/test.py) | Python | 10 | 0 | 0 | 10 |
| [cuda_planning/bullet_robot_ros.py](/cuda_planning/bullet_robot_ros.py) | Python | 116 | 10 | 13 | 139 |
| [cuda_planning/dual_arm_node_ros2.py](/cuda_planning/dual_arm_node_ros2.py) | Python | 218 | 31 | 34 | 283 |
| [cuda_planning/new_planning.py](/cuda_planning/new_planning.py) | Python | 369 | 66 | 50 | 485 |
| [cuda_planning/new_planning_adp.py](/cuda_planning/new_planning_adp.py) | Python | 387 | 57 | 55 | 499 |
| [cuda_planning/new_planning_adp_ros.py](/cuda_planning/new_planning_adp_ros.py) | Python | 403 | 42 | 54 | 499 |
| [cuda_planning/new_planning_efe.py](/cuda_planning/new_planning_efe.py) | Python | 456 | 78 | 66 | 600 |
| [curobo_model/dual_arm_collision_env.yml](/curobo_model/dual_arm_collision_env.yml) | YAML | 16 | 0 | 6 | 22 |
| [curobo_model/dual_arm_model.yml](/curobo_model/dual_arm_model.yml) | YAML | 120 | 0 | 7 | 127 |
| [curobo_model/dual_arm_model/dual_arm_model.urdf](/curobo_model/dual_arm_model/dual_arm_model.urdf) | XML | 459 | 26 | 9 | 494 |
| [dual_arm_model/dual_arm.yaml](/dual_arm_model/dual_arm.yaml) | YAML | 109 | 11 | 9 | 129 |
| [dual_arm_model/dual_arm_collision_checker.py](/dual_arm_model/dual_arm_collision_checker.py) | Python | 20 | 9 | 5 | 34 |
| [dual_arm_model/dual_arm_model.urdf](/dual_arm_model/dual_arm_model.urdf) | XML | 459 | 26 | 9 | 494 |
| [dual_arm_model/dual_arm_model.yml](/dual_arm_model/dual_arm_model.yml) | YAML | 136 | 0 | 9 | 145 |
| [dual_arm_model/dual_arm_model_display.py](/dual_arm_model/dual_arm_model_display.py) | Python | 104 | 24 | 33 | 161 |
| [dual_arm_model/helper.py](/dual_arm_model/helper.py) | Python | 120 | 21 | 30 | 171 |
| [dual_arm_model/ur3/ur3.urdf](/dual_arm_model/ur3/ur3.urdf) | XML | 280 | 31 | 1 | 312 |
| [dual_arm_model/ur3e/ur3e.urdf](/dual_arm_model/ur3e/ur3e.urdf) | XML | 289 | 76 | 1 | 366 |
| [install.log](/install.log) | Log | 27 | 0 | 1 | 28 |
| [model/object/plane.urdf](/model/object/plane.urdf) | XML | 31 | 0 | 0 | 31 |
| [model/plane/plane.urdf](/model/plane/plane.urdf) | XML | 31 | 0 | 0 | 31 |
| [model/ur3/ur3.urdf](/model/ur3/ur3.urdf) | XML | 280 | 31 | 1 | 312 |
| [model/ur3e/ur3e.urdf](/model/ur3e/ur3e.urdf) | XML | 289 | 76 | 1 | 366 |
| [model_new/ur3.yml](/model_new/ur3.yml) | YAML | 83 | 0 | 7 | 90 |
| [model_new/ur3/ur3.urdf](/model_new/ur3/ur3.urdf) | XML | 280 | 33 | 1 | 314 |
| [model_new/ur3e/ur3e.urdf](/model_new/ur3e/ur3e.urdf) | XML | 289 | 76 | 1 | 366 |
| [motion_planning/barebone_mppi_numba.ipynb](/motion_planning/barebone_mppi_numba.ipynb) | JSON | 858 | 0 | 1 | 859 |
| [motion_planning/dstar.py](/motion_planning/dstar.py) | Python | 178 | 0 | 32 | 210 |
| [motion_planning/lqr_planner.py](/motion_planning/lqr_planner.py) | Python | 99 | 6 | 45 | 150 |
| [motion_planning/mppi.py](/motion_planning/mppi.py) | Python | 258 | 42 | 64 | 364 |
| [motion_planning/mppi_adp.py](/motion_planning/mppi_adp.py) | Python | 266 | 42 | 64 | 372 |
| [motion_planning/mppi_adp_multi.py](/motion_planning/mppi_adp_multi.py) | Python | 271 | 28 | 27 | 326 |
| [motion_planning/mppi_adp_multi_bullet.py](/motion_planning/mppi_adp_multi_bullet.py) | Python | 644 | 68 | 76 | 788 |
| [motion_planning/mppi_adp_multi_bullet_new.py](/motion_planning/mppi_adp_multi_bullet_new.py) | Python | 691 | 63 | 83 | 837 |
| [motion_planning/mppi_example.py](/motion_planning/mppi_example.py) | Python | 262 | 62 | 73 | 397 |
| [motion_planning/mppi_example2.py](/motion_planning/mppi_example2.py) | Python | 77 | 4 | 24 | 105 |
| [motion_planning/rrt.py](/motion_planning/rrt.py) | Python | 253 | 11 | 58 | 322 |
| [motion_planning/rrt_pin.py](/motion_planning/rrt_pin.py) | Python | 375 | 46 | 66 | 487 |
| [motion_planning/rrt_star.py](/motion_planning/rrt_star.py) | Python | 234 | 7 | 41 | 282 |
| [motion_planning/ziyouneng.py](/motion_planning/ziyouneng.py) | Python | 55 | 4 | 19 | 78 |
| [number_test.py](/number_test.py) | Python | 266 | 28 | 28 | 322 |
| [test1.py](/test1.py) | Python | 257 | 27 | 21 | 305 |
| [test2.py](/test2.py) | Python | 258 | 27 | 20 | 305 |
| [test3.py](/test3.py) | Python | 21 | 7 | 7 | 35 |
| [tora.py](/tora.py) | Python | 36 | 10 | 9 | 55 |
| [trace.json](/trace.json) | JSON | 965,244 | 0 | 1 | 965,245 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)