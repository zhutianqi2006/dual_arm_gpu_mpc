from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH, DQ_CooperativeDualTaskSpace

desire_abs_pose = [-0.009809, -0.700866, - 0.008828, 0.713171,  0.104515, - 0.00152, - 0.177677, - 0.002256]
desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.001626, - 0.002018, 0.28023, 0.00204]

q = DQ(desire_abs_pose).normalize()
q_inv = DQ(desire_rel_pose).normalize()
q
print(q.inv())