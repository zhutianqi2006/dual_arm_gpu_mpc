# curobo for collision detection
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
desire_abs_pose = DQ([- 0.009809, - 0.700866, - 0.008828, 0.713171, 0.03289, - 0.000662, - 0.283115, - 0.003703])
desire_abs_pose = DQ([0.005744, - 0.683663, 0.020305, 0.729493, 0.017152, - 0.04017, - 0.237569, - 0.031169])

desire_abs_pose = DQ([-0.011961693390801373, -0.701712136452997, -0.008064925867787953, 0.7123145038650708, 0.12203720037624906, -0.002276549048810388, -0.1960031380395545, -0.0024125035955698537])

desire_abs_pose = DQ([0.00085, 0.923642, -0.383209, -0.005971, - 0.077555, 0.113491, 0.278521, - 0.330388])

desire_abs_pose = desire_abs_pose.normalize()
desire_abs_pose_p = desire_abs_pose.P()
desire_abs_pose_d = desire_abs_pose.D()
pos = 2*desire_abs_pose_d*desire_abs_pose_p.inv()
print("deisre_position:",pos)
deisre_l = desire_abs_pose_p*DQ([0,0,0,1.0])*desire_abs_pose_p.conj()
print("deisre_l:",deisre_l)
desire_abs_position = DQ([0.0, -0.4, 0.0, 0.10, 0.0,  0.0, 0.0, 0.0])
desire_abs_position = DQ([0.0, 0.45, 0.0, 0.52, 0.0,  0.0, 0.0, 0.0])
desire_abs_position = DQ([0.0, 0.32, -0.25, 0.40, 0.0,  0.0, 0.0, 0.0])
desire_abs_position = DQ([0.0, 0.32, -0.20, 0.40, 0.0,  0.0, 0.0, 0.0])
desire_abs_rot = desire_abs_pose.P()
print("desire_abs_prim:", desire_abs_rot)
print("desire_abs_dual:", vec4(0.5*desire_abs_position*desire_abs_rot))
