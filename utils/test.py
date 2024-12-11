# curobo for collision detection
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
# -0.65326264  0.65335195 -0.27051351 -0.27055793 -0.03835287  0.05632752 0.35744052 -0.12875727
# 0.2396i - 0.349987j + 0.645015k
desire_abs_pose = DQ([0.27062871, 0.27062912, -0.65327125, 0.65326615, -0.36436003, -0.09292723, -0.03849804, 0.15094222])
# 0.13539089 -0.83626993 -0.32673176  0.41900868 -0.08358347  0.09311829 -0.07891371  0.1513211
desire_abs_pose = DQ([0.00085, 0.923642, -0.383209, -0.005971, 0.187191, 0.157905, 0.379813, 0.076992])
# desire_abs_pose = DQ([-0.65326264, 0.65335195, -0.27051351, -0.27055793, -0.03835287, 0.05632752, 0.35744052, -0.12875727])
# desire_abs_pose = DQ([0.33135394, 0.46202721, -0.80007434, 0.19135431, 0.07548246, 0.23908415, 0.12138431, -0.20045704])
# desire_abs_pose = DQ([0.33138082, 0.46210602, -0.79999345,  0.19145563,  0.07547732,  0.23909355, 0.121403, -0.20044695])
# desire_abs_pose = DQ([0.56105573,  0.56099693, -0.43044662, -0.43036569,  0.0561724,   0.09773239, 0.31534998, -0.11478126]) 
desire_abs_pose = desire_abs_pose.normalize()
desire_abs_pose_p = desire_abs_pose.P()
desire_abs_pose_d = desire_abs_pose.D()
pos = 2*desire_abs_pose_d*desire_abs_pose_p.inv()
print("deisre_position:",pos)
deisre_l = desire_abs_pose_p*DQ([0,0,0,1.0])*desire_abs_pose_p.conj()
print("deisre_l:",deisre_l)
desire_abs_position = DQ([0.0, -0.4, 0.0, 0.68501, 0.0,  0.0, 0.0, 0.0])
desire_abs_rot = desire_abs_pose.P()
print("desire_abs_prim:", desire_abs_rot)
print("desire_abs_dual:", 0.5*desire_abs_position*desire_abs_rot)
