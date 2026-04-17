from pathlib import Path
import sys

_EXAMPLE_PROJECT_ROOT = next(
    (
        candidate
        for candidate in Path(__file__).resolve().parents
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir()
    ),
    None,
)
if _EXAMPLE_PROJECT_ROOT is None:
    raise RuntimeError("Unable to resolve dual_arm_gpu_mpc project root for example script.")
_EXAMPLE_PROJECT_ROOT_STR = str(_EXAMPLE_PROJECT_ROOT)
_EXAMPLE_SRC_ROOT_STR = str(_EXAMPLE_PROJECT_ROOT / "src")
if _EXAMPLE_PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _EXAMPLE_PROJECT_ROOT_STR)
if _EXAMPLE_SRC_ROOT_STR not in sys.path:
    sys.path.insert(0, _EXAMPLE_SRC_ROOT_STR)

from dual_arm_gpu_mpc.common.example_bootstrap import bootstrap_example_paths

bootstrap_example_paths(__file__)
# curobo for collision detection
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
base_frame = DQ([1, 0, 0, 0, 0, 0, 0, 0])
rot = DQ([-0.70711, 0, 0, 0.70711, 0, 0, 0, 0])
rot = rot.normalize()
trans = (1+E_*0.5*(-0.035*i_+0.75*j_+0.07*k_))
a = trans*rot*base_frame
print(a)

base_frame = DQ([1, 0, 0, 0, 0, 0, 0, 0])
rot = DQ([0.70711, 0, 0, 0.70711, 0, 0, 0, 0])
rot = rot.normalize()
trans = (1+E_*0.5*(-0.035*i_-0.75*j_+0.07*k_))
c = trans*rot*base_frame
c = c.normalize()
print(c)


desire_abs_pose = DQ([-0.743697, 0.002919, 0.668503, 0.003245, 0.001775, -0.011929, 0.003932, -0.392559])
desire_abs_pose = desire_abs_pose.normalize()
desire_abs_pose_p = desire_abs_pose.P()
desire_line_d = DQ([0, 0, 0, 1])
a = desire_abs_pose_p*desire_line_d*desire_abs_pose_p.conj()
print("deisre_l:",a)
