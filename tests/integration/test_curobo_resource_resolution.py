from dual_arm_gpu_mpc.common.paths import project_root
from dual_arm_gpu_mpc.robotics.collision.curobo_loader import resolve_curobo_config_paths


def test_curobo_resource_resolution_uses_absolute_paths():
    robot_path, world_path = resolve_curobo_config_paths("dual_arm_model_real.yml", "dual_arm_collision_env.yml")

    assert robot_path.is_absolute()
    assert world_path.is_absolute()
    assert robot_path.exists()
    assert world_path.exists()
    assert project_root() in robot_path.parents
    assert project_root() in world_path.parents

