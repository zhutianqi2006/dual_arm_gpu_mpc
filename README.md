<h1 align="center">
  Dual Arm GPU Controller
</h1>
<p align="center">
  Dual Quaternion Dual-Arm Manipulator GPU Control
</p>
<p align="center">
English | <a href="README_cn.md">简体中文</a>
</p>

## Overview

This project uses:

- `dq_torch` under `cuda_dq_kernel` for batched dual-quaternion kinematics on CUDA
- `curobo` for GPU collision checking
- ROS 2 for online robot IO

NVIDIA Isaac Sim is not required for this repository. For collision checking and the MPPI stack here, the Python library installation of `curobo` is enough.

## Verified Baseline

The setup below was verified locally with:

- Ubuntu 22.04
- Python 3.10.12
- `uv 0.7.19`
- PyTorch `2.9.1+cu126`
- CUDA toolkit `12.6`
- `curobo` installed from source with `pip install -e . --no-build-isolation`

## Dependency Split

It helps to think about the environment in three layers:

1. CUDA + PyTorch
   This is required for both `dq_torch` and `curobo`.
2. Robotics Python dependencies
   This includes `dqrobotics`, `kmeans-pytorch`, and the Python packages that `curobo` installs.
3. ROS 2
   This is required for the online controllers and ROS bridge modules under `src/dual_arm_gpu_mpc/ros/`.

## Recommended Setup With uv

`dual_arm_gpu_mpc` now includes a project-local `uv` entry point:

- [`pyproject.toml`](/home/echoz/2026_tro_dual_arm_code/dual_arm_gpu_mpc/pyproject.toml)
- [`scripts/bootstrap_uv.sh`](/home/echoz/2026_tro_dual_arm_code/dual_arm_gpu_mpc/scripts/bootstrap_uv.sh)

From `dual_arm_gpu_mpc/`:

```bash
cd dual_arm_gpu_mpc
export CUDA_HOME=/usr/local/cuda-12.6
uv sync
bash scripts/bootstrap_uv.sh
```

What this does:

- `uv sync` creates the local `.venv` and installs the pure Python dependencies plus PyTorch `cu126`
- `bootstrap_uv.sh` installs local source dependencies that still need builds:
  - installs `dual_arm_gpu_mpc` itself in editable mode
  - installs `../cuda_dq_kernel` as a local non-editable build so `dq_torch` is importable inside `.venv`
  - installs `../../third_party/curobo` as an editable install if that directory exists

If you change Python dependencies, run:

```bash
cd dual_arm_gpu_mpc
uv lock
uv sync
```

## Install `curobo` Without Isaac Sim

`curobo`'s official installation guide explicitly supports library-only installation and states that Isaac Sim is not a required dependency:
https://curobo.org/get_started/1_install_instructions.html

Required preparation:

```bash
sudo apt install git-lfs
git lfs install
```

Clone and install:

```bash
git clone https://github.com/NVlabs/curobo.git third_party/curobo
cd dual_arm_gpu_mpc
export CUDA_HOME=/usr/local/cuda-12.6
bash scripts/bootstrap_uv.sh
```

The local verification on this machine used a mixed strategy:

- `dq_torch` is installed from local source as a non-editable build
- `curobo` is installed editable with `--no-build-isolation`

That split is intentional. `cuda_dq_kernel` currently works best as a non-editable local build for this workspace, while `curobo` works well as an editable source dependency.

Useful note from the official docs:

- `curobo` recommends Ubuntu 20.04 or 22.04 and Python 3.8-3.10
- `git-lfs` should be installed before cloning
- Isaac Sim is only needed for the Isaac Sim integration path, not for the library path

## ROS 2 Requirement

`dual_arm_gpu_mpc` imports `rclpy`, `sensor_msgs`, `std_msgs`, and `geometry_msgs` through the ROS bridge modules. If you only want to build `dq_torch` and test CUDA kernels offline, ROS 2 is not required. If you want to run the high-level controller, ROS 2 is required.

Example for Ubuntu 22.04:

```bash
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

If `rclpy` is missing, ROS-backed modules such as `src/dual_arm_gpu_mpc/ros/high.py`, `src/dual_arm_gpu_mpc/ros/low.py`, and the MPPI entrypoints under `src/dual_arm_gpu_mpc/controllers/` will fail during import even if `curobo` is installed correctly.

## Optional Conda Bootstrap

A fallback [`environment.yml`](/home/echoz/2026_tro_dual_arm_code/dual_arm_gpu_mpc/environment.yml) is provided inside `dual_arm_gpu_mpc`. Run this from that directory:

```bash
cd dual_arm_gpu_mpc
conda env create -f environment.yml
conda activate dual-arm-gpu
```

That file is only a bootstrap environment. You still need to:

- install a CUDA-enabled PyTorch build
- build `cuda_dq_kernel`
- install `curobo` from source
- install ROS 2 separately if you need the online controller

## Quick Verification

Verify the CUDA extension:

```bash
cd cuda_dq_kernel
python -m pytest tests/pytest/test_import_smoke.py -q
python -m pytest tests/pytest/test_structure_refactor_smoke.py -q
python -m pytest tests/pytest/test_mppi_project_step.py -q
```

Verify `curobo` core imports:

```bash
python - <<'PY'
import curobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
print("curobo import ok")
PY
```

## Project Layout

This workspace now uses:

- `cuda_dq_kernel` for the CUDA extension
- `dual_arm_gpu_mpc/src/dual_arm_gpu_mpc` for the Python application package
- `dual_arm_gpu_mpc/configs/{robot,world}` for project-local curobo configuration files
- `dual_arm_gpu_mpc/analysis/matlab` for MATLAB analysis artifacts

## Running The Simulation

Inside `examples/`:

- `sim1_ur`: point-to-point motion without obstacle avoidance as the baseline task
- `sim2_ur`: static obstacle avoidance
- `sim3_ur`: dynamic obstacle avoidance with a moving obstacle in Bullet
For each scenario, the usual process is:

1. start the simulation environment
2. run the low-level controller
3. run the high-level controller

Representative commands for `sim1_ur`:

```bash
python examples/sim1_ur/bullet_robot_ros.py
python examples/sim1_ur/low_level.py
python examples/sim1_ur/mppi_kmeans_adpan.py
```

Use three separate terminals and start them in that order. The same structure is available under `examples/sim2_ur`, `examples/sim3_ur`, and `examples/exp1_ur`.

`examples/sim3_ur` is the dynamic-obstacle variant. Its Bullet scene includes a cuboid obstacle named `moving_obstacle` that travels back and forth along a fixed line while publishing its world-frame position to `/dock_position_world`, and the MPPI stack updates the cuRobo world obstacle pose from that stream.

## References

| Project | Link |
| -------------------------- | ------------------------------------------------------------------------------------- |
| curobo | https://github.com/NVlabs/curobo |
| dq robotics | https://github.com/dqrobotics/cpp |
| predictive-multi-agent-framework | https://github.com/riddhiman13/predictive-multi-agent-framework |
