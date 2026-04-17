<h1 align="center">
  Dual Arm GPU Controller
</h1>
<p align="center">
  双臂双四元数机械臂 GPU 控制
</p>
<p align="center">
<a href="README.md">English</a> | 简体中文
</p>

## 项目说明

这个工程主要依赖三部分：

- `cuda_dq_kernel`
  提供 CUDA 双四元数与双臂运动学核函数
- `curobo`
  提供 GPU 碰撞检测
- ROS 2
  提供在线控制和消息通信

这个仓库不需要安装 Isaac Sim。这里使用的是 `curobo` 的纯 Python / library 安装路径。

## 已验证环境

当前仓库在下面这套环境中做过本地验证：

- Ubuntu 22.04
- Python 3.10.12
- `uv 0.7.19`
- PyTorch `2.9.1+cu126`
- CUDA toolkit `12.6`
- `curobo` 通过源码 `pip install -e . --no-build-isolation` 安装

## 依赖分层

建议把环境理解成三层：

1. CUDA + PyTorch
   `dq_torch` 和 `curobo` 都需要
2. 机器人学 Python 依赖
   包括 `dqrobotics`、`kmeans-pytorch` 以及 `curobo` 的 Python 依赖
3. ROS 2
   高层控制器和 ROS bridge 需要

## 推荐安装方式：uv

现在 `dual_arm_gpu_mpc` 目录里已经提供了项目级 `uv` 入口：

- [`pyproject.toml`](/home/echoz/2026_tro_dual_arm_code/dual_arm_gpu_mpc/pyproject.toml)
- [`scripts/bootstrap_uv.sh`](/home/echoz/2026_tro_dual_arm_code/dual_arm_gpu_mpc/scripts/bootstrap_uv.sh)

在 `dual_arm_gpu_mpc/` 目录执行：

```bash
cd dual_arm_gpu_mpc
export CUDA_HOME=/usr/local/cuda-12.6
uv sync
bash scripts/bootstrap_uv.sh
```

这两步分别负责：

- `uv sync`
  创建本地 `.venv`，安装纯 Python 依赖和 PyTorch `cu126`
- `bootstrap_uv.sh`
  安装仍然需要本地构建的源码依赖：
  - 会先把 `dual_arm_gpu_mpc` 自身以 editable 方式装进 `.venv`
  - `../cuda_dq_kernel` 会以非 editable 方式安装，保证 `.venv` 中可以直接导入 `dq_torch`
  - 如果存在 `../../third_party/curobo`，会以 editable 方式一起安装

如果你修改了 Python 依赖，建议执行：

```bash
cd dual_arm_gpu_mpc
uv lock
uv sync
```

## 安装 `curobo`，不需要 Isaac Sim

`curobo` 官方安装文档明确支持 library-only 安装，并且说明 Isaac Sim 不是必需依赖：
https://curobo.org/get_started/1_install_instructions.html

先准备：

```bash
sudo apt install git-lfs
git lfs install
```

然后克隆并安装：

```bash
git clone https://github.com/NVlabs/curobo.git third_party/curobo
cd dual_arm_gpu_mpc
export CUDA_HOME=/usr/local/cuda-12.6
bash scripts/bootstrap_uv.sh
```

当前机器上的本地验证使用的是混合策略：

- `dq_torch` 通过本地源码非 editable 安装
- `curobo` 通过 editable + `--no-build-isolation` 安装

这是刻意这样设计的。`cuda_dq_kernel` 当前更适合作为 workspace 里的本地非 editable 构建，而 `curobo` 则适合继续保持 editable 安装。

根据官方文档，当前这条安装路径有几个要点：

- 推荐 Ubuntu 20.04 或 22.04
- 推荐 Python 3.10
- clone `curobo` 前先装好 `git-lfs`
- 只有 Isaac Sim 集成才需要 Isaac Sim 本体

## ROS 2 要求

`dual_arm_gpu_mpc` 里的 ROS 模块会导入：

- `rclpy`
- `sensor_msgs`
- `std_msgs`
- `geometry_msgs`

如果你只是离线编译 `dq_torch`、测试 CUDA 核函数，ROS 2 不是必需的。  
如果你要运行高层控制器，ROS 2 是必需的。

Ubuntu 22.04 示例：

```bash
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

当前仓库里像 `src/dual_arm_gpu_mpc/ros/high.py`、`src/dual_arm_gpu_mpc/ros/low.py`，以及 `src/dual_arm_gpu_mpc/controllers/` 下的 MPPI 入口模块，即使 `curobo` 已经装好，如果没有 `rclpy` 仍然会在 import 阶段失败。

## Conda 兜底方案

[`environment.yml`](/home/echoz/2026_tro_dual_arm_code/dual_arm_gpu_mpc/environment.yml) 现在放在 `dual_arm_gpu_mpc` 目录下，建议在该目录执行：

```bash
cd dual_arm_gpu_mpc
conda env create -f environment.yml
conda activate dual-arm-gpu
```

这份 yaml 只是一个 bootstrap 环境，还需要你继续：

- 安装带 CUDA 的 PyTorch
- 编译 `cuda_dq_kernel`
- 源码安装 `curobo`
- 如果要运行在线控制器，再单独安装 ROS 2

## 快速验证

验证 CUDA 扩展：

```bash
cd cuda_dq_kernel
python -m pytest tests/pytest/test_import_smoke.py -q
python -m pytest tests/pytest/test_structure_refactor_smoke.py -q
python -m pytest tests/pytest/test_mppi_project_step.py -q
```

验证 `curobo` 核心导入：

```bash
python - <<'PY'
import curobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
print("curobo import ok")
PY
```

## 当前目录规范

这个 workspace 现在主要采用下面这组目录：

- `cuda_dq_kernel`
- `dual_arm_gpu_mpc/src/dual_arm_gpu_mpc`
- `dual_arm_gpu_mpc/configs/{robot,world}`
- `dual_arm_gpu_mpc/analysis/matlab`

## 运行仿真

在 `examples/` 目录中，一般流程是：

- `sim1_ur`：点到点任务，不做避障，作为基础场景
- `sim2_ur`：静态障碍物避障
- `sim3_ur`：动态障碍物避障，Bullet 中有真实移动的障碍物
每个场景通常都按下面顺序运行：

1. 先启动仿真环境
2. 再启动底层控制器
3. 最后启动高层控制器

以 `sim1_ur` 为例，命令如下：

```bash
python examples/sim1_ur/bullet_robot_ros.py
python examples/sim1_ur/low_level.py
python examples/sim1_ur/mppi_kmeans_adpan.py
```

需要开 3 个终端，并按这个顺序启动。`examples/sim2_ur`、`examples/sim3_ur` 和 `examples/exp1_ur` 也是同样的结构。

`examples/sim3_ur` 是带动态障碍物的版本。它的 Bullet 场景里有一个名为 `moving_obstacle` 的长方体障碍物，会沿固定直线来回运动，并把世界坐标系下的位置发布到 `/dock_position_world`；MPPI 会用这个位置实时更新 cuRobo world 里的对应障碍物位姿。

## 参考项目

| Project | Link |
| -------------------------- | ------------------------------------------------------------------------------------- |
| curobo | https://github.com/NVlabs/curobo |
| dq robotics | https://github.com/dqrobotics/cpp |
| predictive-multi-agent-framework | https://github.com/riddhiman13/predictive-multi-agent-framework |
