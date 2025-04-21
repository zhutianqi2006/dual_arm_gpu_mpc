<h1 align="center">
  Dual Arm GPU Controller
</h1>
<p align="center">
<p align="center">
  Dual Quaternion Dual-Arm Manipulator GPU Control
</p>
<p align="center">
English | <a href="README_cn.md">简体中文</a> 
</p>

## Environment Setup
Install curobo to achieve efficient parallel collision detection.

1. Ensure pytorch >= 1.10 and cuda=11.8.

2. Clone the cuRobo repository.
```sh
git clone https://github.com/NVlabs/curobo
```
3. Follow the installation instructions for curobo and complete the setup.
https://curobo.org/get_started/1_install_instructions.html

4. Clone this repository.
```sh
cd cuda_dq_kernel
```
5. Compile and install the GPU kernel Python interface.
```sh
pip install .
```
6. Other requirements.

ROS2
```sh
sudo apt install ros-humble-desktop
```
Numpy
```sh
pip install numpy
```
kmeans-pytorch
```sh
pip install kmeans-pytorch
```
## Model Placement
1. Place URDF and meshes in the `robot` folder.
```sh
github_source_code/curobo/src/curobo/content/assets/robot
```
2. The `content` folder contains robot and world configuration files.
```sh
github_source_code/curobo/src/curobo/content/configs
```
Place robot YAML files in the `robot` folder and environment YAML files in the `world` folder.
## Running the Simulation
In the `examples` folder,
first run the simulation environment.
```sh
bullet_robot_ros.py
```
Then run the low-level controller.
```sh
low_level.py
```
Finally, run the high-level controller.
```sh
mppi_xxxxx.py
```
You can observe the motion in the bullet interface.

## Reference Projects
| Project | Link|
| --------------------------| ------------------------------------------------------------------------------------- |
| curobo| https://github.com/NVlabs/curobo  |
| dq robotics | https://github.com/dqrobotics/cpp|
|predictive-multi-agent-framework| https://github.com/riddhiman13/predictive-multi-agent-framework
