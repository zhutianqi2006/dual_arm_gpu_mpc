<h1 align="center">
  Dual arm GPU Controller
</h1>
<p align="center">
<p align="center">
  双臂双四元数机械臂GPU控制
</p>
<p align="center">
<a href="README.md">English</a>| 简体中文
</p>

## 环境配置
安装curobo，利用其实现高效并行障碍物碰撞检测

1 保证 pytorch >= 1.10 cuda=11.8

2 克隆 cuRobo 存储库
```sh
git clone https://github.com/NVlabs/curobo
```
3 查看curobo安装流程 并实现安装
https://curobo.org/get_started/1_install_instructions.html

4 clone 本库
```sh
cd cuda_dq_kernel
```
5 编译安装GPU核函数Python接口
```sh
pip install .
```
6 其他要求

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
## 模型放置
1 robot里 文件夹 放urdf 和meshes
```sh
github_source_code/curobo/src/curobo/content/assets/robot
```
2 content里 文件夹有robot、world文件
```sh
github_source_code/curobo/src/curobo/content/configs
```
robot中放机器人的yml文件 world中放环境中的yml文件
## 运行仿真
examples文件夹下
先运仿真环境
```sh
bullet_robot_ros.py
```
再运行底层控制器
```sh
low_level.py
```
最后运行上层控制器
```sh
mppi_xxxxx.py
```
然后可以在bullet界面看到其运动


## 参考项目
| Project | Link|
| --------------------------| ------------------------------------------------------------------------------------- |
| curobo| https://github.com/NVlabs/curobo  |
| dq robotics | https://github.com/dqrobotics/cpp|
|predictive-multi-agent-framework| https://github.com/riddhiman13/predictive-multi-agent-framework
