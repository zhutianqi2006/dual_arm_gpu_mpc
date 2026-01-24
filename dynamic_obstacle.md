
ros2 run realsense2_camera realsense2_camera_node --ros-args \
  -p enable_color:=true \
  -p enable_depth:=false \
  -p enable_infra1:=false \
  -p enable_infra2:=false \
  -p enable_gyro:=false \
  -p enable_accel:=false \
  -p rgb_camera.profile:="640x480x30"


  ros2 launch apriltag_ros tag_realsense.launch.py