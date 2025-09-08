#!/usr/bin/env python
# system library
import os
import sys
import time
import numpy as np
import math
from math import pi
import threading
# curobo for collision detection
from dqrobotics import i_, j_, k_, E_, DQ, vec8 ,vec4
from dqrobotics.robot_modeling import DQ_SerialManipulatorDH, DQ_CooperativeDualTaskSpace
# DQ Robotics used in cuda
from utils.config_module import ConfigModule
from utils.low_ros_module import LowROSModule
from utils.low_level_rel_module import LowLevelRelChangeModule
import rclpy

class _KeyReader:
    def __enter__(self):
        if os.name == 'nt':
            import msvcrt  # noqa
            self._mode = 'win'
        else:
            import termios, tty  # type: ignore
            self._mode = 'posix'
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self._mode == 'posix':
            import termios  # type: ignore
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read_key(self):
        if self._mode == 'win':
            import msvcrt  # type: ignore
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    return ch.decode('utf-8')
                except Exception:
                    return None
            return None
        else:
            import select
            dr, _, _ = select.select([sys.stdin], [], [], 0)
            if dr:
                ch = sys.stdin.read(1)
                return ch
            return None


def _start_keyboard_listener(low_level_module: LowLevelRelChangeModule, orig_rel_pose, alt_rel_pose):
    def _loop():
        print("[keys] 按 '1' 使用备用 desire_rel_pose，按 '2' 恢复原始，按 'q' 退出\n")
        with _KeyReader() as kr:
            while rclpy.ok():
                ch = kr.read_key()
                if not ch:
                    time.sleep(0.01)
                    continue
                if ch == '1':
                    low_level_module.set_desire_rel_pose(alt_rel_pose)
                    print("[keys] 已切换: desire_rel_pose = 备用值")
                elif ch == '2':
                    low_level_module.set_desire_rel_pose(orig_rel_pose)
                    print("[keys] 已恢复: desire_rel_pose = 原始值")
                elif ch in ('q', 'Q'):
                    print("[keys] 退出进程")
                    os._exit(0)
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t

def main(args=None):
    # 步骤 3: 实例化DQ_SerialManipulatorDH
    os.environ['ROS_DOMAIN_ID'] = '16'
    rclpy.init(args=args)

    # 初始期望位姿（绝对/相对）
    desire_abs_pose = [0.055365, -0.588516, 0.051009, 0.804973, -0.025382, -0.005936, -0.240929, 0.012673]
    desire_rel_pose = [0.043815, 0.998793, 0.006783, 0.021159, 0.001626, -0.002018, 0.28023, 0.00204]
    dq_desire_rel_pose = DQ(desire_rel_pose)
    dq_desire_rel_pose = dq_desire_rel_pose.normalize()
    #dq_alt_rel_pose = [-0.067376, 0.935849, 0.006023, -0.345848, -0.006123, -0.001464, 0.285689, 0.002207]
    # 备用（按 1 时切换到该值）——请按需修改
    alt_rel_pose = [-0.067376, 0.935849, 0.006023, -0.345848, -0.006123, -0.001464, 0.285689, 0.002207]

    config_path = os.path.join(os.path.dirname(__file__), 'ur3_and_ur3e.yaml')
    config = ConfigModule(config_path)

    low_level_module = LowLevelRelChangeModule(config, desire_abs_pose, desire_rel_pose)

    # 启动按键监听线程（不阻塞控制循环）
    _start_keyboard_listener(low_level_module, desire_rel_pose, alt_rel_pose)

    # 控制循环
    while rclpy.ok():
        low_level_module.play_once()


if __name__ == "__main__":
    main()