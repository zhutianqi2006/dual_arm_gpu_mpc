import argparse
import time
import threading
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface


JOINTS = [f"J{i+1}" for i in range(6)]


def ema_filter(x_prev: np.ndarray, x_new: np.ndarray, alpha: float) -> np.ndarray:
    """指数滑动平均滤波。alpha∈[0,1]，越大越灵敏。"""
    if x_prev is None:
        return x_new.copy()
    return alpha * x_new + (1.0 - alpha) * x_prev


def control_thread_fn(host: str, speed: List[float], accel: float, segment_time: float, cycles: int = 1):
    """以很小的 speedJ 让机械臂动一下：+v 保持一段时间，再 -v 保持一段时间，循环 cycles 次。"""
    rtde_c = RTDEControlInterface(host)
    try:
        v = list(map(float, speed))
        v_neg = [-vi for vi in v]
        for _ in range(max(1, cycles)):
            ok = rtde_c.speedJ(v, accel, segment_time)
            time.sleep(segment_time + 0.05)
            ok = rtde_c.speedJ(v_neg, accel, segment_time)
            time.sleep(segment_time + 0.05)
        # 停车
        rtde_c.speedStop()
    finally:
        try:
            rtde_c.disconnect()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', type=str, required=True, help='UR 控制器 IP')
    ap.add_argument('--duration', type=float, default=8.0, help='采集时长 (s)')
    ap.add_argument('--alpha', type=float, default=0.25, help='EMA 滤波系数 (0~1)，对加速度')
    ap.add_argument('--do_control', type=int, default=0, help='是否在采集时轻微 speedJ 控制 (1 开启)')
    ap.add_argument('--speed', type=str, default='0 0 0 0 0 0.5', help='speedJ 关节速度向量 (rad/s)，用空格分隔 6 个数')
    ap.add_argument('--accel', type=float, default=1.0, help='speedJ 加速度上限 (rad/s^2)')
    ap.add_argument('--segment_time', type=float, default=2.0, help='每段 +v 或 -v 的持续时间 (s)')
    ap.add_argument('--cycles', type=int, default=1, help='(+v,-v) 循环次数')
    ap.add_argument('--save_csv', type=str, default='', help='可选：保存到 CSV 的路径')
    args = ap.parse_args()

    host = args.host
    duration = max(0.5, float(args.duration))
    alpha = float(args.alpha)

    # 接收端
    rtde_r = RTDEReceiveInterface(host)

    # 启动控制线程（可选）
    ctrl_th = None
    if args.do_control:
        spd = [float(x) for x in args.speed.strip().split()]
        if len(spd) != 6:
            raise ValueError('speed 必须提供 6 个关节速度 (rad/s)')
        ctrl_th = threading.Thread(target=control_thread_fn,
                                   args=(host, spd, float(args.accel), float(args.segment_time), int(args.cycles)),
                                   daemon=True)
        ctrl_th.start()

    # 采样循环
    t0 = rtde_r.getTimestamp()
    v_prev = np.array(rtde_r.getActualQd(), dtype=float)
    t_prev = t0
    qdd_f_prev = None

    Ts, Vels, Accs = [], [], []  # 列表存储

    print('采集开始... 按 Ctrl+C 可提前结束')
    t_start = time.time()
    try:
        while time.time() - t_start < duration:
            v = np.array(rtde_r.getActualQd(), dtype=float)
            ts = rtde_r.getTimestamp()  # 控制器时间戳 (s)
            dt = max(1e-4, ts - t_prev)
            qdd = (v - v_prev) / dt
            qdd_f = ema_filter(qdd_f_prev, qdd, alpha)

            Ts.append(ts - t0)
            Vels.append(v.tolist())
            Accs.append(qdd_f.tolist())

            v_prev, t_prev, qdd_f_prev = v, ts, qdd_f
            # 尝试跟上 125 Hz，无主动 sleep（由 RTDE 提供节奏）
    except KeyboardInterrupt:
        print('用户中断，收尾...')

    # 可选：等待控制线程退出
    if ctrl_th is not None:
        ctrl_th.join(timeout=0.5)

    Ts = np.array(Ts)
    Vels = np.array(Vels)  # Nx6
    Accs = np.array(Accs)  # Nx6

    # 保存 CSV（可选）
    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['t'] + [f'qd_{i+1}' for i in range(6)] + [f'qdd_{i+1}' for i in range(6)]
            writer.writerow(header)
            for i in range(len(Ts)):
                writer.writerow([Ts[i]] + Vels[i].tolist() + Accs[i].tolist())
        print(f'CSV 已保存到: {args.save_csv}')

    # 画图：速度
    plt.figure('Joint Velocity (actual_qd)')
    for j in range(6):
        plt.plot(Ts, Vels[:, j], label=JOINTS[j])
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend(loc='best')
    plt.grid(True)

    # 画图：加速度（EMA 滤波后）
    plt.figure('Joint Acceleration (diff+EMA)')
    for j in range(6):
        plt.plot(Ts, Accs[:, j], label=JOINTS[j])
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (rad/s^2)')
    plt.legend(loc='best')
    plt.grid(True)

    plt.show()

    try:
        rtde_r.disconnect()
    except Exception:
        pass


if __name__ == '__main__':
    main()
