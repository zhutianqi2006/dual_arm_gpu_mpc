import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_concat_npz(folder: str):
    paths = sorted(glob.glob(os.path.join(folder, 'mppi2_sg_000050.npz')))
    if not paths:
        raise FileNotFoundError(f'No npz files found in {folder}')
    pre_list, post_list, t_list = [], [], []
    r1, r2, dt = None, None, None
    for p in paths:
        data = np.load(p)
        pre = data['pre']  # (T, J)
        post = data['post']
        t = data['t']
        pre_list.append(pre)
        post_list.append(post)
        t_list.append(t + (t_list[-1][-1] + (data['dt'] if dt is None else dt)) if t_list else t)
        if r1 is None:
            r1 = int(data['robot1_q_num'])
            r2 = int(data['robot2_q_num'])
            dt = float(data['dt'])
    pre_all = np.vstack(pre_list)
    post_all = np.vstack(post_list)
    t_all = np.concatenate(t_list)
    return pre_all, post_all, t_all, r1, r2, dt


def plot_per_joint(pre: np.ndarray, post: np.ndarray, t: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    J = pre.shape[1]
    for j in range(J):
        plt.figure()
        plt.plot(t, pre[:, j], label='pre-SG')
        plt.plot(t, post[:, j], label='post-SG')
        plt.xlabel('time [s]')
        plt.ylabel('joint vel')
        plt.title(f'Joint {j+1}')
        plt.legend()
        plt.grid(True)
        fig_path = os.path.join(out_dir, f'joint_{j+1:02d}.png')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./figs', help='Output folder for PNGs')
    parser.add_argument('--show', action='store_true', help='Also show figures interactively')
    args = parser.parse_args()

    pre, post, t, r1, r2, dt = load_concat_npz("./logs")
    plot_per_joint(pre, post, t, args.out)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
