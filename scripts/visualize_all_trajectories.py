# =============================================================
# visualize_all_trajectories.py  —  全量轨迹：原始 vs 生成 对比图
# =============================================================
# 用法:
#   python scripts/visualize_all_trajectories.py                       # 默认保存图片
#   python scripts/visualize_all_trajectories.py --split val
#   python scripts/visualize_all_trajectories.py --model-name GAIL --model-ckpt gail_policy_conservative_001
#   python scripts/visualize_all_trajectories.py --no-save             # 不保存，直接显示
# =============================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    from Convert_expert_data import load_all_with_dataframes
    from myenv import LeftTurnEnv
except ModuleNotFoundError:
    from scripts.Convert_expert_data import load_all_with_dataframes
    from scripts.myenv import LeftTurnEnv


DATA_DIR = "./data/left_turn/"
SAVE_DIR = "./checkpoints/"
TRAIN_RATIO = 0.8
SPLIT_SEED = 42

# 两类轨迹各用一个固定颜色
EXPERT_COLOR = "#2c7bb6"
GENERATED_COLOR = "#d7191c"


def split_train_val(trajs, dfs, train_ratio=TRAIN_RATIO, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(trajs))
    split_idx = min(max(int(len(indices) * train_ratio), 1), len(indices) - 1)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return (
        [trajs[i] for i in train_idx], [dfs[i] for i in train_idx],
        [trajs[i] for i in val_idx], [dfs[i] for i in val_idx],
    )


def rollout_policy(policy, df, ep_idx, all_dfs):
    """在环境上 rollout，返回生成轨迹坐标。"""
    env = LeftTurnEnv(all_dfs, max_dev=1e9, goal_radius=-1.0)
    obs, _ = env.reset(options={"episode_idx": ep_idx})

    xs, ys = [float(obs[0])], [float(obs[1])]
    t_max = len(df) - 1
    for _ in range(t_max):
        action, _ = policy.predict(obs[None], deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action[0])
        xs.append(float(obs[0]))
        ys.append(float(obs[1]))
        if terminated or truncated:
            break

    env.close()
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def visualize_all_trajectories(all_dfs, policy, model_name: str, split_name: str):
    """将所有轨迹画在一张图：原始(Expert) 与 生成(Policy)。"""
    fig, ax = plt.subplots(figsize=(10, 9))

    ade_list: list[float] = []

    for ep_idx, (df, _goal) in enumerate(all_dfs):
        expert_x = df["HV_X"].to_numpy(dtype=np.float32)
        expert_y = df["HV_Y"].to_numpy(dtype=np.float32)

        gen_x, gen_y = rollout_policy(policy, df, ep_idx=ep_idx, all_dfs=all_dfs)

        # 原始轨迹（统一颜色）
        ax.plot(
            expert_x,
            expert_y,
            color=EXPERT_COLOR,
            lw=1.0,
            alpha=0.35,
        )

        # 生成轨迹（统一颜色）
        ax.plot(
            gen_x,
            gen_y,
            color=GENERATED_COLOR,
            lw=1.0,
            alpha=0.35,
            ls="--",
        )

        m = min(len(expert_x), len(gen_x))
        if m > 0:
            err = np.sqrt((expert_x[:m] - gen_x[:m]) ** 2 + (expert_y[:m] - gen_y[:m]) ** 2)
            ade_list.append(float(np.mean(err)))

    ax.set_xlabel("HV_X (m)", fontsize=12)
    ax.set_ylabel("HV_Y (m)", fontsize=12)
    ax.set_title(
        f"All trajectories ({split_name})  —  Expert vs {model_name}\n"
        f"episodes={len(all_dfs)}, mean ADE={np.mean(ade_list):.3f} m",
        fontsize=13,
    )
    ax.set_aspect("equal", "datalim")
    ax.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], color=EXPERT_COLOR, lw=2, label="Expert (original)"),
        Line2D([0], [0], color=GENERATED_COLOR, lw=2, ls="--", label=f"{model_name} (generated)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="best")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot all expert vs generated trajectories in one figure")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--model-name", default="GAIL", help="Legend name of generated policy")
    parser.add_argument("--model-ckpt", default="gail_policy_conservative", help="Checkpoint file under ./checkpoints")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--no-save", action="store_true", help="Do not save image; show figure window instead")
    args = parser.parse_args()

    print("Loading data ...")
    trajs, dfs = load_all_with_dataframes(args.data_dir)
    _, train_dfs, _, val_dfs = split_train_val(trajs, dfs)

    if args.split == "train":
        target_dfs = train_dfs
    elif args.split == "val":
        target_dfs = val_dfs
    else:
        target_dfs = dfs

    ckpt_path = Path(SAVE_DIR) / args.model_ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model not found: {ckpt_path}")

    print(f"Loading policy: {args.model_name} from {ckpt_path}")
    policy = ActorCriticPolicy.load(str(ckpt_path))

    fig = visualize_all_trajectories(
        all_dfs=target_dfs,
        policy=policy,
        model_name=args.model_name,
        split_name=args.split,
    )

    out_dir = Path("./evaluate")
    out_dir.mkdir(exist_ok=True)

    if not args.no_save:
        out_path = out_dir / f"all_traj_expert_vs_{args.model_name.lower()}_{args.split}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
