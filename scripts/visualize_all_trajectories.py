# =============================================================
# visualize_all_trajectories.py  —  全量轨迹：Expert vs 单模型 对比图（GAIL/AIRL/BC 各一张）
# =============================================================
# 用法:
#   python scripts/visualize_all_trajectories.py                       # 默认保存图片
#   python scripts/visualize_all_trajectories.py --split val
#   python scripts/visualize_all_trajectories.py --gail-ckpt gail_policy_conservative_001 --airl-ckpt airl_policy_001 --bc-ckpt bc_policy_hv_ax_ay_001
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

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

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

# 轨迹颜色
EXPERT_COLOR = "#2c7bb6"
GAIL_COLOR = "#d7191c"
AIRL_COLOR = "#fdae61"
BC_COLOR = "#1a9641"

Y_MAX = 35.0
TRAJ_POINT_SIZE = 5
TRAJ_POINT_EDGE_WIDTH = 0.5


def split_train_val(trajs, dfs, train_ratio=TRAIN_RATIO, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(trajs))
    split_idx = min(max(int(len(indices) * train_ratio), 1), len(indices) - 1)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return (
        [trajs[i] for i in train_idx], [dfs[i] for i in train_idx],
        [trajs[i] for i in val_idx], [dfs[i] for i in val_idx],
    )


def truncate_xy_by_ymax(xs: np.ndarray, ys: np.ndarray, y_max: float = Y_MAX):
    """截断轨迹，移除 y > y_max 的部分。"""
    mask = ys <= y_max
    if not np.any(mask):
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    return xs[mask].astype(np.float32), ys[mask].astype(np.float32)


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
    xs_arr = np.asarray(xs, dtype=np.float32)
    ys_arr = np.asarray(ys, dtype=np.float32)
    return truncate_xy_by_ymax(xs_arr, ys_arr, y_max=Y_MAX)


def visualize_model_vs_expert(all_dfs, policy: ActorCriticPolicy, model_name: str, split_name: str):
    """将所有轨迹画在一张图：Expert 与单个模型生成轨迹。"""
    fig, ax = plt.subplots(figsize=(10, 9))

    ade_list: list[float] = []
    model_color = {
        "GAIL": GAIL_COLOR,
        "AIRL": AIRL_COLOR,
        "BC": BC_COLOR,
    }.get(model_name, "#000000")

    # 收集所有轨迹，先统一画专家，再统一画生成，确保生成轨迹图层在上
    expert_trajs: list[tuple[np.ndarray, np.ndarray]] = []
    gen_trajs: list[tuple[np.ndarray, np.ndarray]] = []

    for ep_idx, (df, _goal) in enumerate(all_dfs):
        expert_x_raw = df["HV_X"].to_numpy(dtype=np.float32)
        expert_y_raw = df["HV_Y"].to_numpy(dtype=np.float32)
        expert_x, expert_y = truncate_xy_by_ymax(expert_x_raw, expert_y_raw, y_max=Y_MAX)
        if len(expert_x) == 0:
            continue
        expert_trajs.append((expert_x, expert_y))

        gen_x, gen_y = rollout_policy(policy, df, ep_idx=ep_idx, all_dfs=all_dfs)
        gen_trajs.append((gen_x, gen_y))

        m = min(len(expert_x), len(gen_x))
        if m > 0:
            err = np.sqrt((expert_x[:m] - gen_x[:m]) ** 2 + (expert_y[:m] - gen_y[:m]) ** 2)
            ade_list.append(float(np.mean(err)))

    # 第一遍：画所有专家轨迹（底层）
    for expert_x, expert_y in expert_trajs:
        ax.scatter(
            expert_x,
            expert_y,
            facecolors="none",
            edgecolors=EXPERT_COLOR,
            s=TRAJ_POINT_SIZE,
            alpha=0.6,
            marker="o",
            linewidths=TRAJ_POINT_EDGE_WIDTH,
        )

    # 第二遍：画所有生成轨迹（上层）
    for gen_x, gen_y in gen_trajs:
        if len(gen_x) == 0:
            continue
        ax.scatter(
            gen_x,
            gen_y,
            facecolors="none",
            edgecolors=model_color,
            s=TRAJ_POINT_SIZE,
            alpha=0.6,
            marker="o",
            linewidths=TRAJ_POINT_EDGE_WIDTH,
        )

    mean_ade = float(np.mean(ade_list)) if ade_list else float("nan")

    title_lines = [
        f"全部轨迹对比（{split_name}）— 真实轨迹 vs {model_name}",
        f"轨迹数={len(all_dfs)}，截断条件：y≤{Y_MAX:.0f} 米",
        f"ADE：{model_name}={mean_ade:.3f} 米",
    ]

    ax.set_xlabel("主车横向位置 X（米）", fontsize=20)
    ax.set_ylabel("主车纵向位置 Y（米）", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(-35, 25)
    ax.set_ylim(0, 45)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=EXPERT_COLOR, linestyle="None", markersize=5, label="真实轨迹"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor=model_color, linestyle="None", markersize=5, label="生成轨迹"),
    ]
    ax.legend(handles=legend_elements, fontsize=16, loc="best")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="绘制真实轨迹与 GAIL/AIRL/BC 生成轨迹对比图")
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--gail-ckpt", default="gail_policy_conservative", help="GAIL checkpoint file under ./checkpoints")
    parser.add_argument("--airl-ckpt", default="airl_policy_003", help="AIRL checkpoint file under ./checkpoints")
    parser.add_argument("--bc-ckpt", default="bc_policy_hv_ax_ay_001", help="BC checkpoint file under ./checkpoints")
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

    ckpt_map = {
        "GAIL": args.gail_ckpt,
        "AIRL": args.airl_ckpt,
        "BC": args.bc_ckpt,
    }
    policies: dict[str, ActorCriticPolicy] = {}
    for name, ckpt_name in ckpt_map.items():
        ckpt_path = Path(SAVE_DIR) / ckpt_name
        if not ckpt_path.exists() and ckpt_path.with_suffix(".zip").exists():
            ckpt_path = ckpt_path.with_suffix(".zip")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model not found: {ckpt_path}")
        print(f"Loading policy: {name} from {ckpt_path}")
        policies[name] = ActorCriticPolicy.load(str(ckpt_path))

    out_dir = Path("./evaluate")
    out_dir.mkdir(exist_ok=True)

    figs = []
    for model_name in ["GAIL", "AIRL", "BC"]:
        fig = visualize_model_vs_expert(
            all_dfs=target_dfs,
            policy=policies[model_name],
            model_name=model_name,
            split_name=args.split,
        )
        figs.append(fig)

        if not args.no_save:
            out_path = out_dir / f"all_traj_points_expert_vs_{model_name.lower()}_{args.split}.svg"
            fig.savefig(out_path, format="svg", bbox_inches="tight")
            print(f"Saved → {out_path}")

    if args.no_save:
        plt.show()


if __name__ == "__main__":
    main()
