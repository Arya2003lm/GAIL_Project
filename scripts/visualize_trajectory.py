# =============================================================
# visualize_trajectory.py  —  策略轨迹模仿效果可视化
# =============================================================
# 用法:
#   python scripts/visualize_trajectory.py                  # 可视化验证集第0条轨迹，BC vs GAIL
#   python scripts/visualize_trajectory.py --ep 3           # 验证集第3条
#   python scripts/visualize_trajectory.py --ep 0 --split train  # 训练集第0条
#   python scripts/visualize_trajectory.py --ep 0 5 8      # 同时画3条轨迹的对比图（subplots）
#   python scripts/visualize_trajectory.py --save           # 保存图片
# =============================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    from Convert_expert_data import load_all_with_dataframes
    from myenv import LeftTurnEnv
    from relevant_traj import make_reference_traj
except ModuleNotFoundError:
    from scripts.Convert_expert_data import load_all_with_dataframes
    from scripts.myenv import LeftTurnEnv
    from scripts.relevant_traj import make_reference_traj

# ---------- 配置 ----------
DATA_DIR     = "./data/left_turn/"
SAVE_DIR     = "./checkpoints/"
TRAIN_RATIO  = 0.8
SPLIT_SEED   = 42

MODELS = {
    "BC":   "bc_policy_hv_ax_ay",
    "GAIL": "gail_policy_conservative_001",
    "AIRL": "airl_policy_002",
}
MODEL_COLORS = {
    "Expert":    "#2c7bb6",
    "BC":        "#d7191c",
    "GAIL":      "#1a9641",
    "AIRL":      "#f28e2b",
    "Reference": "#984ea3",
}


def resolve_checkpoint_path(save_dir: str | Path, model_name: str) -> Path | None:
    """解析 checkpoint 路径：优先精确命中，否则自动匹配最新编号版本。"""
    save_root = Path(save_dir)
    direct = save_root / model_name
    if direct.exists():
        return direct

    direct_zip = save_root / f"{model_name}.zip"
    if direct_zip.exists():
        return direct_zip

    pattern = re.compile(rf"^{re.escape(model_name)}_(\d+)$")
    candidates: list[tuple[int, Path]] = []
    for p in save_root.glob(f"{model_name}_*"):
        if not p.is_file():
            continue
        candidate_name = p.stem if p.suffix == ".zip" else p.name
        m = pattern.match(candidate_name)
        if m:
            candidates.append((int(m.group(1)), p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# ---------- 数据切分 ----------
def split_train_val(trajs, dfs, train_ratio=TRAIN_RATIO, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(trajs))
    split_idx = min(max(int(len(indices) * train_ratio), 1), len(indices) - 1)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return (
        [trajs[i] for i in train_idx], [dfs[i]   for i in train_idx],
        [trajs[i] for i in val_idx],   [dfs[i]   for i in val_idx],
    )


# ---------- 单策略 rollout ----------
def rollout_policy(policy, df, goal, ep_idx, all_dfs):
    """
    在 LeftTurnEnv 上 rollout policy，返回预测轨迹坐标列表。
    关闭提前终止，确保按专家轨迹长度完整对齐。
    """
    env = LeftTurnEnv(all_dfs, max_dev=1e9, goal_radius=-1.0)
    obs, _ = env.reset(options={"episode_idx": ep_idx})

    xs, ys = [obs[0]], [obs[1]]
    T = len(df) - 1
    for _ in range(T):
        action, _ = policy.predict(obs[None], deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action[0])
        xs.append(obs[0])
        ys.append(obs[1])
        if terminated or truncated:
            break
    env.close()
    return np.array(xs), np.array(ys)


# ---------- 单条轨迹可视化 ----------
def visualize_single(
    ep_idx: int,
    all_dfs: list,
    policies: dict,
    title: str = "",
    ax: plt.Axes = None,
    show_ego: bool = True,
    show_actions: bool = True,
    show_reference: bool = True,
):
    """
    在给定的 ax 上画单条轨迹的专家 vs 各策略对比。
    ax=None 时新建图窗。
    show_actions=True 时另开一行 subplot 展示 ax/ay 随时间变化。
    """
    df, goal = all_dfs[ep_idx]
    expert_x = df["HV_X"].to_numpy(dtype=np.float32)
    expert_y = df["HV_Y"].to_numpy(dtype=np.float32)
    ego_x    = df["ego_X"].to_numpy(dtype=np.float32)
    ego_y    = df["ego_Y"].to_numpy(dtype=np.float32)
    expert_ax = df["HV_ax"].to_numpy(dtype=np.float32)
    expert_ay = df["HV_ay"].to_numpy(dtype=np.float32)
    T = len(df)
    timesteps = np.arange(T) * 0.1  # dt=0.1s

    # ---------- 创建 axes ----------
    if show_actions:
        fig, axes = plt.subplots(
            2, 1,
            figsize=(9, 10),
            gridspec_kw={"height_ratios": [2, 1]},
        )
        ax_traj, ax_act = axes
    else:
        if ax is None:
            fig, ax_traj = plt.subplots(figsize=(8, 7))
        else:
            ax_traj = ax
            fig = ax_traj.get_figure()
        ax_act = None

    # ---------- 轨迹图 ----------
    ax_traj.plot(expert_x, expert_y, color=MODEL_COLORS["Expert"],
                 lw=2.0, label="Expert (ground truth)", zorder=3)

    # 参考轨迹
    if show_reference:
        start = np.array([expert_x[0], expert_y[0]])
        goal_arr = np.asarray(goal, dtype=np.float64)
        ref_xs, ref_ys = make_reference_traj(start, goal_arr)
        ax_traj.plot(ref_xs, ref_ys,
                     color=MODEL_COLORS["Reference"], lw=1.8, ls=(0, (5, 2)),
                     label="Reference", zorder=2, alpha=0.85)

    # rollout 各策略
    pred_trajs = {}
    for name, policy in policies.items():
        px, py = rollout_policy(policy, df, goal, ep_idx, all_dfs)
        pred_trajs[name] = (px, py)
        ax_traj.plot(px, py, color=MODEL_COLORS.get(name, "gray"),
                     lw=1.8, ls="--", label=name, zorder=4)

    # ego（AV）轨迹
    if show_ego:
        ax_traj.plot(ego_x, ego_y, color="orange", lw=1.2, ls=":", alpha=0.7, label="AV (ego)")

    # 起点 / 终点 / 目标点
    ax_traj.scatter([expert_x[0]], [expert_y[0]],  marker="o", s=80, color="black", zorder=5, label="HV start")
    ax_traj.scatter([expert_x[-1]], [expert_y[-1]], marker="s", s=80, color=MODEL_COLORS["Expert"], zorder=5, label="Expert end")
    ax_traj.scatter([goal[0]], [goal[1]], marker="*", s=200, color="gold", edgecolors="k", zorder=6, label="Goal")

    ax_traj.set_xlabel("X (m)", fontsize=11)
    ax_traj.set_ylabel("Y (m)", fontsize=11)
    ax_traj.set_title(title or f"Trajectory comparison  (episode {ep_idx})", fontsize=12)
    ax_traj.legend(fontsize=9, loc="best")
    ax_traj.set_aspect("equal", "datalim")
    ax_traj.grid(True, alpha=0.3)

    # ---------- 动作对比图 ----------
    if ax_act is not None:
        t = timesteps
        ax_act.plot(t, expert_ax, color=MODEL_COLORS["Expert"], lw=1.8, label="Expert HV_ax")
        ax_act.plot(t, expert_ay, color=MODEL_COLORS["Expert"], lw=1.8, ls=":", label="Expert HV_ay")

        for name, policy in policies.items():
            # 用专家观测驱动策略，得到每步预测动作
            pred_acts = []
            env = LeftTurnEnv(all_dfs, max_dev=1e9, goal_radius=-1.0)
            obs, _ = env.reset(options={"episode_idx": ep_idx})
            for step_i in range(T - 1):
                action, _ = policy.predict(obs[None], deterministic=True)
                pred_acts.append(action[0])
                obs, _, terminated, truncated, _ = env.step(action[0])
                if terminated or truncated:
                    break
            env.close()

            if pred_acts:
                pa = np.array(pred_acts)
                t_pred = timesteps[:len(pa)]
                ax_act.plot(t_pred, pa[:, 0], color=MODEL_COLORS.get(name, "gray"),
                            lw=1.5, ls="--", label=f"{name} HV_ax")
                ax_act.plot(t_pred, pa[:, 1], color=MODEL_COLORS.get(name, "gray"),
                            lw=1.5, ls="-.", label=f"{name} HV_ay")

        ax_act.axhline(0, color="gray", lw=0.6, ls="--")
        ax_act.set_xlabel("Time (s)", fontsize=10)
        ax_act.set_ylabel("Acceleration (m/s²)", fontsize=10)
        ax_act.set_title("Action comparison: HV_ax / HV_ay over time", fontsize=11)
        ax_act.legend(fontsize=8, ncol=2, loc="upper right")
        ax_act.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------- 多条轨迹汇总图 ----------
def visualize_multi(ep_indices: list[int], all_dfs: list, policies: dict, ncols: int = 3):
    n = len(ep_indices)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = np.array(axes).flatten()

    for i, ep_idx in enumerate(ep_indices):
        visualize_single(ep_idx, all_dfs, policies,
                         title=f"ep={ep_idx}", ax=axes[i], show_actions=False, show_reference=True)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # 添加全局图例
    legend_patches = [mpatches.Patch(color=c, label=n) for n, c in MODEL_COLORS.items()
                      if n in ("Expert", "Reference", *policies.keys())]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=len(legend_patches), fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0))
    fig.suptitle(f"Multi-episode trajectory comparison  ({len(ep_indices)} episodes)", fontsize=13)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    return fig


# ---------- 主入口 ----------
def main():
    parser = argparse.ArgumentParser(description="Visualize policy trajectory imitation")
    parser.add_argument("--ep", type=int, nargs="+", default=[0],
                        help="Episode indices to visualize (space-separated)")
    parser.add_argument("--split", choices=["train", "val"], default="val",
                        help="Which split to use (train / val)")
    parser.add_argument("--no-ego", action="store_true", help="Hide AV (ego) trajectory")
    parser.add_argument("--no-actions", action="store_true", help="Hide action subplot")
    parser.add_argument("--no-reference", action="store_true", help="Hide reference trajectory")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to evaluate/ directory instead of showing")
    parser.add_argument("--all-eps", action="store_true",
                        help="Export comparison figures for all episodes in selected split")
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    # ------ 加载数据 ------
    print("Loading data ...")
    trajs, dfs = load_all_with_dataframes(args.data_dir)
    _, train_dfs, _, val_dfs = split_train_val(trajs, dfs)
    target_dfs = train_dfs if args.split == "train" else val_dfs
    n_ep = len(target_dfs)
    print(f"Using '{args.split}' split: {n_ep} episodes")

    # ------ 加载策略 ------
    policies = {}
    for name, ckpt in MODELS.items():
        path = resolve_checkpoint_path(SAVE_DIR, ckpt)
        if path is not None:
            policies[name] = ActorCriticPolicy.load(str(path))
            print(f"  Loaded {name} from {path}")
        else:
            print(f"  Skipping {name}: not found for key '{ckpt}' in {SAVE_DIR}")

    if not policies:
        print("No models found, exiting.")
        return

    # ------ 校验 episode 索引 ------
    ep_indices = [ep % n_ep for ep in args.ep]

    save_dir = Path("./evaluate")
    save_dir.mkdir(exist_ok=True)

    # ------ 导出全部 episode（逐张图） ------
    if args.all_eps:
        out_dir = save_dir / "traj_compare"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exporting all episodes to: {out_dir}")

        for ep in range(n_ep):
            fig = visualize_single(
                ep, target_dfs, policies,
                title=f"[{args.split}] Episode {ep}  —  Expert vs " + " vs ".join(policies.keys()),
                show_ego=not args.no_ego,
                show_actions=not args.no_actions,
                show_reference=not args.no_reference,
            )
            out = out_dir / f"traj_vis_{args.split}_ep{ep:03d}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"Saved all episode figures → {out_dir}")
        return

    if len(ep_indices) == 1:
        ep = ep_indices[0]
        fig = visualize_single(
            ep, target_dfs, policies,
            title=f"[{args.split}] Episode {ep}  —  Expert vs " + " vs ".join(policies.keys()),
            show_ego=not args.no_ego,
            show_actions=not args.no_actions,
            show_reference=not args.no_reference,
        )
        if args.save:
            out = save_dir / f"traj_vis_{args.split}_ep{ep}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
        else:
            plt.show()
    else:
        fig = visualize_multi(ep_indices, target_dfs, policies)
        if args.save:
            eps_str = "_".join(str(e) for e in ep_indices)
            out = save_dir / f"traj_vis_{args.split}_ep{eps_str}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
