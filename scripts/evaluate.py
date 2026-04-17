# =============================================================
# evaluate.py  —  基于位置误差的 ADE/FDE 评估脚本
# =============================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    from Convert_expert_data import load_all_with_dataframes
    from myenv import LeftTurnEnv
except ModuleNotFoundError:
    from scripts.Convert_expert_data import load_all_with_dataframes
    from scripts.myenv import LeftTurnEnv


DEFAULT_DATA_DIR = "./data/left_turn/"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SPLIT_SEED = 42
DEFAULT_BC_PATH = "./checkpoints/bc_policy_hv_ax_ay"
DEFAULT_GAIL_PATH = "./checkpoints/gail_policy_conservative"


def split_train_val(
    trajs: list,
    dfs: list,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
):
    """按固定随机种子将 trajs/dfs 同步划分为 train/val。"""
    if len(trajs) != len(dfs):
        raise ValueError(f"trajs 与 dfs 数量不一致: {len(trajs)} vs {len(dfs)}")
    if len(trajs) < 2:
        raise ValueError("轨迹数量过少，至少需要 2 条才能划分 train/val")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(trajs))
    split_idx = int(len(indices) * train_ratio)
    split_idx = min(max(split_idx, 1), len(indices) - 1)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_trajs = [trajs[i] for i in train_idx]
    train_dfs = [dfs[i] for i in train_idx]
    val_trajs = [trajs[i] for i in val_idx]
    val_dfs = [dfs[i] for i in val_idx]

    print(
        f"[Split] train={len(train_trajs)} ({len(train_trajs)/len(trajs):.1%}), "
        f"val={len(val_trajs)} ({len(val_trajs)/len(trajs):.1%}), seed={seed}"
    )
    return train_trajs, train_dfs, val_trajs, val_dfs


def evaluate_position_metrics(
    policy: ActorCriticPolicy,
    val_dfs: list,
    deterministic: bool = True,
):
    """
    位置误差评估：
      - ADE: 所有预测时间步的位置欧氏误差平均
      - FDE: 每条轨迹最后一个时间步的位置欧氏误差，再对轨迹平均

    注：
      - 误差按 t=1..T-1 统计（reset 初始时刻 t=0 不计入）
      - 与专家位置真值比较的是 HV_X/HV_Y
    """
    if len(val_dfs) == 0:
        raise ValueError("val_dfs 为空，无法评估")

    # 评估时关闭提前终止，避免不同策略轨迹长度不一致
    env = LeftTurnEnv(
        val_dfs,
        max_dev=1e9,
        goal_radius=-1.0,
        playback=False,
    )

    all_step_err = []
    ep_ade = []
    ep_fde = []

    for ep_idx, (df, _) in enumerate(val_dfs):
        gt_xy = df[["HV_X", "HV_Y"]].to_numpy(dtype=np.float32)
        if len(gt_xy) < 2:
            continue

        obs, _ = env.reset(options={"episode_idx": ep_idx})
        ep_err = []

        # 预测并对齐到专家 t=1..T-1
        for t in range(1, len(gt_xy)):
            action, _ = policy.predict(obs[None], deterministic=deterministic)
            obs, _, terminated, truncated, _ = env.step(action[0])

            pred_xy = np.array([obs[0], obs[1]], dtype=np.float32)
            err = float(np.linalg.norm(pred_xy - gt_xy[t]))
            ep_err.append(err)
            all_step_err.append(err)

            if terminated or truncated:
                # 正常情况下到最后一步才 truncated；这里保守中断
                break

        if ep_err:
            ep_ade.append(float(np.mean(ep_err)))
            ep_fde.append(float(ep_err[-1]))

    env.close()

    if not ep_ade:
        raise RuntimeError("没有有效轨迹可用于 ADE/FDE 计算")

    return {
        "n_episodes": int(len(ep_ade)),
        "n_steps": int(len(all_step_err)),
        "ADE": float(np.mean(all_step_err)),          # micro ADE
        "ADE_ep_mean": float(np.mean(ep_ade)),        # macro ADE
        "FDE": float(np.mean(ep_fde)),
        "ADE_p95": float(np.quantile(all_step_err, 0.95)),
        "FDE_p95": float(np.quantile(ep_fde, 0.95)),
    }


def print_metrics(name: str, metrics: dict):
    print(f"\n[{name}] 位置误差评估")
    print(f"  评估轨迹数: {metrics['n_episodes']}")
    print(f"  总步数: {metrics['n_steps']}")
    print(f"  ADE (step mean): {metrics['ADE']:.3f}")
    print(f"  ADE (ep mean):   {metrics['ADE_ep_mean']:.3f}")
    print(f"  FDE:             {metrics['FDE']:.3f}")
    print(f"  ADE p95:         {metrics['ADE_p95']:.3f}")
    print(f"  FDE p95:         {metrics['FDE_p95']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC/GAIL with position ADE/FDE")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--bc-path", type=str, default=DEFAULT_BC_PATH)
    parser.add_argument("--gail-path", type=str, default=DEFAULT_GAIL_PATH)
    args = parser.parse_args()

    print("========== 加载数据 ==========")
    trajs, dfs = load_all_with_dataframes(args.data_dir)
    _, _, _, val_dfs = split_train_val(
        trajs,
        dfs,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
    )

    model_items = [
        ("BC", Path(args.bc_path)),
        ("GAIL", Path(args.gail_path)),
    ]

    print("\n========== 位置 ADE/FDE ==========")
    for name, ckpt in model_items:
        if not ckpt.exists():
            print(f"\n[{name}] 跳过：未找到模型 {ckpt}")
            continue

        policy = ActorCriticPolicy.load(str(ckpt))
        metrics = evaluate_position_metrics(policy, val_dfs)
        print_metrics(name, metrics)


if __name__ == "__main__":
    main()
