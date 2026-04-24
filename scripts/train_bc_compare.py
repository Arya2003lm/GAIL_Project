"""
train_bc_compare.py  —  独立 BC 训练脚本（用于对比算法）

特点：
1) 不依赖 AIRL/GAIL 训练流程，单独训练 BC。
2) 默认训练轮次更高（BC_EPOCHS=300）。
3) 使用与 train_airl 相同的数据与评估口径（动作误差 + 位置 ADE/FDE）。
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
from pathlib import Path

import numpy as np
import torch
from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.types import Trajectory
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    from Convert_expert_data import (
        load_all_with_dataframes as ced_load_all_with_dataframes,
        validate as ced_validate_data,
        observation_space as ced_observation_space,
        action_space as ced_action_space,
    )
    from myenv import LeftTurnEnv
except ModuleNotFoundError:
    from scripts.Convert_expert_data import (
        load_all_with_dataframes as ced_load_all_with_dataframes,
        validate as ced_validate_data,
        observation_space as ced_observation_space,
        action_space as ced_action_space,
    )
    from scripts.myenv import LeftTurnEnv


# =============================================================
# 0. 配置
# =============================================================

DATA_DIR = "./data/left_turn/"
SAVE_DIR = "./checkpoints/"
BC_POLICY_NAME = "bc_policy_compare"

TRAIN_RATIO = 0.8
SPLIT_SEED = 28

# 对比实验建议使用更高轮次
BC_EPOCHS = 100
BC_LOG_INTERVAL = 20

POLICY_NET_ARCH = dict(pi=[256, 256], vf=[256, 256])
POLICY_ACTIVATION_FN = torch.nn.ReLU

ENV_PROGRESS_REWARD_SCALE = 1.0
ENV_GOAL_REACHED_BONUS = 5.0
ENV_REF_PATH_PENALTY_THRESHOLD = 5.0
ENV_REF_PATH_PENALTY_SCALE = 1.0

Path(SAVE_DIR).mkdir(exist_ok=True)

observation_space = ced_observation_space
action_space = ced_action_space


# =============================================================
# 1. 工具函数
# =============================================================


def _make_next_versioned_name(base_name: str, save_dir: str = SAVE_DIR) -> str:
    save_root = Path(save_dir)
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    max_idx = 0

    for file in save_root.glob(f"{base_name}_*.zip"):
        m = pattern.match(file.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    return f"{base_name}_{max_idx + 1:03d}"


def split_train_val(
    trajs: list[Trajectory],
    dfs: list,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SPLIT_SEED,
):
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


def make_policy():
    return ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=POLICY_NET_ARCH,
        activation_fn=POLICY_ACTIVATION_FN,
    )


# =============================================================
# 2. BC训练与评估
# =============================================================


def train_bc_on(trajs: list[Trajectory], save_name: str, n_epochs: int = BC_EPOCHS):
    print(f"\n========== BC 训练 ({save_name}) ==========")
    transitions = rollout.flatten_trajectories(trajs)
    print(f"[BC] transitions={len(transitions)}, trajectories={len(trajs)}, epochs={n_epochs}")

    trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=transitions,
        policy=make_policy(),
        rng=np.random.default_rng(42),
        batch_size=64,
        ent_weight=1e-3,
        l2_weight=1e-4,
    )

    done = 0
    while done < n_epochs:
        step_epochs = min(BC_LOG_INTERVAL, n_epochs - done)
        trainer.train(n_epochs=step_epochs)
        done += step_epochs
        print(f"  epoch {done}/{n_epochs}")

    versioned_name = _make_next_versioned_name(save_name)
    save_path = f"{SAVE_DIR}/{versioned_name}"
    trainer.policy.save(save_path)
    trainer.saved_path = save_path
    print(f"[BC] 策略已保存至 {save_path}")
    return trainer


def evaluate_action_error(trainer, trajs: list[Trajectory], n_episodes: int = 50):
    print("\n========== BC 动作误差评估 ==========")
    policy = trainer.policy

    eval_trajs = trajs[:n_episodes]
    if len(eval_trajs) == 0:
        print("  [评估] 没有可用轨迹")
        return

    abs_err_all = []
    sq_err_all = []
    l2_err_all = []

    for traj in eval_trajs:
        for t in range(len(traj.acts)):
            obs = traj.obs[t]
            action, _ = policy.predict(obs[None], deterministic=True)
            pred = np.asarray(action[0], dtype=np.float32)
            gt = np.asarray(traj.acts[t], dtype=np.float32)
            err = pred - gt
            abs_err_all.append(np.abs(err))
            sq_err_all.append(err ** 2)
            l2_err_all.append(float(np.sqrt(np.sum(err ** 2))))

    abs_err_all = np.asarray(abs_err_all, dtype=np.float32)
    sq_err_all = np.asarray(sq_err_all, dtype=np.float32)
    l2_err_all = np.asarray(l2_err_all, dtype=np.float32)

    mae = np.mean(abs_err_all, axis=0)
    rmse = np.sqrt(np.mean(sq_err_all, axis=0))
    p95 = np.quantile(abs_err_all, 0.95, axis=0)

    print(f"  评估轨迹数: {len(eval_trajs)}")
    print(f"  总步数: {len(l2_err_all)}")
    print(f"  单步L2误差: mean={np.mean(l2_err_all):.3f}, p95={np.quantile(l2_err_all, 0.95):.3f}")
    print(f"  HV_ax: MAE={mae[0]:.3f}, RMSE={rmse[0]:.3f}, P95|err|={p95[0]:.3f}")
    print(f"  HV_ay: MAE={mae[1]:.3f}, RMSE={rmse[1]:.3f}, P95|err|={p95[1]:.3f}")


def evaluate_position_ade_fde(policy: ActorCriticPolicy, eval_dfs: list, title: str = "BC 位置 ADE/FDE"):
    print(f"\n========== {title} ==========")
    if not eval_dfs:
        print("  [评估] 没有可用 DataFrame")
        return

    env = LeftTurnEnv(
        eval_dfs,
        max_dev=1e9,
        goal_radius=-1.0,
        progress_reward_scale=ENV_PROGRESS_REWARD_SCALE,
        goal_reached_bonus=ENV_GOAL_REACHED_BONUS,
        ref_path_penalty_threshold=ENV_REF_PATH_PENALTY_THRESHOLD,
        ref_path_penalty_scale=ENV_REF_PATH_PENALTY_SCALE,
    )

    all_step_err = []
    ep_ade = []
    ep_fde = []

    for ep_idx, (df, _) in enumerate(eval_dfs):
        gt_xy = df[["HV_X", "HV_Y"]].to_numpy(dtype=np.float32)
        if len(gt_xy) < 2:
            continue

        obs, _ = env.reset(options={"episode_idx": ep_idx})
        ep_err = []

        for t in range(1, len(gt_xy)):
            action, _ = policy.predict(obs[None], deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action[0])
            pred_xy = np.array([obs[0], obs[1]], dtype=np.float32)
            err = float(np.linalg.norm(pred_xy - gt_xy[t]))
            ep_err.append(err)
            all_step_err.append(err)
            if terminated or truncated:
                break

        if ep_err:
            ep_ade.append(float(np.mean(ep_err)))
            ep_fde.append(float(ep_err[-1]))

    env.close()

    if not ep_ade:
        print("  [评估] 没有有效轨迹")
        return

    print(f"  评估轨迹数: {len(ep_ade)}")
    print(f"  总步数: {len(all_step_err)}")
    print(f"  ADE (step mean): {np.mean(all_step_err):.3f}")
    print(f"  ADE (ep mean):   {np.mean(ep_ade):.3f}")
    print(f"  FDE:             {np.mean(ep_fde):.3f}")
    print(f"  ADE p95:         {np.quantile(all_step_err, 0.95):.3f}")
    print(f"  FDE p95:         {np.quantile(ep_fde, 0.95):.3f}")


if __name__ == "__main__":
    trajs, dfs = ced_load_all_with_dataframes(DATA_DIR)
    train_trajs, train_dfs, val_trajs, val_dfs = split_train_val(trajs, dfs)

    ced_validate_data(train_trajs)

    # 对比算法：默认使用 train split 训练；若想全量训练可替换成 trajs
    bc_trainer = train_bc_on(train_trajs, save_name=BC_POLICY_NAME, n_epochs=BC_EPOCHS)

    print("\n[BC] 在验证集上评估：")
    evaluate_action_error(bc_trainer, val_trajs)
    evaluate_position_ade_fde(bc_trainer.policy, val_dfs)
