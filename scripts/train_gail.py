# =============================================================
# train.py  —  无保护左转模仿学习完整训练脚本
# =============================================================
# 依赖：pip install imitation stable-baselines3 gymnasium
# =============================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 OpenMP 重复加载冲突

import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from types import SimpleNamespace
from imitation.data.types import Trajectory
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
try:
    from Convert_expert_data import (
        load_all_trajectories as ced_load_all_trajectories,
        load_all_dataframes as ced_load_all_dataframes,
        validate as ced_validate_data,
        observation_space as ced_observation_space,
        action_space as ced_action_space,
    )
    from myenv import LeftTurnEnv
except ModuleNotFoundError:
    from scripts.Convert_expert_data import (
        load_all_trajectories as ced_load_all_trajectories,
        load_all_dataframes as ced_load_all_dataframes,
        validate as ced_validate_data,
        observation_space as ced_observation_space,
        action_space as ced_action_space,
    )
    from scripts.myenv import LeftTurnEnv
try:
    from Convert_expert_data import load_all_with_dataframes as ced_load_all_with_dataframes
except ModuleNotFoundError:
    from scripts.Convert_expert_data import load_all_with_dataframes as ced_load_all_with_dataframes
# =============================================================
# 0. 全局配置
# =============================================================

N_VEHICLES  = 1                        # 最多观测几辆他车
OBS_DIM     = ced_observation_space.shape[0]
ACT_DIM     = ced_action_space.shape[0]
DATA_DIR    = "./data/left_turn/"      # .asc 数据目录
SAVE_DIR    = "./checkpoints/"
BC_POLICY_NAME = "bc_policy_hv_ax_ay"
TRAIN_RATIO = 0.8
SPLIT_SEED = 28
GAIL_SMOKE_TIMESTEPS = 80_000
GAIL_FULL_TIMESTEPS = 500_000
RUN_ALIGNMENT_CHECK = False
RUN_SINGLE_TRAJ_DEBUG = False
RUN_BC_RETRAIN = False       # True: 重新训练 BC；False: 读取已有 BC_POLICY_NAME
BC_TRAIN_ON_ALL_DATA = True  # 按 train_origin 的方式：BC 直接用全量 trajs 训练
RUN_GAIL_SMOKE_TEST = False
RUN_GAIL_FULL_TRAIN = True
ENV_PROGRESS_REWARD_SCALE = 1.0
ENV_GOAL_REACHED_BONUS = 5.0
ENV_REF_PATH_PENALTY_THRESHOLD = 5.0
ENV_REF_PATH_PENALTY_SCALE = 1.0
ENV_LOG_INTERVAL_STEPS = 2000
GAIL_ENV_REWARD_WEIGHT = 0.52  # 训练时总奖励 = GAIL奖励 + weight * 环境奖励
Path(SAVE_DIR).mkdir(exist_ok=True)

observation_space = ced_observation_space
action_space = ced_action_space
POLICY_NET_ARCH = dict(pi=[256, 256], vf=[256, 256])
POLICY_ACTIVATION_FN = torch.nn.ReLU


def _make_next_versioned_name(base_name: str, save_dir: str = SAVE_DIR) -> str:
    """返回下一个可用的序号名称，如 base_name_001、base_name_002。"""
    save_root = Path(save_dir)
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    max_idx = 0

    for file in save_root.glob(f"{base_name}_*.zip"):
        m = pattern.match(file.stem)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    return f"{base_name}_{max_idx + 1:03d}"


def _resolve_latest_checkpoint_name(base_name: str, save_dir: str = SAVE_DIR) -> str:
    """优先返回基础名；若不存在则返回最新序号版本名。"""
    save_root = Path(save_dir)
    base_path = save_root / base_name
    if base_path.exists() or base_path.with_suffix(".zip").exists():
        return base_name

    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    candidates: list[tuple[int, str]] = []
    for file in save_root.glob(f"{base_name}_*.zip"):
        m = pattern.match(file.stem)
        if m:
            candidates.append((int(m.group(1)), file.stem))

    if not candidates:
        raise FileNotFoundError(
            f"未找到模型: {base_path} 或 {save_root / (base_name + '_###.zip')}"
        )

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# _dfs 存储所有轨迹 DataFrame，make_custom_env 通过闭包引用
_dfs: list = []


def make_custom_env():
    """供 DummyVecEnv 调用的工厂函数，返回自定义 LeftTurnEnv 实例。"""
    if not _dfs:
        raise RuntimeError("请先调用 init_custom_env_dfs(dfs) 初始化数据")
    return LeftTurnEnv(
        _dfs,
        progress_reward_scale=ENV_PROGRESS_REWARD_SCALE,
        goal_reached_bonus=ENV_GOAL_REACHED_BONUS,
        ref_path_penalty_threshold=ENV_REF_PATH_PENALTY_THRESHOLD,
        ref_path_penalty_scale=ENV_REF_PATH_PENALTY_SCALE,
    )


def init_custom_env_dfs(dfs: list) -> None:
    """在 GAIL 训练前调用，将 DataFrames 注入全局变量。"""
    global _dfs
    _dfs = dfs


def load_all_trajectories() -> list[Trajectory]:
    return ced_load_all_trajectories(DATA_DIR)


def load_all_dataframes():
    """加载带有派生列的 DataFrame 列表，供 LeftTurnEnv 使用。"""
    return ced_load_all_dataframes(DATA_DIR)


def split_train_val(
    trajs: list[Trajectory],
    dfs: list,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SPLIT_SEED,
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


# =============================================================
# 3. 数据验证（训练前跑一次）
# =============================================================

def validate_data(trajs: list[Trajectory]):
    ced_validate_data(trajs)


def print_observation_alignment_check(
    trajs: list[Trajectory],
    dfs: list,
    n_steps: int = 3000,
):
    """按轨迹回放专家动作，比较专家层观测与环境产出观测的分布。
    这样才能验证 _build_obs() 与 build_obs() 是否一致，
    而不是用随机动作驱动 HV 偏离轨迹。
    """
    print("\n========== 观测对齐检查 ==========")
    if not trajs:
        print("\n[对齐检查] 无轨迹数据，跳过")
        return

    obs_names = [
        "HV_X", "HV_Y", "HV_vx", "HV_vy",
        "dx", "dy", "yaw_prev", "d_des", "vx_prev", "vy_prev",
        "eHMI_state", "eHMI_duration",
        "Δpx", "Δpy", "Δvx", "Δvy",
    ]

    env = LeftTurnEnv(dfs, playback=True)  # playback=True: 直接从 DataFrame 读取 HV 状态

    expert_obs_list: list[np.ndarray] = []
    env_obs_list:    list[np.ndarray] = []
    collected = 0

    for ep_idx, traj in enumerate(trajs):
        if collected >= n_steps:
            break
        T = len(traj.acts)

        # 将环境定位到和 traj 匹配的同一条 DataFrame
        obs_env, _ = env.reset(options={"episode_idx": ep_idx})

        for t in range(T):
            if collected >= n_steps:
                break
            expert_obs_list.append(traj.obs[t])          # 专家观测
            env_obs_list.append(obs_env.astype(np.float32))  # 环境观测

            # 用专家动作驱动环境
            obs_env, _, terminated, truncated, _ = env.step(traj.acts[t])
            collected += 1

            if terminated or truncated:
                break

        print(f"\r[对齐检查] 回放轨迹 {ep_idx + 1}/{len(trajs)}，已收集 {collected}/{n_steps} 步", end="")

    env.close()
    print()  # 换行

    expert_obs = np.array(expert_obs_list, dtype=np.float32)
    env_obs    = np.array(env_obs_list,    dtype=np.float32)

    exp_mean, exp_std = expert_obs.mean(axis=0), expert_obs.std(axis=0)
    env_mean, env_std = env_obs.mean(axis=0),    env_obs.std(axis=0)
    mean_gap_sigma = np.abs(env_mean - exp_mean) / np.maximum(exp_std, 1e-6)

    exp_oor = ((expert_obs < observation_space.low) | (expert_obs > observation_space.high)).mean(axis=0)
    env_oor = ((env_obs   < observation_space.low) | (env_obs   > observation_space.high)).mean(axis=0)

    report = pd.DataFrame({
        "dim":            obs_names,
        "expert_mean":    exp_mean,
        "env_mean":       env_mean,
        "expert_std":     exp_std,
        "env_std":        env_std,
        "mean_gap_sigma": mean_gap_sigma,
        "expert_oor%":    exp_oor * 100,
        "env_oor%":       env_oor * 100,
    })

    print("\n========== 观测对齐检查（expert vs LeftTurnEnv 回放专家动作） ==========")
    print(f"  expert样本数: {len(expert_obs)}, env样本数: {len(env_obs)}")
    print("  说明: mean_gap_sigma 越小越好（<0.1 表示 _build_obs 与 build_obs 基本一致）")
    print(report.round(3).to_string(index=False))

    worst = report.sort_values("mean_gap_sigma", ascending=False).head(5)
    print("\n  mean_gap_sigma 最大的5个维度:")
    print(worst[["dim", "mean_gap_sigma", "expert_mean", "env_mean"]].round(3).to_string(index=False))


def print_single_trajectory_replay_debug(
    trajs: list[Trajectory],
    dfs: list,
    episode_idx: int = 0,
    max_steps: int = 8,
):
    """打印某条轨迹的专家观测、源 DataFrame、env 回放观测，便于逐步对比。"""
    print("\n========== 单条轨迹源数据 vs env回放 ==========")
    if not trajs or not dfs:
        print("[调试] trajs 或 dfs 为空，跳过")
        return

    ep = int(np.clip(episode_idx, 0, min(len(trajs), len(dfs)) - 1))
    traj = trajs[ep]
    df, goal = dfs[ep]

    obs_names = [
        "HV_X", "HV_Y", "HV_vx", "HV_vy",
        "dx", "dy", "yaw_prev", "d_des", "vx_prev", "vy_prev",
        "eHMI_state", "eHMI_duration",
        "Δpx", "Δpy", "Δvx", "Δvy",
    ]
    raw_cols = [
        "HV_X", "HV_Y", "HV_vx", "HV_vy", "HV_yaw",
        "ego_X", "ego_Y", "ego_vx", "ego_vy",
        "dx", "dy", "d_des", "eHMI_state", "eHMI_duration",
    ]

    env = LeftTurnEnv(dfs, playback=True)
    env_obs, _ = env.reset(options={"episode_idx": ep})

    rows: list[dict] = []
    steps = min(max_steps, len(traj.acts), len(df))
    print(f"[调试] episode={ep}, steps={steps}, goal=({goal[0]:.3f}, {goal[1]:.3f})")

    for t in range(steps):
        src_obs = traj.obs[t].astype(np.float32)
        row = df.iloc[t]
        env_obs_arr = env_obs.astype(np.float32)

        record = {"t": t}
        for i, name in enumerate(obs_names):
            record[f"src_{name}"] = float(src_obs[i])
            record[f"env_{name}"] = float(env_obs_arr[i])
            record[f"diff_{name}"] = float(env_obs_arr[i] - src_obs[i])
        for col in raw_cols:
            record[f"raw_{col}"] = float(row[col])
        if t < len(traj.acts):
            record["act_HV_ax"] = float(traj.acts[t][0])
            record["act_HV_ay"] = float(traj.acts[t][1])
        rows.append(record)

        if t < steps - 1:
            env_obs, _, terminated, truncated, _ = env.step(traj.acts[t])
            if terminated or truncated:
                break

    env.close()

    df_debug = pd.DataFrame(rows)

    key_cols = [
        "t",
        "raw_HV_X", "src_HV_X", "env_HV_X", "diff_HV_X",
        "raw_HV_Y", "src_HV_Y", "env_HV_Y", "diff_HV_Y",
        "raw_dx", "src_dx", "env_dx", "diff_dx",
        "raw_dy", "src_dy", "env_dy", "diff_dy",
        "raw_ego_X", "raw_ego_Y",
        "raw_eHMI_state", "raw_eHMI_duration",
        "act_HV_ax", "act_HV_ay",
    ]
    existing_key_cols = [c for c in key_cols if c in df_debug.columns]
    print("\n[调试] 关键字段对比：")
    print(df_debug[existing_key_cols].round(3).to_string(index=False))

    diff_cols = [c for c in df_debug.columns if c.startswith("diff_")]
    diff_summary = pd.DataFrame({
        "dim": [c.removeprefix("diff_") for c in diff_cols],
        "max_abs_diff": [float(np.max(np.abs(df_debug[c]))) for c in diff_cols],
        "mean_abs_diff": [float(np.mean(np.abs(df_debug[c]))) for c in diff_cols],
    }).sort_values("max_abs_diff", ascending=False)
    print("\n[调试] 各维度差值统计：")
    print(diff_summary.round(6).to_string(index=False))


# =============================================================
# 4. 定义策略网络
# =============================================================

def make_policy():
    return ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=POLICY_NET_ARCH,  # actor / critic 各两层
        activation_fn=POLICY_ACTIVATION_FN,
    )


# =============================================================
# 5. BC 训练
# =============================================================

def train_bc_on(trajs: list[Trajectory], save_name: str):
    """在给定轨迹上训练 BC，并保存至 checkpoints/{save_name}。"""
    print(f"\n========== BC 训练 ({save_name}) ==========")
    transitions = rollout.flatten_trajectories(trajs)
    print(f"[BC] 共 {len(transitions)} 条 transitions，来自 {len(trajs)} 条轨迹")

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

    for epoch in range(0, 100, 10):
        trainer.train(n_epochs=10)
        print(f"  epoch {epoch+10}/100")

    versioned_name = _make_next_versioned_name(save_name)
    save_path = f"{SAVE_DIR}/{versioned_name}"
    trainer.policy.save(save_path)
    print(f"[BC] 策略已保存至 {save_path}")
    return trainer


def train_bc(trajs: list[Trajectory]):
    """向后兼容：使用默认 BC_POLICY_NAME 训练。"""
    return train_bc_on(trajs, save_name=BC_POLICY_NAME)


def load_bc_trainer() -> SimpleNamespace:
    """从磁盘读取已保存的 BC 策略，并包装成带 policy 属性的对象。"""
    resolved_name = _resolve_latest_checkpoint_name(BC_POLICY_NAME)
    save_path = Path(SAVE_DIR) / resolved_name

    policy = ActorCriticPolicy.load(str(save_path))
    print(f"[BC] 已从 {save_path} 读取策略")
    return SimpleNamespace(policy=policy)


def initialize_ppo_from_bc(learner: PPO, bc_policy: ActorCriticPolicy | None):
    """用 BC 策略初始化 PPO 的 actor 相关权重，critic 保持 PPO 自己初始化。"""
    if bc_policy is None:
        print("[GAIL] 未提供 BC 策略，使用 PPO 随机初始化")
        return

    src_state = bc_policy.state_dict()
    dst_state = learner.policy.state_dict()
    copied_keys: list[str] = []
    actor_prefixes = (
        "features_extractor.",
        "pi_features_extractor.",
        "mlp_extractor.policy_net.",
        "action_net.",
    )

    for key, value in src_state.items():
        if key == "log_std" or key.startswith(actor_prefixes):
            if key in dst_state and dst_state[key].shape == value.shape:
                dst_state[key] = value.detach().clone()
                copied_keys.append(key)

    learner.policy.load_state_dict(dst_state, strict=False)
    print(f"[GAIL] 已用 BC 初始化 actor 权重，复制参数 {len(copied_keys)} 个")


class MixedRewardVecEnvWrapper(VecEnvWrapper):
    """将 GAIL 判别器奖励与环境奖励加权融合。"""

    def __init__(self, venv, env_reward_weight: float, log_interval_steps: int = 0):
        super().__init__(venv)
        self.env_reward_weight = float(env_reward_weight)
        self.log_interval_steps = int(log_interval_steps)
        self._log_step_count = 0
        self._log_acc = {
            "gail_reward": 0.0,
            "env_reward": 0.0,
            "mixed_reward": 0.0,
            "progress_reward": 0.0,
            "ref_path_penalty": 0.0,
            "ref_min_dist": 0.0,
            "goal_bonus": 0.0,
        }
        self._log_samples = 0

    def reset(self):
        return self.venv.reset()

    def _accumulate_logs(self, infos, gail_rews, env_rews, mixed_rews):
        for info, gail_r, env_r, mixed_r in zip(infos, gail_rews, env_rews, mixed_rews):
            self._log_acc["gail_reward"] += float(gail_r)
            self._log_acc["env_reward"] += float(env_r)
            self._log_acc["mixed_reward"] += float(mixed_r)
            self._log_acc["progress_reward"] += float(info.get("progress_reward", 0.0))
            self._log_acc["ref_path_penalty"] += float(info.get("ref_path_penalty", 0.0))
            self._log_acc["ref_min_dist"] += float(info.get("ref_min_dist", 0.0))
            self._log_acc["goal_bonus"] += float(info.get("goal_bonus", 0.0))
            self._log_samples += 1

    def _maybe_print_logs(self, n_envs: int):
        if self.log_interval_steps <= 0 or self._log_samples <= 0:
            return

        self._log_step_count += int(n_envs)
        if self._log_step_count < self.log_interval_steps:
            return

        denom = max(self._log_samples, 1)
        print(
            "[EnvReward] "
            f"steps≈{self._log_step_count} | "
            f"gail={self._log_acc['gail_reward'] / denom:.3f}, "
            f"env={self._log_acc['env_reward'] / denom:.3f}, "
            f"mixed={self._log_acc['mixed_reward'] / denom:.3f}, "
            f"progress={self._log_acc['progress_reward'] / denom:.3f}, "
            f"ref_penalty={self._log_acc['ref_path_penalty'] / denom:.3f}, "
            f"ref_min_dist={self._log_acc['ref_min_dist'] / denom:.3f}, "
            f"goal_bonus={self._log_acc['goal_bonus'] / denom:.3f}"
        )
        self._log_step_count = 0
        self._log_samples = 0
        for key in self._log_acc:
            self._log_acc[key] = 0.0

    def step_wait(self):
        obs, gail_rews, dones, infos = self.venv.step_wait()
        env_rews = np.array(
            [float(info.get("original_env_rew", 0.0)) for info in infos],
            dtype=np.float32,
        )
        mixed_rews = np.asarray(gail_rews, dtype=np.float32) + self.env_reward_weight * env_rews

        for info, gail_r, env_r, mixed_r in zip(infos, gail_rews, env_rews, mixed_rews):
            info["gail_reward"] = float(gail_r)
            info["env_reward"] = float(env_r)
            info["mixed_reward"] = float(mixed_r)

        self._accumulate_logs(infos, gail_rews, env_rews, mixed_rews)
        self._maybe_print_logs(len(infos))

        return obs, mixed_rews, dones, infos


# =============================================================
# 6. GAIL 训练（BC 收敛后再跑，可选）
# =============================================================

def train_gail(
    trajs: list[Trajectory],
    bc_policy: ActorCriticPolicy | None = None,
    total_timesteps: int = GAIL_FULL_TIMESTEPS,
    save_name: str = "gail_policy",
    verbose: int = 0,
):
    print("\n========== GAIL 训练 ==========")
    print(f"[GAIL] total_timesteps={total_timesteps}, save_name={save_name}")

    # GAIL 需要一个向量化环境用于策略和环境交互
    venv = DummyVecEnv([make_custom_env for _ in range(4)])

    # RL 学习器（GAIL 内部用）
    learner = PPO(
        policy="MlpPolicy",
        env=venv,
        batch_size=64,
        ent_coef=0.01,
        learning_rate=1e-4,
        n_epochs=10,
        verbose=verbose,
        policy_kwargs=dict(
            net_arch=POLICY_NET_ARCH,
            activation_fn=POLICY_ACTIVATION_FN,
        ),
    )
    initialize_ppo_from_bc(learner, bc_policy)

    # 判别器
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    trainer = GAIL(
        demonstrations=trajs,
        demo_batch_size=256 if total_timesteps <= GAIL_SMOKE_TIMESTEPS else 512,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        # LeftTurnEnv 存在提前终止（到目标/偏离轨迹），episode 长度天然可变
        # imitation 默认要求固定长度，这里显式放开。
        allow_variable_horizon=True,
    )

    if GAIL_ENV_REWARD_WEIGHT != 0.0 or ENV_LOG_INTERVAL_STEPS > 0:
        trainer.venv_train = MixedRewardVecEnvWrapper(
            trainer.venv_train,
            env_reward_weight=GAIL_ENV_REWARD_WEIGHT,
            log_interval_steps=ENV_LOG_INTERVAL_STEPS,
        )
        trainer.gen_algo.set_env(trainer.venv_train)
        if GAIL_ENV_REWARD_WEIGHT != 0.0:
            print(
                "[GAIL] 训练奖励已融合: "
                f"R_total = R_gail + {GAIL_ENV_REWARD_WEIGHT:.3f} * R_env"
            )
        if ENV_LOG_INTERVAL_STEPS > 0:
            print(
                "[GAIL] 环境奖励日志已开启: "
                f"每约 {ENV_LOG_INTERVAL_STEPS} 个环境步打印一次 ref_path_penalty / ref_min_dist"
            )

    trainer.train(total_timesteps=total_timesteps)

    versioned_name = _make_next_versioned_name(save_name)
    save_path = f"{SAVE_DIR}/{versioned_name}"
    trainer.policy.save(save_path)
    print(f"[GAIL] 策略已保存至 {save_path}")
    return trainer


def run_gail_smoke_test(trajs: list[Trajectory], bc_policy: ActorCriticPolicy | None = None):
    print("\n========== GAIL Smoke Test ==========")
    return train_gail(
        trajs=trajs,
        bc_policy=bc_policy,
        total_timesteps=GAIL_SMOKE_TIMESTEPS,
        save_name="gail_policy_smoke_init_bc",
        verbose=1,
    )


def run_gail_full_training(trajs: list[Trajectory], bc_policy: ActorCriticPolicy | None = None):
    print("\n========== GAIL Full Training ==========")
    return train_gail(
        trajs=trajs,
        bc_policy=bc_policy,
        total_timesteps=GAIL_FULL_TIMESTEPS,
        save_name="gail_policy_conservative",
        verbose=1,
    )


# =============================================================
# 7. 评估
# =============================================================

def evaluate(trainer, trajs: list[Trajectory], n_episodes=50):
    print("\n========== 评估 ==========")
    policy = trainer.policy

    rewards = []
    successes = 0
    abs_err_all = []
    sq_err_all = []
    l2_err_all = []
    ep_mae_l2 = []

    eval_trajs = trajs[:n_episodes]
    if len(eval_trajs) == 0:
        print("  [评估] 没有可用轨迹")
        return

    for traj in eval_trajs:
        ep_reward = 0.0
        ep_l2 = []

        # traj.obs 是 (T+1, obs_dim)，动作是 (T, act_dim)
        T = len(traj.acts)
        for t in range(T):
            obs = traj.obs[t]
            action, _ = policy.predict(obs[None], deterministic=True)
            hv_ax_pred, hv_ay_pred = float(action[0][0]), float(action[0][1])

            # 离线评估：预测动作与专家动作 L2 误差
            gt_hv_ax = float(traj.acts[t][0])
            gt_hv_ay = float(traj.acts[t][1])
            err_vec = np.array([hv_ax_pred - gt_hv_ax, hv_ay_pred - gt_hv_ay], dtype=np.float32)
            abs_err_all.append(np.abs(err_vec))
            sq_err_all.append(err_vec ** 2)
            error = float(np.sqrt(np.sum(err_vec ** 2)))
            l2_err_all.append(error)
            ep_l2.append(error)
            ep_reward -= error   # 用负误差当 reward

        rewards.append(ep_reward)
        ep_mae = float(np.mean(ep_l2)) if ep_l2 else np.inf
        ep_mae_l2.append(ep_mae)
        if ep_mae < 0.3:   # 单步 L2 误差阈值，可调
            successes += 1

    abs_err_all = np.array(abs_err_all, dtype=np.float32)
    sq_err_all = np.array(sq_err_all, dtype=np.float32)
    l2_err_all = np.array(l2_err_all, dtype=np.float32)

    mae = np.mean(abs_err_all, axis=0)
    rmse = np.sqrt(np.mean(sq_err_all, axis=0))
    p95 = np.quantile(abs_err_all, 0.95, axis=0)

    print(f"  评估轨迹数: {len(eval_trajs)}")
    print(f"  总步数: {len(l2_err_all)}")
    print(f"  每集平均单步L2误差均值: {np.mean(ep_mae_l2):.3f}  (各集ep_mae的均值，不受轨迹长度影响)")
    print(f"  单步L2误差: mean={np.mean(l2_err_all):.3f}, p95={np.quantile(l2_err_all, 0.95):.3f}")
    print(f"  低误差 episode 比例(单步L2<0.3): {successes}/{len(rewards)}")

    print("\n  分维动作误差（pred vs expert）:")
    print(f"    HV_ax: MAE={mae[0]:.3f}, RMSE={rmse[0]:.3f}, P95|err|={p95[0]:.3f}")
    print(f"    HV_ay: MAE={mae[1]:.3f}, RMSE={rmse[1]:.3f}, P95|err|={p95[1]:.3f}")


def evaluate_position_ade_fde(
    policy: ActorCriticPolicy,
    eval_dfs: list,
    title: str = "Position ADE/FDE",
):
    """基于位置(HV_X, HV_Y)的验证集 ADE/FDE 评估。"""
    print(f"\n========== {title} ==========")
    if not eval_dfs:
        print("  [评估] 没有可用 DataFrame")
        return

    # 关闭提前终止，让各策略都按同样时间长度对齐到专家轨迹
    env = LeftTurnEnv(
        eval_dfs,
        max_dev=1e9,
        goal_radius=-1.0,
        progress_reward_scale=ENV_PROGRESS_REWARD_SCALE,
        goal_reached_bonus=ENV_GOAL_REACHED_BONUS,
        ref_path_penalty_threshold=ENV_REF_PATH_PENALTY_THRESHOLD,
        ref_path_penalty_scale=ENV_REF_PATH_PENALTY_SCALE,
    )

    all_step_err: list[float] = []
    ep_ade: list[float] = []
    ep_fde: list[float] = []

    for ep_idx, (df, _) in enumerate(eval_dfs):
        gt_xy = df[["HV_X", "HV_Y"]].to_numpy(dtype=np.float32)
        if len(gt_xy) < 2:
            continue

        obs, _ = env.reset(options={"episode_idx": ep_idx})
        ep_err: list[float] = []

        # 对齐专家时刻 t=1..T-1（t=0 是 reset 初始状态）
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


# =============================================================
# 8. 主流程
# =============================================================

if __name__ == "__main__":
    # Step 1: 单次遍历加载，trajs[i] 与 dfs[i] 保证来自同一文件（消除索引错位）
    trajs, dfs = ced_load_all_with_dataframes(DATA_DIR)

    # Step 1.5: 0.8/0.2 划分 train/val，后续 BC/GAIL 仅用 train，val 用于评估
    train_trajs, train_dfs, val_trajs, val_dfs = split_train_val(trajs, dfs)

    # Step 2: 验证数据，确认范围正确
    validate_data(train_trajs)

    # Step 2.5: 初始化自定义环境（GAIL 共用）
    init_custom_env_dfs(train_dfs)

    # Step 2.6: 打印环境-数据观测对齐检查（trajs/dfs 已保证一一对应）
    if RUN_ALIGNMENT_CHECK:
        print_observation_alignment_check(train_trajs, train_dfs, n_steps=3000)

    # Step 2.7: 打印单条轨迹的源数据与 env 回放数据，便于逐步排查
    if RUN_SINGLE_TRAJ_DEBUG:
        print_single_trajectory_replay_debug(train_trajs, train_dfs, episode_idx=0, max_steps=8)

    # Step 3: BC 训练或读取（统一得到 bc_trainer，供后续 GAIL 初始化）
    if RUN_BC_RETRAIN:
        bc_train_trajs = trajs if BC_TRAIN_ON_ALL_DATA else train_trajs
        data_tag = "全量 trajs（train_origin 风格）" if BC_TRAIN_ON_ALL_DATA else "train split"
        print(f"\n[BC] RUN_BC_RETRAIN=True，使用 {data_tag} 重训并覆盖保存到 {BC_POLICY_NAME}")
        bc_trainer = train_bc(bc_train_trajs)
    else:
        print(f"\n[BC] RUN_BC_RETRAIN=False，读取已有模型 {BC_POLICY_NAME}")
        bc_trainer = load_bc_trainer()

    # Step 4: 在验证集上评估 BC
    print("\n[BC] 在验证集上评估：")
    evaluate(bc_trainer, val_trajs)
    evaluate_position_ade_fde(bc_trainer.policy, val_dfs, title="BC 位置 ADE/FDE")

    # Step 5: 先跑短程 GAIL smoke test（使用 BC 权重初始化）
    if RUN_GAIL_SMOKE_TEST:
        gail_smoke_trainer = run_gail_smoke_test(train_trajs, bc_policy=bc_trainer.policy)
        print("\n[GAIL Smoke] 在验证集上做离线动作误差评估：")
        evaluate(gail_smoke_trainer, val_trajs)
        evaluate_position_ade_fde(gail_smoke_trainer.policy, val_dfs, title="GAIL Smoke 位置 ADE/FDE")

    # Step 6: 完整 GAIL 训练入口（默认关闭，需要时把 RUN_GAIL_FULL_TRAIN 改成 True）
    if RUN_GAIL_FULL_TRAIN:
        gail_full_trainer = run_gail_full_training(train_trajs, bc_policy=bc_trainer.policy)
        print("\n[GAIL Full] 在验证集上做离线动作误差评估：")
        evaluate(gail_full_trainer, val_trajs)
        evaluate_position_ade_fde(gail_full_trainer.policy, val_dfs, title="GAIL Full 位置 ADE/FDE")