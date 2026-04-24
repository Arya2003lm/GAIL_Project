"""
train_airl.py  —  参考 train_gail.py 的 AIRL 模仿学习训练脚本

核心特性：
1) 使用 Convert_expert_data.py 读取并构造专家轨迹
2) 环境固定为 myenv.py 中的 LeftTurnEnv
3) 可选 BC 预训练，并用 BC actor 权重初始化 AIRL 的 PPO learner
4) 训练后保存 AIRL 策略，并在验证集做离线动作误差评估
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from imitation.algorithms import bc
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

try:
	from Convert_expert_data import (
		load_all_with_dataframes as ced_load_all_with_dataframes,
		validate as ced_validate_data,
		observation_space as ced_observation_space,
		action_space as ced_action_space,
	)
	from config_airl import (
		EXPERIMENT_PROFILE,
		GENERAL_CONFIG,
		PROFILE_CONFIGS,
		build_run_record,
		make_auto_run_key,
		save_run_record,
	)
	from myenv import LeftTurnEnv
except ModuleNotFoundError:
	from scripts.Convert_expert_data import (
		load_all_with_dataframes as ced_load_all_with_dataframes,
		validate as ced_validate_data,
		observation_space as ced_observation_space,
		action_space as ced_action_space,
	)
	from scripts.config_airl import (
		EXPERIMENT_PROFILE,
		GENERAL_CONFIG,
		PROFILE_CONFIGS,
		build_run_record,
		make_auto_run_key,
		save_run_record,
	)
	from scripts.myenv import LeftTurnEnv


# =============================================================
# 0. 全局配置
# =============================================================

DATA_DIR = GENERAL_CONFIG["data_dir"]
SAVE_DIR = GENERAL_CONFIG["save_dir"]

BC_POLICY_NAME = GENERAL_CONFIG["bc_policy_name"]
AIRL_POLICY_NAME = GENERAL_CONFIG["airl_policy_name"]
BC_POLICY_LOAD_NAME = GENERAL_CONFIG["bc_policy_load_name"]

TRAIN_RATIO = GENERAL_CONFIG["train_ratio"]
SPLIT_SEED = GENERAL_CONFIG["split_seed"]

RUN_BC_RETRAIN = GENERAL_CONFIG["run_bc_retrain"]
BC_TRAIN_ON_ALL_DATA = GENERAL_CONFIG["bc_train_on_all_data"]

RUN_AIRL_SMOKE_TEST = GENERAL_CONFIG["run_airl_smoke_test"]
RUN_AIRL_FULL_TRAIN = GENERAL_CONFIG["run_airl_full_train"]

AIRL_SMOKE_TIMESTEPS = GENERAL_CONFIG["airl_smoke_timesteps"]

GLOBAL_SEED = GENERAL_CONFIG["global_seed"]
N_ENVS = GENERAL_CONFIG["n_envs"]

if EXPERIMENT_PROFILE not in PROFILE_CONFIGS:
	raise ValueError(
		f"未知 EXPERIMENT_PROFILE={EXPERIMENT_PROFILE}，可选: {list(PROFILE_CONFIGS.keys())}"
	)

_active_cfg = PROFILE_CONFIGS[EXPERIMENT_PROFILE]
AIRL_FULL_TIMESTEPS = _active_cfg["airl_full_timesteps"]
PPO_LEARNING_RATE = _active_cfg["ppo_learning_rate"]
PPO_N_STEPS = _active_cfg["ppo_n_steps"]
PPO_BATCH_SIZE = _active_cfg["ppo_batch_size"]
PPO_N_EPOCHS = _active_cfg["ppo_n_epochs"]
PPO_ENT_COEF = _active_cfg["ppo_ent_coef"]
PPO_GAMMA = _active_cfg["ppo_gamma"]
PPO_GAE_LAMBDA = _active_cfg["ppo_gae_lambda"]
PPO_CLIP_RANGE = _active_cfg["ppo_clip_range"]
PPO_MAX_GRAD_NORM = _active_cfg["ppo_max_grad_norm"]
AIRL_DEMO_BATCH_SIZE_SMOKE = _active_cfg["airl_demo_batch_size_smoke"]
AIRL_DEMO_BATCH_SIZE_FULL = _active_cfg["airl_demo_batch_size_full"]
AIRL_DISC_UPDATES_PER_ROUND = _active_cfg["airl_disc_updates_per_round"]
AIRL_GEN_REPLAY_BUFFER_CAPACITY = _active_cfg["airl_gen_replay_buffer_capacity"]

ENV_PROGRESS_REWARD_SCALE = GENERAL_CONFIG["env_progress_reward_scale"]
ENV_GOAL_REACHED_BONUS = GENERAL_CONFIG["env_goal_reached_bonus"]
ENV_REF_PATH_PENALTY_THRESHOLD = GENERAL_CONFIG["env_ref_path_penalty_threshold"]
ENV_REF_PATH_PENALTY_SCALE = GENERAL_CONFIG["env_ref_path_penalty_scale"]
AIRL_TRAIN_MAX_DEV = GENERAL_CONFIG["airl_train_max_dev"]
ENV_LOG_INTERVAL_STEPS = GENERAL_CONFIG["env_log_interval_steps"]
AIRL_ENV_REWARD_WEIGHT = GENERAL_CONFIG["airl_env_reward_weight"]

POLICY_NET_ARCH = dict(pi=[256, 256], vf=[256, 256])
POLICY_ACTIVATION_FN = torch.nn.ReLU

Path(SAVE_DIR).mkdir(exist_ok=True)

observation_space = ced_observation_space
action_space = ced_action_space


def get_active_profile_params() -> dict:
	return dict(_active_cfg)


def _make_next_versioned_name(base_name: str, save_dir: str = SAVE_DIR) -> str:
	"""返回下一个可用名称，如 base_name_001、base_name_002。"""
	save_root = Path(save_dir)
	pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
	max_idx = 0

	for file in save_root.glob(f"{base_name}_*"):
		if not file.is_file():
			continue
		candidate_name = file.stem if file.suffix == ".zip" else file.name
		m = pattern.match(candidate_name)
		if m:
			max_idx = max(max_idx, int(m.group(1)))

	return f"{base_name}_{max_idx + 1:03d}"


def _resolve_latest_checkpoint_name(base_name: str, save_dir: str = SAVE_DIR) -> str:
	"""优先返回基础名；若不存在则返回最新序号版本名。"""
	save_root = Path(save_dir)
	base_path_no_ext = save_root / base_name
	base_path_zip = save_root / f"{base_name}.zip"
	if base_path_no_ext.exists():
		return base_path_no_ext.name
	if base_path_zip.exists():
		return base_path_zip.name

	pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
	candidates: list[tuple[int, str]] = []
	for file in save_root.glob(f"{base_name}_*"):
		if not file.is_file():
			continue
		candidate_name = file.stem if file.suffix == ".zip" else file.name
		m = pattern.match(candidate_name)
		if m:
			candidates.append((int(m.group(1)), file.name))

	if not candidates:
		raise FileNotFoundError(
			f"未找到模型: {base_path_no_ext} 或 {base_path_zip} 或 {save_root / (base_name + '_###')}"
		)

	candidates.sort(key=lambda x: x[0])
	return candidates[-1][1]


# _dfs 存储所有轨迹 DataFrame，make_custom_env 通过闭包引用
_dfs: list = []


def init_custom_env_dfs(dfs: list) -> None:
	global _dfs
	_dfs = dfs


def make_custom_env():
	"""供 DummyVecEnv 调用的工厂函数，返回 LeftTurnEnv。"""
	if not _dfs:
		raise RuntimeError("请先调用 init_custom_env_dfs(dfs) 初始化数据")
	return LeftTurnEnv(
		_dfs,
		max_dev=AIRL_TRAIN_MAX_DEV,
		progress_reward_scale=ENV_PROGRESS_REWARD_SCALE,
		goal_reached_bonus=ENV_GOAL_REACHED_BONUS,
		ref_path_penalty_threshold=ENV_REF_PATH_PENALTY_THRESHOLD,
		ref_path_penalty_scale=ENV_REF_PATH_PENALTY_SCALE,
	)


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


def validate_data(trajs: list[Trajectory]):
	ced_validate_data(trajs)


def make_policy():
	return ActorCriticPolicy(
		observation_space=observation_space,
		action_space=action_space,
		lr_schedule=lambda _: 3e-4,
		net_arch=POLICY_NET_ARCH,
		activation_fn=POLICY_ACTIVATION_FN,
	)


def train_bc_on(trajs: list[Trajectory], save_name: str):
	"""在给定轨迹上训练 BC，并保存策略。"""
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
		print(f"  epoch {epoch + 10}/100")

	versioned_name = _make_next_versioned_name(save_name)
	save_path = f"{SAVE_DIR}/{versioned_name}"
	trainer.policy.save(save_path)
	trainer.saved_path = save_path
	print(f"[BC] 策略已保存至 {save_path}")
	return trainer


def load_bc_trainer() -> SimpleNamespace:
	"""读取磁盘上已有 BC 策略，并包装成带 policy 属性对象。"""
	if BC_POLICY_LOAD_NAME is not None and str(BC_POLICY_LOAD_NAME).strip() != "":
		print(f"[BC] 指定加载配置 BC_POLICY_LOAD_NAME={BC_POLICY_LOAD_NAME}")
		candidate = Path(str(BC_POLICY_LOAD_NAME).strip())
		if not candidate.is_absolute():
			candidate = Path(SAVE_DIR) / candidate

		# 兼容填写不带 .zip 的名称
		if not candidate.exists() and candidate.suffix == "" and candidate.with_suffix(".zip").exists():
			candidate = candidate.with_suffix(".zip")

		if not candidate.exists():
			raise FileNotFoundError(f"指定 BC 模型不存在: {candidate}")

		policy = ActorCriticPolicy.load(str(candidate))
		print(f"[BC] 已按指定路径读取策略: {candidate}")
		return SimpleNamespace(policy=policy, loaded_path=str(candidate))

	resolved_name = _resolve_latest_checkpoint_name(BC_POLICY_NAME)
	save_path = Path(SAVE_DIR) / resolved_name

	policy = ActorCriticPolicy.load(str(save_path))
	print(f"[BC] 已从 {save_path} 读取策略（自动解析最新版本）")
	return SimpleNamespace(policy=policy, loaded_path=str(save_path))


def initialize_ppo_from_bc(learner: PPO, bc_policy: ActorCriticPolicy | None):
	"""用 BC 策略初始化 PPO actor 相关权重，critic 保持随机初始化。"""
	if bc_policy is None:
		print("[AIRL] 未提供 BC 策略，使用 PPO 随机初始化")
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
	print(f"[AIRL] 已用 BC 初始化 actor 权重，复制参数 {len(copied_keys)} 个")


class MixedRewardVecEnvWrapper(VecEnvWrapper):
	"""将 AIRL 奖励与环境奖励加权融合，同时打印奖励分量日志。"""

	def __init__(self, venv, env_reward_weight: float, log_interval_steps: int = 0):
		super().__init__(venv)
		self.env_reward_weight = float(env_reward_weight)
		self.log_interval_steps = int(log_interval_steps)
		self._step_count = 0
		self._acc = {
			"airl_reward": 0.0,
			"env_reward": 0.0,
			"mixed_reward": 0.0,
			"progress_reward": 0.0,
			"ref_path_penalty": 0.0,
			"ref_min_dist": 0.0,
			"goal_bonus": 0.0,
		}
		self._samples = 0

	def reset(self):
		return self.venv.reset()

	def _log_step(self, infos, airl_rews, env_rews, mixed_rews):
		for info, airl_r, env_r, mixed_r in zip(infos, airl_rews, env_rews, mixed_rews):
			self._acc["airl_reward"] += float(airl_r)
			self._acc["env_reward"] += float(env_r)
			self._acc["mixed_reward"] += float(mixed_r)
			self._acc["progress_reward"] += float(info.get("progress_reward", 0.0))
			self._acc["ref_path_penalty"] += float(info.get("ref_path_penalty", 0.0))
			self._acc["ref_min_dist"] += float(info.get("ref_min_dist", 0.0))
			self._acc["goal_bonus"] += float(info.get("goal_bonus", 0.0))
			self._samples += 1

		self._step_count += len(infos)
		if self.log_interval_steps <= 0 or self._step_count < self.log_interval_steps or self._samples <= 0:
			return

		d = max(self._samples, 1)
		print(
			"[EnvReward] "
			f"steps≈{self._step_count} | "
			f"airl={self._acc['airl_reward']/d:.3f}, "
			f"env={self._acc['env_reward']/d:.3f}, "
			f"mixed={self._acc['mixed_reward']/d:.3f}, "
			f"progress={self._acc['progress_reward']/d:.3f}, "
			f"ref_penalty={self._acc['ref_path_penalty']/d:.3f}, "
			f"ref_min_dist={self._acc['ref_min_dist']/d:.3f}, "
			f"goal_bonus={self._acc['goal_bonus']/d:.3f}"
		)
		self._step_count = 0
		self._samples = 0
		for k in self._acc:
			self._acc[k] = 0.0

	def step_wait(self):
		obs, airl_rews, dones, infos = self.venv.step_wait()
		env_rews = np.array(
			[float(info.get("original_env_rew", 0.0)) for info in infos],
			dtype=np.float32,
		)
		mixed_rews = np.asarray(airl_rews, dtype=np.float32) + self.env_reward_weight * env_rews

		for info, airl_r, env_r, mixed_r in zip(infos, airl_rews, env_rews, mixed_rews):
			info["airl_reward"] = float(airl_r)
			info["env_reward"] = float(env_r)
			info["mixed_reward"] = float(mixed_r)

		self._log_step(infos, airl_rews, env_rews, mixed_rews)
		return obs, mixed_rews, dones, infos


def train_airl(
	trajs: list[Trajectory],
	bc_policy: ActorCriticPolicy | None = None,
	total_timesteps: int = AIRL_FULL_TIMESTEPS,
	save_name: str = AIRL_POLICY_NAME,
	verbose: int = 1,
):
	print("\n========== AIRL 训练 ==========")
	print(f"[AIRL] total_timesteps={total_timesteps}, save_name={save_name}")
	print(f"[AIRL] EXPERIMENT_PROFILE={EXPERIMENT_PROFILE}")
	print(
		"[AIRL] 稳健参数: "
		f"n_envs={N_ENVS}, ppo_lr={PPO_LEARNING_RATE}, "
		f"ppo_n_steps={PPO_N_STEPS}, ppo_batch={PPO_BATCH_SIZE}, "
		f"disc_updates={AIRL_DISC_UPDATES_PER_ROUND}, "
		f"env_reward_weight={AIRL_ENV_REWARD_WEIGHT}"
	)

	venv = DummyVecEnv([make_custom_env for _ in range(N_ENVS)])

	learner = PPO(
		policy="MlpPolicy",
		env=venv,
		seed=GLOBAL_SEED,
		n_steps=PPO_N_STEPS,
		batch_size=PPO_BATCH_SIZE,
		ent_coef=PPO_ENT_COEF,
		learning_rate=PPO_LEARNING_RATE,
		n_epochs=PPO_N_EPOCHS,
		gamma=PPO_GAMMA,
		gae_lambda=PPO_GAE_LAMBDA,
		clip_range=PPO_CLIP_RANGE,
		max_grad_norm=PPO_MAX_GRAD_NORM,
		verbose=verbose,
		policy_kwargs=dict(
			net_arch=POLICY_NET_ARCH,
			activation_fn=POLICY_ACTIVATION_FN,
		),
	)
	initialize_ppo_from_bc(learner, bc_policy)

	reward_net = BasicShapedRewardNet(
		observation_space=venv.observation_space,
		action_space=venv.action_space,
		normalize_input_layer=RunningNorm,
	)

	trainer = AIRL(
		demonstrations=trajs,
		demo_batch_size=(
			AIRL_DEMO_BATCH_SIZE_SMOKE
			if total_timesteps <= AIRL_SMOKE_TIMESTEPS
			else AIRL_DEMO_BATCH_SIZE_FULL
		),
		gen_replay_buffer_capacity=AIRL_GEN_REPLAY_BUFFER_CAPACITY,
		n_disc_updates_per_round=AIRL_DISC_UPDATES_PER_ROUND,
		venv=venv,
		gen_algo=learner,
		reward_net=reward_net,
		# LeftTurnEnv 会出现提前终止，显式允许可变长度 episode
		allow_variable_horizon=True,
	)

	# 挂载混合奖励 wrapper：R_total = R_airl + alpha * R_env
	if AIRL_ENV_REWARD_WEIGHT != 0.0 or ENV_LOG_INTERVAL_STEPS > 0:
		trainer.venv_train = MixedRewardVecEnvWrapper(
			trainer.venv_train,
			env_reward_weight=AIRL_ENV_REWARD_WEIGHT,
			log_interval_steps=ENV_LOG_INTERVAL_STEPS,
		)
		trainer.gen_algo.set_env(trainer.venv_train)
		print(
			"[AIRL] 训练奖励已融合: "
			f"R_total = R_airl + {AIRL_ENV_REWARD_WEIGHT:.3f} * R_env"
		)
		if ENV_LOG_INTERVAL_STEPS > 0:
			print(f"[AIRL] 环境奖励日志已开启：每约 {ENV_LOG_INTERVAL_STEPS} 步打印一次")

	trainer.train(total_timesteps=total_timesteps)

	versioned_name = _make_next_versioned_name(save_name)
	save_path = f"{SAVE_DIR}/{versioned_name}"
	trainer.policy.save(save_path)
	trainer.saved_path = save_path
	print(f"[AIRL] 策略已保存至 {save_path}")
	return trainer


def run_airl_smoke_test(trajs: list[Trajectory], bc_policy: ActorCriticPolicy | None = None):
	print("\n========== AIRL Smoke Test ==========")
	return train_airl(
		trajs=trajs,
		bc_policy=bc_policy,
		total_timesteps=AIRL_SMOKE_TIMESTEPS,
		save_name=f"{AIRL_POLICY_NAME}_smoke_init_bc",
		verbose=1,
	)


def run_airl_full_training(trajs: list[Trajectory], bc_policy: ActorCriticPolicy | None = None):
	print("\n========== AIRL Full Training ==========")
	return train_airl(
		trajs=trajs,
		bc_policy=bc_policy,
		total_timesteps=AIRL_FULL_TIMESTEPS,
		save_name=AIRL_POLICY_NAME,
		verbose=1,
	)


def evaluate_action_error(trainer, trajs: list[Trajectory], n_episodes=50):
	"""离线动作误差评估（pred action vs expert action）。"""
	print("\n========== 动作误差评估 ==========")
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
		return None

	for traj in eval_trajs:
		ep_reward = 0.0
		ep_l2 = []
		T = len(traj.acts)

		for t in range(T):
			obs = traj.obs[t]
			action, _ = policy.predict(obs[None], deterministic=True)
			hv_ax_pred, hv_ay_pred = float(action[0][0]), float(action[0][1])

			gt_hv_ax = float(traj.acts[t][0])
			gt_hv_ay = float(traj.acts[t][1])

			err_vec = np.array([hv_ax_pred - gt_hv_ax, hv_ay_pred - gt_hv_ay], dtype=np.float32)
			abs_err_all.append(np.abs(err_vec))
			sq_err_all.append(err_vec ** 2)
			error = float(np.sqrt(np.sum(err_vec ** 2)))
			l2_err_all.append(error)
			ep_l2.append(error)
			ep_reward -= error

		rewards.append(ep_reward)
		ep_mae = float(np.mean(ep_l2)) if ep_l2 else np.inf
		ep_mae_l2.append(ep_mae)
		if ep_mae < 0.3:
			successes += 1

	abs_err_all = np.array(abs_err_all, dtype=np.float32)
	sq_err_all = np.array(sq_err_all, dtype=np.float32)
	l2_err_all = np.array(l2_err_all, dtype=np.float32)

	mae = np.mean(abs_err_all, axis=0)
	rmse = np.sqrt(np.mean(sq_err_all, axis=0))
	p95 = np.quantile(abs_err_all, 0.95, axis=0)

	print(f"  评估轨迹数: {len(eval_trajs)}")
	print(f"  总步数: {len(l2_err_all)}")
	print(f"  每集平均单步L2误差均值: {np.mean(ep_mae_l2):.3f}")
	print(f"  单步L2误差: mean={np.mean(l2_err_all):.3f}, p95={np.quantile(l2_err_all, 0.95):.3f}")
	print(f"  低误差 episode 比例(单步L2<0.3): {successes}/{len(rewards)}")

	print("\n  分维动作误差（pred vs expert）:")
	print(f"    HV_ax: MAE={mae[0]:.3f}, RMSE={rmse[0]:.3f}, P95|err|={p95[0]:.3f}")
	print(f"    HV_ay: MAE={mae[1]:.3f}, RMSE={rmse[1]:.3f}, P95|err|={p95[1]:.3f}")

	return {
		"title": "动作误差评估",
		"eval_trajectories": int(len(eval_trajs)),
		"total_steps": int(len(l2_err_all)),
		"ep_mae_l2_mean": float(np.mean(ep_mae_l2)),
		"l2_mean": float(np.mean(l2_err_all)),
		"l2_p95": float(np.quantile(l2_err_all, 0.95)),
		"success_episodes": int(successes),
		"success_ratio": float(successes / max(len(rewards), 1)),
		"hv_ax_mae": float(mae[0]),
		"hv_ax_rmse": float(rmse[0]),
		"hv_ax_p95": float(p95[0]),
		"hv_ay_mae": float(mae[1]),
		"hv_ay_rmse": float(rmse[1]),
		"hv_ay_p95": float(p95[1]),
	}


def evaluate_position_ade_fde(
	policy: ActorCriticPolicy,
	eval_dfs: list,
	title: str = "Position ADE/FDE",
):
	"""基于位置(HV_X, HV_Y)的验证集 ADE/FDE 评估。"""
	print(f"\n========== {title} ==========")
	if not eval_dfs:
		print("  [评估] 没有可用 DataFrame")
		return None

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
		return None

	print(f"  评估轨迹数: {len(ep_ade)}")
	print(f"  总步数: {len(all_step_err)}")
	print(f"  ADE (step mean): {np.mean(all_step_err):.3f}")
	print(f"  ADE (ep mean):   {np.mean(ep_ade):.3f}")
	print(f"  FDE:             {np.mean(ep_fde):.3f}")
	print(f"  ADE p95:         {np.quantile(all_step_err, 0.95):.3f}")
	print(f"  FDE p95:         {np.quantile(ep_fde, 0.95):.3f}")

	return {
		"title": title,
		"eval_episodes": int(len(ep_ade)),
		"total_steps": int(len(all_step_err)),
		"ade_step_mean": float(np.mean(all_step_err)),
		"ade_ep_mean": float(np.mean(ep_ade)),
		"fde": float(np.mean(ep_fde)),
		"ade_p95": float(np.quantile(all_step_err, 0.95)),
		"fde_p95": float(np.quantile(ep_fde, 0.95)),
	}


def auto_record_training_result(
	run_type: str,
	profile_name: str,
	model_path: str | None,
	action_metrics: dict | None,
	position_metrics: dict | None,
	extra: dict | None = None,
):
	result = {
		"action_metrics": action_metrics,
		"position_metrics": position_metrics,
	}
	record = build_run_record(
		run_type=run_type,
		profile_name=profile_name,
		profile_params=get_active_profile_params(),
		result=result,
		model_path=model_path,
		description=f"自动记录: {run_type} / profile={profile_name}",
		extra=extra,
	)
	run_key = make_auto_run_key(run_type, profile_name)
	save_run_record(run_key, record)
	print(f"[RECORDED_RUNS] 已自动登记: {run_key}")


if __name__ == "__main__":
	# Step 1: 单次遍历加载，确保 trajs[i] 与 dfs[i] 一一对应
	trajs, dfs = ced_load_all_with_dataframes(DATA_DIR)

	# Step 2: 划分 train/val，后续 AIRL 只用 train，val 用于评估
	train_trajs, train_dfs, val_trajs, val_dfs = split_train_val(trajs, dfs)

	# Step 3: 数据校验
	validate_data(train_trajs)

	# Step 4: 初始化环境数据（AIRL 与评估共享）
	init_custom_env_dfs(train_dfs)

	# Step 5: BC 训练或读取（用于 AIRL 初始化）
	if RUN_BC_RETRAIN:
		bc_train_trajs = trajs if BC_TRAIN_ON_ALL_DATA else train_trajs
		data_tag = "全量 trajs（train_origin 风格）" if BC_TRAIN_ON_ALL_DATA else "train split"
		print(f"\n[BC] RUN_BC_RETRAIN=True，使用 {data_tag} 重训并保存")
		bc_trainer = train_bc_on(bc_train_trajs, save_name=BC_POLICY_NAME)
	else:
		if BC_POLICY_LOAD_NAME is not None and str(BC_POLICY_LOAD_NAME).strip() != "":
			print(f"\n[BC] RUN_BC_RETRAIN=False，优先读取指定模型 {BC_POLICY_LOAD_NAME}")
		else:
			print(f"\n[BC] RUN_BC_RETRAIN=False，读取已有模型 {BC_POLICY_NAME}（自动最新）")
		bc_trainer = load_bc_trainer()

	# Step 6: 先评估 BC
	print("\n[BC] 在验证集上评估：")
	bc_action_metrics = evaluate_action_error(bc_trainer, val_trajs)
	bc_position_metrics = evaluate_position_ade_fde(bc_trainer.policy, val_dfs, title="BC 位置 ADE/FDE")
	if RUN_BC_RETRAIN:
		auto_record_training_result(
			run_type="bc_train",
			profile_name=EXPERIMENT_PROFILE,
			model_path=getattr(bc_trainer, "saved_path", None),
			action_metrics=bc_action_metrics,
			position_metrics=bc_position_metrics,
			extra={
				"bc_policy_name": BC_POLICY_NAME,
				"bc_train_on_all_data": BC_TRAIN_ON_ALL_DATA,
			},
		)

	# Step 7: AIRL Smoke Test（可选）
	if RUN_AIRL_SMOKE_TEST:
		airl_smoke_trainer = run_airl_smoke_test(train_trajs, bc_policy=bc_trainer.policy)
		print("\n[AIRL Smoke] 在验证集上评估：")
		smoke_action_metrics = evaluate_action_error(airl_smoke_trainer, val_trajs)
		smoke_position_metrics = evaluate_position_ade_fde(airl_smoke_trainer.policy, val_dfs, title="AIRL Smoke 位置 ADE/FDE")
		auto_record_training_result(
			run_type="airl_smoke",
			profile_name=EXPERIMENT_PROFILE,
			model_path=getattr(airl_smoke_trainer, "saved_path", None),
			action_metrics=smoke_action_metrics,
			position_metrics=smoke_position_metrics,
			extra={
				"bc_init_path": getattr(bc_trainer, "saved_path", getattr(bc_trainer, "loaded_path", None)),
			},
		)

	# Step 8: AIRL Full Training（默认开启）
	if RUN_AIRL_FULL_TRAIN:
		airl_full_trainer = run_airl_full_training(train_trajs, bc_policy=bc_trainer.policy)
		print("\n[AIRL Full] 在验证集上评估：")
		full_action_metrics = evaluate_action_error(airl_full_trainer, val_trajs)
		full_position_metrics = evaluate_position_ade_fde(airl_full_trainer.policy, val_dfs, title="AIRL Full 位置 ADE/FDE")
		auto_record_training_result(
			run_type="airl_full",
			profile_name=EXPERIMENT_PROFILE,
			model_path=getattr(airl_full_trainer, "saved_path", None),
			action_metrics=full_action_metrics,
			position_metrics=full_position_metrics,
			extra={
				"bc_init_path": getattr(bc_trainer, "saved_path", getattr(bc_trainer, "loaded_path", None)),
			},
		)
