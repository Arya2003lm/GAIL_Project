"""AIRL 训练配置与历史实验记录。

说明：
1. `RECORDED_RUNS` 记录历史参数与结果，便于回溯。
2. `PROFILE_CONFIGS` 是当前可切换的三组实验配置。
3. 当前 `long` 配置已按最新建议做进一步优化：
   - 更低 PPO 学习率
   - 更小 clip range
   - 更高 n_epochs
   - 更大 demo batch / replay buffer
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

# =============================================================
# 1. 通用基础配置
# =============================================================

GENERAL_CONFIG = {
    "data_dir": "./data/left_turn/",
    "save_dir": "./checkpoints/",
    "bc_policy_name": "bc_policy_hv_ax_ay",
    "airl_policy_name": "airl_policy",
    "bc_policy_load_name": "bc_policy_hv_ax_ay_001",
    "train_ratio": 0.8,
    "split_seed": 28,
    "run_bc_retrain": False,
    "bc_train_on_all_data": True,
    "run_airl_smoke_test": False,
    "run_airl_full_train": True,
    "airl_smoke_timesteps": 80_000,
    "global_seed": 42,
    "n_envs": 4,
    "airl_train_max_dev": 7.5,
    "env_progress_reward_scale": 1.2,
    "env_goal_reached_bonus": 12.0,
    "env_ref_path_penalty_threshold": 3.8,
    "env_ref_path_penalty_scale": 1.5,
    "env_log_interval_steps": 2000,
    "airl_env_reward_weight": 0.2,
}

# =============================================================
# 2. 历史实验记录：原有参数 + 训练结果
# =============================================================

STATIC_RECORDED_RUNS = {
    "long_original": {
        "description": "long 模式首个效果较好的版本（作为当前调参基线）",
        "profile_params": {
            "airl_full_timesteps": 1_000_000,
            "ppo_learning_rate": 1e-4,
            "ppo_n_steps": 2048,
            "ppo_batch_size": 256,
            "ppo_n_epochs": 10,
            "ppo_ent_coef": 1e-3,
            "ppo_gamma": 0.995,
            "ppo_gae_lambda": 0.97,
            "ppo_clip_range": 0.1,
            "ppo_max_grad_norm": 0.5,
            "airl_demo_batch_size_smoke": 128,
            "airl_demo_batch_size_full": 512,
            "airl_disc_updates_per_round": 1,
            "airl_gen_replay_buffer_capacity": 4096,
        },
        "result": {
            "title": "AIRL Full 位置 ADE/FDE",
            "eval_episodes": 29,
            "total_steps": 3138,
            "ade_step_mean": 5.250,
            "ade_ep_mean": 5.022,
            "fde": 13.455,
            "ade_p95": 14.861,
            "fde_p95": 21.665,
        },
    },
}

AUTO_RECORDED_RUNS_PATH = Path(__file__).resolve().parent.parent / "results" / "airl_recorded_runs.json"


def _load_auto_recorded_runs() -> dict:
    if not AUTO_RECORDED_RUNS_PATH.exists():
        return {}

    try:
        payload = json.loads(AUTO_RECORDED_RUNS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    runs = payload.get("runs", {})
    return runs if isinstance(runs, dict) else {}


def _write_auto_recorded_runs(runs: dict) -> None:
    AUTO_RECORDED_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
    }
    AUTO_RECORDED_RUNS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


AUTO_RECORDED_RUNS = _load_auto_recorded_runs()
RECORDED_RUNS = {**deepcopy(STATIC_RECORDED_RUNS), **deepcopy(AUTO_RECORDED_RUNS)}


def build_run_record(
    *,
    run_type: str,
    profile_name: str,
    profile_params: dict,
    result: dict,
    model_path: str | None = None,
    description: str = "",
    extra: dict | None = None,
) -> dict:
    record = {
        "description": description,
        "run_type": run_type,
        "profile_name": profile_name,
        "profile_params": deepcopy(profile_params),
        "result": deepcopy(result),
        "model_path": model_path,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        record["extra"] = deepcopy(extra)
    return record


def save_run_record(run_key: str, record: dict) -> None:
    runs = _load_auto_recorded_runs()
    runs[run_key] = deepcopy(record)
    _write_auto_recorded_runs(runs)
    AUTO_RECORDED_RUNS[run_key] = deepcopy(record)
    RECORDED_RUNS[run_key] = deepcopy(record)


def make_auto_run_key(run_type: str, profile_name: str) -> str:
    """生成形如 `airl_full_long_001` 的自动编号 key，不会覆盖已有记录。"""
    prefix = f"{run_type}_{profile_name}_"
    existing = _load_auto_recorded_runs()
    # 统计同前缀的已有条数（兼容时间戳旧格式与新编号格式）
    max_idx = 0
    for key in existing:
        if key.startswith(prefix):
            suffix = key[len(prefix):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
    return f"{prefix}{max_idx + 1:03d}"

# =============================================================
# 3. 当前可切换配置
# =============================================================

# 可选: "ppo" | "airl" | "long" | "long_env0" | "stable_goal" | "stable_goal_v2"
EXPERIMENT_PROFILE = "long_env0"

PROFILE_CONFIGS = {
    # 组1：PPO 先调（加大学习能力，降低发散）
    "ppo": {
        "airl_full_timesteps": 500_000,
        "ppo_learning_rate": 1e-4,
        "ppo_n_steps": 2048,
        "ppo_batch_size": 256,
        "ppo_n_epochs": 10,
        "ppo_ent_coef": 1e-3,
        "ppo_gamma": 0.995,
        "ppo_gae_lambda": 0.97,
        "ppo_clip_range": 0.1,
        "ppo_max_grad_norm": 0.5,
        "airl_demo_batch_size_smoke": 128,
        "airl_demo_batch_size_full": 256,
        "airl_disc_updates_per_round": 2,
        "airl_gen_replay_buffer_capacity": 2048,
    },
    # 组2：在组1基础上减弱判别器压制
    "airl": {
        "airl_full_timesteps": 500_000,
        "ppo_learning_rate": 1e-4,
        "ppo_n_steps": 2048,
        "ppo_batch_size": 256,
        "ppo_n_epochs": 10,
        "ppo_ent_coef": 1e-3,
        "ppo_gamma": 0.995,
        "ppo_gae_lambda": 0.97,
        "ppo_clip_range": 0.1,
        "ppo_max_grad_norm": 0.5,
        "airl_demo_batch_size_smoke": 128,
        "airl_demo_batch_size_full": 512,
        "airl_disc_updates_per_round": 1,
        "airl_gen_replay_buffer_capacity": 4096,
    },
    # 组3：在 long_original 基础上进一步优化尾部误差与终点误差
    "long": {
        "airl_full_timesteps": 1_200_000,
        "ppo_learning_rate": 7e-5,
        "ppo_n_steps": 2048,
        "ppo_batch_size": 256,
        "ppo_n_epochs": 12,
        "ppo_ent_coef": 1e-3,
        "ppo_gamma": 0.995,
        "ppo_gae_lambda": 0.97,
        "ppo_clip_range": 0.08,
        "ppo_max_grad_norm": 0.5,
        "airl_demo_batch_size_smoke": 128,
        "airl_demo_batch_size_full": 1024,
        "airl_disc_updates_per_round": 1,
        "airl_gen_replay_buffer_capacity": 8192,
    },
    # 组4：稳健到终点优先（减少离群轨迹）
    "stable_goal": {
        "airl_full_timesteps": 1_500_000,
        "ppo_learning_rate": 5e-5,
        "ppo_n_steps": 2048,
        "ppo_batch_size": 256,
        "ppo_n_epochs": 12,
        "ppo_ent_coef": 1e-3,
        "ppo_gamma": 0.995,
        "ppo_gae_lambda": 0.97,
        "ppo_clip_range": 0.06,
        "ppo_max_grad_norm": 0.5,
        "airl_demo_batch_size_smoke": 128,
        "airl_demo_batch_size_full": 1024,
        "airl_disc_updates_per_round": 1,
        "airl_gen_replay_buffer_capacity": 8192,
    },
    # 组5：稳健到终点 v2（更强调到终点与减少中途终止）
    "stable_goal_v2": {
        "airl_full_timesteps": 1_800_000,
        "ppo_learning_rate": 5e-5,
        "ppo_n_steps": 2048,
        "ppo_batch_size": 256,
        "ppo_n_epochs": 12,
        "ppo_ent_coef": 1e-3,
        "ppo_gamma": 0.995,
        "ppo_gae_lambda": 0.97,
        "ppo_clip_range": 0.06,
        "ppo_max_grad_norm": 0.5,
        "airl_demo_batch_size_smoke": 128,
        "airl_demo_batch_size_full": 1024,
        "airl_disc_updates_per_round": 1,
        "airl_gen_replay_buffer_capacity": 8192,
    },
    # 组6：long 参数 + 环境奖励权重置 0（纯 AIRL 奖励）
    "long_env0": {
        "airl_full_timesteps": 1_200_000,
        "ppo_learning_rate": 7e-5,
        "ppo_n_steps": 2048,
        "ppo_batch_size": 256,
        "ppo_n_epochs": 12,
        "ppo_ent_coef": 1e-3,
        "ppo_gamma": 0.995,
        "ppo_gae_lambda": 0.97,
        "ppo_clip_range": 0.08,
        "ppo_max_grad_norm": 0.5,
        "airl_demo_batch_size_smoke": 128,
        "airl_demo_batch_size_full": 1024,
        "airl_disc_updates_per_round": 1,
        "airl_gen_replay_buffer_capacity": 8192,
    },
}

# profile 级别覆盖：仅影响当前激活 profile
if EXPERIMENT_PROFILE == "long_env0":
    GENERAL_CONFIG["airl_env_reward_weight"] = 0.0
    # 避免与其它 AIRL 训练结果混淆
    GENERAL_CONFIG["airl_policy_name"] = "airl_policy_long_env0"
