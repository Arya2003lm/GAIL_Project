# =============================================================
# train.py  —  无保护左转模仿学习完整训练脚本
# =============================================================
# 依赖：pip install imitation stable-baselines3 highway-env gymnasium
# =============================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 避免 OpenMP 重复加载冲突

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from pathlib import Path
from types import SimpleNamespace
from gymnasium import spaces
from imitation.data.types import Trajectory
from imitation.data import rollout
from imitation.algorithms import bc
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
try:
    from Convert_expert_data import (
        load_all_trajectories as ced_load_all_trajectories,
        validate as ced_validate_data,
        observation_space as ced_observation_space,
        action_space as ced_action_space,
    )
except ModuleNotFoundError:
    from scripts.Convert_expert_data import (
        load_all_trajectories as ced_load_all_trajectories,
        validate as ced_validate_data,
        observation_space as ced_observation_space,
        action_space as ced_action_space,
    )

# =============================================================
# 0. 全局配置
# =============================================================

N_VEHICLES  = 1                        # 最多观测几辆他车
OBS_DIM     = ced_observation_space.shape[0]
ACT_DIM     = ced_action_space.shape[0]
DATA_DIR    = "./data/left_turn/"      # .asc 数据目录
SAVE_DIR    = "./checkpoints/"
BC_POLICY_NAME = "bc_policy_hv_ax_ay_origin"
Path(SAVE_DIR).mkdir(exist_ok=True)

observation_space = ced_observation_space
action_space = ced_action_space


class HighwayActionAdapter(gym.ActionWrapper):
    """把专家动作 [HV_ax, HV_ay] 转成 highway-env 可执行动作。"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = ced_action_space

    def action(self, action: np.ndarray) -> np.ndarray:
        act = np.asarray(action, dtype=np.float32)
        hv_ax_global = float(np.clip(act[0], self.action_space.low[0], self.action_space.high[0]))
        hv_ay_global = float(np.clip(act[1], self.action_space.low[1], self.action_space.high[1]))

        ego = self.env.unwrapped.controlled_vehicles[0]
        heading = float(getattr(ego, "heading", 0.0))
        speed = float(getattr(ego, "speed", 0.0))
        wheelbase = float(getattr(ego, "LENGTH", 5.0))

        # 全局坐标系加速度 -> 车体坐标系纵/横向加速度
        a_long = hv_ax_global * np.cos(heading) + hv_ay_global * np.sin(heading)
        a_lat = -hv_ax_global * np.sin(heading) + hv_ay_global * np.cos(heading)

        # 由横向加速度近似反解转向角: a_lat = v^2 * tan(delta) / L
        speed_sq = max(speed ** 2, 1e-3)
        steering = float(np.arctan(a_lat * wheelbase / speed_sq))

        action_cfg = self.env.unwrapped.config.get("action", {})
        acc_low, acc_high = action_cfg.get("acceleration_range", [-6.0, 3.0])
        steer_low, steer_high = action_cfg.get("steering_range", [-np.pi / 3, np.pi / 3])
        return np.array(
            [
                np.clip(a_long, acc_low, acc_high),
                np.clip(steering, steer_low, steer_high),
            ],
            dtype=np.float32,
        )


class HighwayObservationAdapter(gym.Wrapper):
    """把 highway-env 状态转换为与专家数据一致的 16 维观测。"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = ced_observation_space
        self._prev_yaw = 0.0
        self._prev_vx = 0.0
        self._prev_vy = 0.0

    def _build_obs(self) -> np.ndarray:
        base = self.env.unwrapped
        ego = base.controlled_vehicles[0]
        pos = np.asarray(getattr(ego, "position", np.zeros(2)), dtype=np.float32)
        vel = np.asarray(getattr(ego, "velocity", np.zeros(2)), dtype=np.float32)
        yaw = float(getattr(ego, "heading", 0.0))
        destination = np.asarray(getattr(ego, "destination", pos), dtype=np.float32)

        dx_dy = destination - pos
        d_des = float(np.linalg.norm(dx_dy))

        # 找最近的一辆背景车
        delta_px = delta_py = delta_vx = delta_vy = 0.0
        others = [v for v in base.road.vehicles if v is not ego]
        if others:
            def distance_sq(vehicle):
                p = np.asarray(getattr(vehicle, "position", np.zeros(2)), dtype=np.float32)
                d = p - pos
                return float(d[0] ** 2 + d[1] ** 2)

            other = min(others, key=distance_sq)
            other_pos = np.asarray(getattr(other, "position", np.zeros(2)), dtype=np.float32)
            other_vel = np.asarray(getattr(other, "velocity", np.zeros(2)), dtype=np.float32)
            delta = other_pos - pos
            delta_v = other_vel - vel
            delta_px, delta_py = float(delta[0]), float(delta[1])
            delta_vx, delta_vy = float(delta_v[0]), float(delta_v[1])

        obs = np.array([
            float(pos[0]),
            float(pos[1]),
            float(vel[0]),
            float(vel[1]),
            float(dx_dy[0]),
            float(dx_dy[1]),
            self._prev_yaw,
            d_des,
            self._prev_vx,
            self._prev_vy,
            0.0,
            0.0,
            delta_px,
            delta_py,
            delta_vx,
            delta_vy,
        ], dtype=np.float32)

        self._prev_yaw = yaw
        self._prev_vx = float(vel[0])
        self._prev_vy = float(vel[1])
        return obs

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        ego = base.controlled_vehicles[0]
        vel = np.asarray(getattr(ego, "velocity", np.zeros(2)), dtype=np.float32)
        self._prev_yaw = float(getattr(ego, "heading", 0.0))
        self._prev_vx = float(vel[0])
        self._prev_vy = float(vel[1])
        return self._build_obs(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._build_obs(), reward, terminated, truncated, info


def make_highway_expert_env() -> gym.Env:
    import highway_env  # noqa: F401

    env_config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": max(N_VEHICLES + 1, 5),
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": True,
            "normalize": False,
            "flatten": False,
        },
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": [-6.0, 3.0],
            "steering_range": [-np.pi / 3, np.pi / 3],
        },
        "controlled_vehicles": 1,
        "duration": 15,
    }

    env = gym.make("intersection-v1", config=env_config)
    env = HighwayActionAdapter(env)
    env = HighwayObservationAdapter(env)
    return env


def load_all_trajectories() -> list[Trajectory]:
    return ced_load_all_trajectories(DATA_DIR)


# =============================================================
# 3. 数据验证（训练前跑一次）
# =============================================================

def validate_data(trajs: list[Trajectory]):
    ced_validate_data(trajs)


def print_observation_alignment_check(trajs: list[Trajectory], n_steps: int = 3000):
    """打印专家观测与 highway 包装观测的对齐检查结果。"""
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

    expert_obs = np.vstack([t.obs[:-1] for t in trajs]).astype(np.float32)

    env = make_highway_expert_env()
    obs, _ = env.reset()
    env_obs_list = [obs.astype(np.float32)]
    for _ in range(max(n_steps - 1, 0)):
        print(f"\r[对齐检查] 收集环境观测... {len(env_obs_list)}/{n_steps}", end="")
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        env_obs_list.append(obs.astype(np.float32))
        if terminated or truncated:
            obs, _ = env.reset()
            env_obs_list.append(obs.astype(np.float32))
    env.close()

    env_obs = np.array(env_obs_list[:n_steps], dtype=np.float32)

    exp_mean, exp_std = expert_obs.mean(axis=0), expert_obs.std(axis=0)
    env_mean, env_std = env_obs.mean(axis=0), env_obs.std(axis=0)
    mean_gap_sigma = np.abs(env_mean - exp_mean) / np.maximum(exp_std, 1e-6)

    exp_oor = ((expert_obs < observation_space.low) | (expert_obs > observation_space.high)).mean(axis=0)
    env_oor = ((env_obs < observation_space.low) | (env_obs > observation_space.high)).mean(axis=0)

    report = pd.DataFrame({
        "dim": obs_names,
        "expert_mean": exp_mean,
        "env_mean": env_mean,
        "expert_std": exp_std,
        "env_std": env_std,
        "mean_gap_sigma": mean_gap_sigma,
        "expert_oor%": exp_oor * 100,
        "env_oor%": env_oor * 100,
    })

    print("\n========== 观测对齐检查（expert vs highway wrapped env） ==========")
    print(f"  expert样本数: {len(expert_obs)}, env样本数: {len(env_obs)}")
    print("  说明: mean_gap_sigma 越小越好（<1 通常可接受）")
    print(report.round(3).to_string(index=False))

    worst = report.sort_values("mean_gap_sigma", ascending=False).head(5)
    print("\n  mean_gap_sigma 最大的5个维度:")
    print(worst[["dim", "mean_gap_sigma", "expert_mean", "env_mean"]].round(3).to_string(index=False))


# =============================================================
# 4. 定义策略网络
# =============================================================

def make_policy():
    return ActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # actor / critic 各两层
        activation_fn=torch.nn.ReLU,
    )


# =============================================================
# 5. BC 训练
# =============================================================

def train_bc(trajs: list[Trajectory]):
    print("\n========== BC 训练 ==========")
    transitions = rollout.flatten_trajectories(trajs)
    print(f"[BC] 共 {len(transitions)} 条 transitions")

    trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=transitions,
        policy=make_policy(),
        rng=np.random.default_rng(42),
        batch_size=64,
        ent_weight=1e-3,    # 熵正则：防止策略退化为确定性
        l2_weight=1e-4,     # 权重衰减
    )

    # 每 10 epoch 打印一次 loss
    for epoch in range(0, 100, 10):
        trainer.train(n_epochs=10)
        print(f"  epoch {epoch+10}/100")

    save_path = f"{SAVE_DIR}/{BC_POLICY_NAME}"
    trainer.policy.save(save_path)
    print(f"[BC] 策略已保存至 {save_path}")
    return trainer


def load_bc_trainer() -> SimpleNamespace:
    """从磁盘读取已保存的 BC 策略，并包装成带 policy 属性的对象。"""
    save_path = Path(SAVE_DIR) / BC_POLICY_NAME
    if not save_path.exists():
        raise FileNotFoundError(f"未找到已保存的 BC 策略: {save_path}")

    policy = ActorCriticPolicy.load(str(save_path))
    print(f"[BC] 已从 {save_path} 读取策略")
    return SimpleNamespace(policy=policy)


# =============================================================
# 6. GAIL 训练（BC 收敛后再跑，可选）
# =============================================================

def train_gail(trajs: list[Trajectory]):
    print("\n========== GAIL 训练 ==========")

    # GAIL 需要一个向量化环境用于策略和环境交互
    # 这里把 highway-env 包装成与专家观测/动作语义尽量一致的形式
    venv = DummyVecEnv([make_highway_expert_env for _ in range(4)])

    # RL 学习器（GAIL 内部用）
    learner = PPO(
        policy="MlpPolicy",
        env=venv,
        batch_size=64,
        ent_coef=0.01,
        learning_rate=3e-4,
        n_epochs=10,
        verbose=0,
    )

    # 判别器
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
    )

    trainer = GAIL(
        demonstrations=trajs,
        demo_batch_size=512,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    trainer.train(total_timesteps=300_000)

    save_path = f"{SAVE_DIR}/gail_policy"
    trainer.policy.save(save_path)
    print(f"[GAIL] 策略已保存至 {save_path}")
    return trainer


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


# =============================================================
# 8. 主流程
# =============================================================

if __name__ == "__main__":
    # Step 1: 加载数据
    trajs = load_all_trajectories()

    # Step 2: 验证数据，确认范围正确
    validate_data(trajs)

    # Step 2.5: 打印环境-数据观测对齐检查
    print_observation_alignment_check(trajs, n_steps=3000)

    # Step 3: 优先读取已训练好的 BC；若不存在则重新训练
    try:
        bc_trainer = load_bc_trainer()
    except FileNotFoundError:
        print("[BC] 未找到已保存策略，开始重新训练")
        bc_trainer = train_bc(trajs)

    # Step 4: 评估 BC
    evaluate(bc_trainer, trajs)

    # Step 5: 如果 BC 效果不够好，再跑 GAIL（慢，需要 highway-env）
    # gail_trainer = train_gail(trajs)
    # evaluate(gail_trainer)