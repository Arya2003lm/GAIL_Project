# =============================================================
# myenv.py  —  无保护左转自定义 Gymnasium 环境
# =============================================================
# 设计原则：
#   - AV（自动驾驶车）轨迹 + eHMI 直接从真实数据逐帧回放
#   - HV（人工驾驶车）由 GAIL/BC 策略控制，动力学用 Euler 积分仿真
#   - 观测维度 16，与 Convert_expert_data.py 中专家数据完全一致
#   - 动作 [HV_ax, HV_ay]：全局坐标系下的加速度分量，单位 m/s²
# =============================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

try:
    from Convert_expert_data import (
        obs_low,
        obs_high,
        action_space as ced_action_space,
        observation_space as ced_observation_space,
    )
except ModuleNotFoundError:
    from scripts.Convert_expert_data import (
        obs_low,
        obs_high,
        action_space as ced_action_space,
        observation_space as ced_observation_space,
    )


class LeftTurnEnv(gym.Env):
    """
    无保护左转自定义环境。

    Parameters
    ----------
    dataframes : list[tuple[pd.DataFrame, np.ndarray]]
        ``load_all_dataframes()`` 的返回值。
        每个元素为 (df_含派生列, goal_xy)。
    max_dev : float
        HV 偏离参考轨迹超过此距离（m）时提前终止。
    goal_radius : float
        到达目标点的判定半径（m）。
    """

    metadata = {"render_modes": []}

    DT: float = 0.1  # 时间步长（s），与数据采集频率保持一致

    def __init__(
        self,
        dataframes: list[tuple[pd.DataFrame, np.ndarray]],
        max_dev: float = 10.0,
        goal_radius: float = 2.0,
        playback: bool = False,
    ):
        super().__init__()

        if not dataframes:
            raise ValueError("dataframes 不能为空，请先调用 load_all_dataframes()")

        self.dataframes = dataframes
        self.max_dev = max_dev
        self.goal_radius = goal_radius
        # playback=True: step() 直接从 DataFrame 读取 HV 状态，不进行 Euler 积分
        # 用于对齐检查，验证 _build_obs 的构造逻辑是否正确
        self.playback = playback

        # 观测/动作空间与专家数据完全一致
        self.observation_space = ced_observation_space
        self.action_space = ced_action_space

        # 运行时状态（reset 时初始化）
        self.current_df: pd.DataFrame = dataframes[0][0]
        self.current_goal: np.ndarray = dataframes[0][1]
        self.t: int = 0
        self.max_t: int = 0

        # HV 状态（由策略动作驱动）
        self._hv_x: float = 0.0
        self._hv_y: float = 0.0
        self._hv_vx: float = 0.0
        self._hv_vy: float = 0.0
        self._hv_yaw: float = 0.0

        # 上一步状态（用于构建当前帧的 prev_* 观测维度）
        self._prev_hv_vx: float = 0.0
        self._prev_hv_vy: float = 0.0
        self._prev_hv_yaw: float = 0.0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 支持 options={"episode_idx": i} 指定轨迹（用于对齐检查）
        if options is not None and "episode_idx" in options:
            idx = int(options["episode_idx"]) % len(self.dataframes)
        else:
            idx = int(self.np_random.integers(len(self.dataframes)))
        self.current_df, self.current_goal = self.dataframes[idx]
        self.max_t = len(self.current_df) - 1
        self.t = 0

        # 从数据第 0 帧初始化 HV 状态
        row0 = self.current_df.iloc[0]
        self._hv_x   = float(row0["HV_X"])
        self._hv_y   = float(row0["HV_Y"])
        self._hv_vx  = float(row0["HV_vx"])
        self._hv_vy  = float(row0["HV_vy"])
        self._hv_yaw = float(row0["HV_yaw"])

        # 第 0 帧的 prev_* 也设为第 0 帧（没有更早的历史）
        self._prev_hv_vx  = self._hv_vx
        self._prev_hv_vy  = self._hv_vy
        self._prev_hv_yaw = self._hv_yaw

        return self._build_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        # 保存上一帧状态（供 obs 的 prev_* 维度使用）
        self._prev_hv_vx  = self._hv_vx
        self._prev_hv_vy  = self._hv_vy
        self._prev_hv_yaw = self._hv_yaw

        # 推进数据回放指针（AV / eHMI 随时间步更新）
        self.t = min(self.t + 1, self.max_t)

        if self.playback:
            # 回放模式：HV 状态直接读取真实数据，排除积分误差
            row = self.current_df.iloc[self.t]
            self._hv_x = float(row["HV_X"])
            self._hv_y = float(row["HV_Y"])
            self._hv_vx = float(row["HV_vx"])
            self._hv_vy = float(row["HV_vy"])
            self._hv_yaw = float(row["HV_yaw"])
        else:
            ax = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
            ay = float(np.clip(action[1], self.action_space.low[1], self.action_space.high[1]))

            # HV 动力学：Euler 积分（全局坐标系）
            self._hv_vx += ax * self.DT
            self._hv_vy += ay * self.DT
            self._hv_x += self._hv_vx * self.DT
            self._hv_y += self._hv_vy * self.DT
            self._hv_yaw = float(np.arctan2(self._hv_vy, self._hv_vx + 1e-6))

        obs        = self._build_obs()
        reward     = 0.0          # GAIL 会用判别器奖励覆盖此值
        terminated = self._is_done()
        truncated  = self.t >= self.max_t

        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    def _build_obs(self) -> np.ndarray:
        """构建与专家数据结构完全一致的 16 维观测向量。"""
        row = self.current_df.iloc[self.t]

        # 到目标方向和距离
        dx    = float(self.current_goal[0]) - self._hv_x
        dy    = float(self.current_goal[1]) - self._hv_y
        d_des = float(np.sqrt(dx ** 2 + dy ** 2))

        # AV (ego) 从数据回放，计算相对 HV 的位移和速度差
        av_x  = float(row["ego_X"])
        av_y  = float(row["ego_Y"])
        av_vx = float(row["ego_vx"])
        av_vy = float(row["ego_vy"])

        delta_px = av_x  - self._hv_x
        delta_py = av_y  - self._hv_y
        delta_vx = av_vx - self._hv_vx
        delta_vy = av_vy - self._hv_vy

        obs = np.array([
            self._hv_x,                   # 0  HV_X
            self._hv_y,                   # 1  HV_Y
            self._hv_vx,                  # 2  HV_vx
            self._hv_vy,                  # 3  HV_vy
            dx,                           # 4  dx  (目标 - 当前位置)
            dy,                           # 5  dy
            self._prev_hv_yaw,            # 6  HV_yaw_{t-1}
            d_des,                        # 7  d_des
            self._prev_hv_vx,             # 8  HV_vx_{t-1}
            self._prev_hv_vy,             # 9  HV_vy_{t-1}
            float(row["eHMI_state"]),     # 10 eHMI_state
            float(row["eHMI_duration"]),  # 11 eHMI_duration
            delta_px,                     # 12 Δpx (ego - HV)
            delta_py,                     # 13 Δpy
            delta_vx,                     # 14 Δvx
            delta_vy,                     # 15 Δvy
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    def _is_done(self) -> bool:
        """提前终止条件。"""
        # 1. 到达目标点
        dx = float(self.current_goal[0]) - self._hv_x
        dy = float(self.current_goal[1]) - self._hv_y
        if np.sqrt(dx ** 2 + dy ** 2) < self.goal_radius:
            return True

        # 2. HV 偏离参考轨迹过远（防止策略随机游走）
        row   = self.current_df.iloc[self.t]
        ref_x = float(row["HV_X"])
        ref_y = float(row["HV_Y"])
        if np.sqrt((self._hv_x - ref_x) ** 2 + (self._hv_y - ref_y) ** 2) > self.max_dev:
            return True

        return False