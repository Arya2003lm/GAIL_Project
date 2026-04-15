# =============================================================
# data_loader.py  —  基于真实 .asc 数据的观测构建
# =============================================================

import numpy as np
import pandas as pd
from pathlib import Path
from gymnasium import spaces
from imitation.data.types import Trajectory

# =============================================================
# 0. 全局配置
# =============================================================

# 目标点：默认不写死，每条轨迹在截断后使用最后一帧 HV 位置
# 如需固定目标点，可在 load_trajectory/load_all_trajectories 传入 goal

# 动作：[HV_ax, HV_ay]
# 使用全局坐标系下的加速度分量作为动作监督目标
ACT_COLS = ["HV_ax", "HV_ay"]
ACT_CLIP_Q = (0.01, 0.99)  # 按分位数裁剪动作（1%~99%）

# 观测维度：10（自身）+ 1辆背景车×4 + eHMI×2 = 16
OBS_DIM = 16

# 观测空间上下界（根据数据实测值 + 余量）
obs_low = np.array([
    -30,  -10,   # HV_X,  HV_Y        （位置，起点附近）
    -4,   -1,    # HV_vx, HV_vy       （速度分量，m/s）
    -10,  -1,   # dx, dy             （到目标的距离分量）
    -np.pi,      # HV_yaw_{t-1}       （上一步航向角）
    0,           # d_des              （到目标标量距离）
    -4,   -1,    # HV_vx_{t-1}, HV_vy_{t-1}
    0,    0,     # eHMI_state, eHMI_duration
    # 背景车相对量
    -50,   -20,    # Δpx, Δpy           （ego相对HV的位置差）
    -20,  -15,   # Δvx, Δvy           （ego相对HV的速度差）
], dtype=np.float32)

obs_high = np.array([
    22,   40,    # HV_X,  HV_Y
    11,   15,    # HV_vx, HV_vy
    51,   25,    # dx, dy
    np.pi,
    60,          # d_des
    11,   15,    # HV_vx_{t-1}, HV_vy_{t-1}
    2, 20,   # eHMI_state(0/1/2), eHMI_duration(仿真时间，单位 s)
    # 背景车相对量
    102,   6,   # Δpx, Δpy
    5,     1,    # Δvx, Δvy
], dtype=np.float32)

observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

# 动作空间
action_space = spaces.Box(
    low=np.array([-6.0,  -2.0], dtype=np.float32),   # [min_HV_ax, min_HV_ay]
    high=np.array([ 3.0,   2.0], dtype=np.float32),  # [max_HV_ax, max_HV_ay]
)


# =============================================================
# 1. 截断函数
# =============================================================

def truncate(df: pd.DataFrame) -> pd.DataFrame:
    """
    截取真正的交互段：
      起点：HV_X 第一次 > -30
      终点：HV_Y 第一次 > 35
    """
    start_mask = df["HV_X"] > -30
    if not start_mask.any():
        return pd.DataFrame()   # 整条轨迹都不满足，跳过
    start_idx = start_mask.idxmax()

    end_mask = df["HV_Y"] > 35
    if not end_mask.any():
        end_idx = len(df) - 1   # 没到终点就用最后一行
    else:
        end_idx = end_mask.idxmax()

    return df.iloc[start_idx : end_idx + 1].reset_index(drop=True)


# =============================================================
# 2. 派生列计算
# =============================================================

def compute_derived(df: pd.DataFrame, goal: np.ndarray) -> pd.DataFrame:
    """把原始列转换为观测所需的派生量"""
    df = df.copy()

    # HV 速度：km/h → m/s，再分解为 vx, vy
    df["HV_v_ms"] = df["HV_v_km/h"] / 3.6
    df["HV_vx"]   = df["HV_v_ms"] * np.cos(df["HV_yaw"])
    df["HV_vy"]   = df["HV_v_ms"] * np.sin(df["HV_yaw"])

    # ego 速度分解（背景车）
    df["ego_vx"] = df["ego_speed"] * np.cos(df["ego_yaw"])
    df["ego_vy"] = df["ego_speed"] * np.sin(df["ego_yaw"])

    # HV 加速度：从全局坐标系转换为车体坐标系
    # HV_ax, HV_ay 是全局坐标系下的加速度分量
    # 纵向加速度（longitudinal）= 沿车头方向的加速度
    # 横向加速度（lateral）= 垂直于车头方向的加速度
    df["HV_a_long"] = df["HV_ax"] * np.cos(df["HV_yaw"]) + df["HV_ay"] * np.sin(df["HV_yaw"])
    df["HV_a_lat"]  = -df["HV_ax"] * np.sin(df["HV_yaw"]) + df["HV_ay"] * np.cos(df["HV_yaw"])

    # 仿真时间定义： (MeasurementTime - Measurement time error) / 1000
    if "MeasurementTime" in df.columns and "Measurement time error" in df.columns:
        mt = pd.to_numeric(df["MeasurementTime"], errors="coerce").to_numpy(dtype=np.float64)
        mte = pd.to_numeric(df["Measurement time error"], errors="coerce").to_numpy(dtype=np.float64)
        t = (mt - mte) / 1000.0
    elif "MeasurementTime" in df.columns and "MeasurementTimeError" in df.columns:
        mt = pd.to_numeric(df["MeasurementTime"], errors="coerce").to_numpy(dtype=np.float64)
        mte = pd.to_numeric(df["MeasurementTimeError"], errors="coerce").to_numpy(dtype=np.float64)
        t = (mt - mte) / 1000.0
    elif "MeasurementTime" in df.columns:
        t = pd.to_numeric(df["MeasurementTime"], errors="coerce").to_numpy(dtype=np.float64) / 1000.0
    elif "time_cost[s]" in df.columns:
        t = pd.to_numeric(df["time_cost[s]"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        t = np.arange(len(df), dtype=np.float64)

    # eHMI 信号：不显示=0，eHMI_green=1，eHMI_red=2
    green = df["eHMI_green_on"] if "eHMI_green_on" in df.columns else pd.Series(0, index=df.index)
    red = df["eHMI_red_on"] if "eHMI_red_on" in df.columns else pd.Series(0, index=df.index)
    df["eHMI_state"] = np.select(
        [red > 0.5, green > 0.5],
        [2, 1],
        default=0,
    ).astype(np.float32)

    # eHMI 持续时间：从 0 -> 非0 切换开始计时；回到 0 时清零

    states = df["eHMI_state"].to_numpy(dtype=np.int32)
    durations = np.zeros(len(df), dtype=np.float32)
    active_start = None
    prev_state = 0
    for i, (s, ti) in enumerate(zip(states, t)):
        if s == 0:
            durations[i] = 0.0
            active_start = None
        else:
            if prev_state == 0:
                active_start = ti
            if active_start is None:
                active_start = ti
            durations[i] = float(max(ti - active_start, 0.0))
        prev_state = s
    df["eHMI_duration"] = durations

    # 到目标点的距离分量和标量
    df["dx"]    = goal[0] - df["HV_X"]
    df["dy"]    = goal[1] - df["HV_Y"]
    df["d_des"] = np.sqrt(df["dx"]**2 + df["dy"]**2)

    # 背景车相对 HV 的位置和速度差（Δlat = ego - HV）
    df["delta_px"] = df["ego_X"]  - df["HV_X"]
    df["delta_py"] = df["ego_Y"]  - df["HV_Y"]
    df["delta_vx"] = df["ego_vx"] - df["HV_vx"]
    df["delta_vy"] = df["ego_vy"] - df["HV_vy"]

    return df


# =============================================================
# 3. 观测构建：单行 → obs 向量
# =============================================================

def build_obs(row: pd.Series, prev_row: pd.Series) -> np.ndarray:
    """
        自身状态 (12维):
      HV_X, HV_Y                          — 当前位置
      HV_vx, HV_vy                        — 当前速度分量
      dx, dy                              — 到目标距离分量
      HV_yaw_{t-1}                        — 上一步航向角
      d_des                               — 到目标标量距离
      HV_vx_{t-1}, HV_vy_{t-1}           — 上一步速度分量
            eHMI_state, eHMI_duration          — eHMI 信号与持续时间

    背景车相对观测 (4维):
      delta_px, delta_py                  — ego相对HV位置差
      delta_vx, delta_vy                  — ego相对HV速度差
    """
    ego_obs = np.array([
        row["HV_X"],
        row["HV_Y"],
        row["HV_vx"],
        row["HV_vy"],
        row["dx"],
        row["dy"],
        prev_row["HV_yaw"],        # θ_{t-1}
        row["d_des"],
        prev_row["HV_vx"],         # vx_{t-1}
        prev_row["HV_vy"],         # vy_{t-1}
        row["eHMI_state"],
        row["eHMI_duration"],
    ], dtype=np.float32)

    surr_obs = np.array([
        row["delta_px"],
        row["delta_py"],
        row["delta_vx"],
        row["delta_vy"],
    ], dtype=np.float32)

    return np.concatenate([ego_obs, surr_obs])  # shape (16,)


# =============================================================
# 4. 单个 .asc 文件 → Trajectory
# =============================================================

def load_trajectory(asc_path: str, goal: np.ndarray | None = None) -> Trajectory | None:
    # 读取
    df = pd.read_csv(asc_path)

    # 截断
    df = truncate(df)
    if len(df) < 5:
        print(f"  [跳过] {Path(asc_path).name}：截断后长度 {len(df)} < 5")
        return None

    # 默认目标点：截断后最后一行的 HV 位置
    if goal is None:
        goal = np.array([df["HV_X"].iloc[-1], df["HV_Y"].iloc[-1]], dtype=np.float32)

    # 计算派生列
    df = compute_derived(df, goal)

    # 逐行构建观测
    obs_list = []
    for t in range(len(df)):
        prev = df.iloc[t - 1] if t > 0 else df.iloc[0]
        obs_list.append(build_obs(df.iloc[t], prev))

    obs_arr = np.array(obs_list, dtype=np.float32)           # (T, 16)
    act_arr = df[ACT_COLS].to_numpy(dtype=np.float32)        # (T, 2)

    # imitation 要求 obs (T+1)，末尾补最后一帧作为 terminal next_obs
    obs_with_next = np.vstack([obs_arr, obs_arr[[-1]]])      # (T+1, 16)

    return Trajectory(
        obs=obs_with_next,
        acts=act_arr,
        infos=np.array([{}] * len(act_arr)),
        terminal=True,
    )


def clip_trajectories_actions_by_quantile(
    trajs: list[Trajectory],
    low_q: float = ACT_CLIP_Q[0],
    high_q: float = ACT_CLIP_Q[1],
) -> tuple[list[Trajectory], np.ndarray, np.ndarray]:
    """按全数据分位数裁剪动作，返回裁剪后的轨迹和上下界。"""
    if not trajs:
        return trajs, np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

    all_acts = np.vstack([t.acts for t in trajs])
    clip_low = np.quantile(all_acts, low_q, axis=0).astype(np.float32)
    clip_high = np.quantile(all_acts, high_q, axis=0).astype(np.float32)

    clipped_trajs: list[Trajectory] = []
    for t in trajs:
        clipped_acts = np.clip(t.acts, clip_low, clip_high).astype(np.float32)
        clipped_trajs.append(
            Trajectory(
                obs=t.obs,
                acts=clipped_acts,
                infos=t.infos,
                terminal=t.terminal,
            )
        )

    return clipped_trajs, clip_low, clip_high


# =============================================================
# 5. 批量加载
# =============================================================

def load_all_trajectories(data_dir: str, goal: np.ndarray | None = None) -> list[Trajectory]:
    files = sorted(Path(data_dir).glob("*.asc"))
    print(f"找到 {len(files)} 个 .asc 文件")

    trajs = []
    for f in files:
        traj = load_trajectory(str(f), goal)
        if traj is not None:
            trajs.append(traj)

    # 按全数据分位数裁剪动作，减小离群值影响
    trajs, clip_low, clip_high = clip_trajectories_actions_by_quantile(trajs)
    print(
        "动作分位数裁剪: "
        f"HV_ax[{clip_low[0]:.3f}, {clip_high[0]:.3f}], "
        f"HV_ay[{clip_low[1]:.3f}, {clip_high[1]:.3f}]"
    )

    lengths = [len(t.acts) for t in trajs]
    print(f"成功加载：{len(trajs)} 条轨迹")
    print(f"轨迹长度：min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
    return trajs


# =============================================================
# 6. 快速验证（加载后调用一次）
# =============================================================

def validate(trajs: list[Trajectory]):
    all_obs  = np.vstack([t.obs  for t in trajs])
    all_acts = np.vstack([t.acts for t in trajs])

    obs_names = [
        "HV_X", "HV_Y", "HV_vx", "HV_vy",
        "dx", "dy", "yaw_prev", "d_des", "vx_prev", "vy_prev",
        "eHMI_state", "eHMI_duration",
        "Δpx(ego-HV)", "Δpy(ego-HV)", "Δvx(ego-HV)", "Δvy(ego-HV)",
    ]
    act_names = ["HV_ax", "HV_ay"]

    print("\n=== 观测统计 ===")
    df_obs = pd.DataFrame(all_obs, columns=obs_names)
    print(df_obs.describe().T[["min", "max", "mean", "std"]].round(3))

    print("\n=== 动作统计 ===")
    df_act = pd.DataFrame(all_acts, columns=act_names)
    print(df_act.describe().T[["min", "max", "mean", "std"]].round(3))

    # 检查是否超出定义的 obs space
    out_low  = (all_obs < obs_low).any(axis=0)
    out_high = (all_obs > obs_high).any(axis=0)
    bad = np.where(out_low | out_high)[0]
    if len(bad):
        print(f"\n[警告] 以下维度超出 obs_space，需调整边界: "
              f"{[obs_names[i] for i in bad]}")
    else:
        print("\n所有观测值在 obs_space 范围内 ✓")


# =============================================================
# 使用示例
# =============================================================
if __name__ == "__main__":
    # 单文件测试
    traj = load_trajectory("data/left_turn/exper8_100_54_-25_0.asc")
    if traj:
        print(f"\n单文件加载成功：obs {traj.obs.shape}, acts {traj.acts.shape}")
        validate([traj])

    # 批量加载（替换为你的目录）
    trajs = load_all_trajectories("./data/left_turn/")
    validate(trajs)