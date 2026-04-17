# =============================================================
# relevant_traj.py  —  为每条轨迹生成左转参考轨迹
# =============================================================
# 参考轨迹由两段构成：
#   1. 直线段：从轨迹起点 → 转弯点 TURN_POINT
#   2. 圆弧段：从 TURN_POINT → 轨迹终点（goal）
#      （不再使用样条/Bezier）
# =============================================================
# 用法：
#   python scripts/relevant_traj.py               # 显示所有轨迹的参考轨迹叠加图
#   python scripts/relevant_traj.py --save        # 保存到 evaluate/ref_trajs.png
#   python scripts/relevant_traj.py --ep 3 7 12   # 只画指定 episode
# =============================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from Convert_expert_data import load_all_with_dataframes
except ModuleNotFoundError:
    from scripts.Convert_expert_data import load_all_with_dataframes

# ---------- 全局常量 ----------
DATA_DIR    = "./data/left_turn/"
TURN_POINT  = np.array([-4, 14.54], dtype=np.float64)   # 直线→弧线 分界点
N_PTS_LINE  = 100    # 直线段采样点数
N_PTS_ARC   = 150    # 圆弧段采样点数
# 兼容旧接口保留（圆弧方案中不使用）
CTRL_ARM_RATIO = 0.4


# =============================================================
# 核心函数：生成单条参考轨迹
# =============================================================

def make_reference_traj(
    start: np.ndarray,
    goal: np.ndarray,
    turn_point: np.ndarray = TURN_POINT,
    n_line: int = N_PTS_LINE,
    n_arc: int = N_PTS_ARC,
    ctrl_arm_ratio: float = CTRL_ARM_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成从 start 经 turn_point 到 goal 的左转参考轨迹。

    参数
    ----
    start       : 轨迹起点 [x, y]
    goal        : 轨迹终点 [x, y]
    turn_point  : 直线与弧线的分界点，默认 (-4.3, 14.54)
    n_line      : 直线段点数
    n_arc       : 圆弧段点数
    ctrl_arm_ratio : 兼容旧接口参数（圆弧方案中不使用）

    返回
    ----
    xs, ys : 参考轨迹的 X / Y 坐标数组（已合并两段，去除重复连接点）
    """
    start      = np.asarray(start,      dtype=np.float64)
    goal       = np.asarray(goal,       dtype=np.float64)
    turn_point = np.asarray(turn_point, dtype=np.float64)

    # ---- 直线段：start → turn_point ----
    t_line = np.linspace(0, 1, n_line)
    line_x = start[0] + t_line * (turn_point[0] - start[0])
    line_y = start[1] + t_line * (turn_point[1] - start[1])

    # ---- 圆弧段：turn_point → goal ----
    # 约束：终点切线与“终点后方车道”一致（+Y 方向）。
    # 对圆而言，切线与半径垂直；若终点切线为 +Y，则终点半径须水平，
    # 因而圆心位于 y = goal_y 这条水平线上。
    x0, y0 = float(turn_point[0]), float(turn_point[1])
    x1, y1 = float(goal[0]), float(goal[1])

    denom = 2.0 * (x1 - x0)
    num = (x1 * x1) - (x0 * x0) - ((y0 - y1) * (y0 - y1))

    if abs(denom) < 1e-9:
        # 几何退化：无法同时满足“过两点 + 终点切线竖直”。退化为直线连接避免崩溃。
        arc_x = np.linspace(x0, x1, n_arc)
        arc_y = np.linspace(y0, y1, n_arc)
    else:
        cx = num / denom
        cy = y1
        r = np.hypot(x1 - cx, y1 - cy)

        if r < 1e-9:
            arc_x = np.linspace(x0, x1, n_arc)
            arc_y = np.linspace(y0, y1, n_arc)
        else:
            th0 = np.arctan2(y0 - cy, x0 - cx)
            th1 = np.arctan2(y1 - cy, x1 - cx)

            # 在终点选择使切线更接近 +Y 的旋转方向
            t_ccw = np.array([-np.sin(th1), np.cos(th1)])
            t_cw = -t_ccw
            use_ccw = float(np.dot(t_ccw, np.array([0.0, 1.0]))) >= float(np.dot(t_cw, np.array([0.0, 1.0])))

            if use_ccw:
                dtheta = th1 - th0
                while dtheta <= 0:
                    dtheta += 2.0 * np.pi
            else:
                dtheta = th1 - th0
                while dtheta >= 0:
                    dtheta -= 2.0 * np.pi

            theta = th0 + np.linspace(0.0, dtheta, n_arc)
            arc_x = cx + r * np.cos(theta)
            arc_y = cy + r * np.sin(theta)

    # 合并（去掉直线末点，与弧线首点重复）
    xs = np.concatenate([line_x[:-1], arc_x])
    ys = np.concatenate([line_y[:-1], arc_y])
    return xs, ys


# =============================================================
# 批量生成所有轨迹的参考轨迹
# =============================================================

def make_all_reference_trajs(dfs: list) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    对 dfs 中每条轨迹生成参考轨迹。

    返回
    ----
    list of (xs, ys)，与 dfs 等长
    """
    ref_trajs = []
    for df, goal in dfs:
        start = np.array([df["HV_X"].iloc[0], df["HV_Y"].iloc[0]], dtype=np.float64)
        goal_arr = np.asarray(goal, dtype=np.float64)
        xs, ys = make_reference_traj(start, goal_arr)
        ref_trajs.append((xs, ys))
    return ref_trajs


# =============================================================
# 可视化
# =============================================================

def plot_reference_trajs(
    dfs: list,
    ep_indices: list[int] | None = None,
    save_path: str | None = None,
    show_expert: bool = True,
):
    """
    将各轨迹的专家轨迹（细线）和参考轨迹（粗虚线）叠加绘制。

    参数
    ----
    dfs         : load_all_with_dataframes 返回的 dfs
    ep_indices  : 要画的 episode 下标列表；None 表示全部
    save_path   : 若非 None，保存图片到此路径
    show_expert : 是否同时绘制专家轨迹
    """
    if ep_indices is None:
        ep_indices = list(range(len(dfs)))

    colors = cm.tab20(np.linspace(0, 1, len(ep_indices)))
    fig, ax = plt.subplots(figsize=(10, 9))

    for color, ep_idx in zip(colors, ep_indices):
        df, goal = dfs[ep_idx]
        start = np.array([df["HV_X"].iloc[0], df["HV_Y"].iloc[0]])
        goal_arr = np.asarray(goal, dtype=np.float64)

        ref_xs, ref_ys = make_reference_traj(start, goal_arr)

        if show_expert:
            ax.plot(
                df["HV_X"].to_numpy(), df["HV_Y"].to_numpy(),
                color=color, lw=0.9, alpha=0.45, zorder=2,
            )
        ax.plot(
            ref_xs, ref_ys,
            color=color, lw=2.0, ls="--", alpha=0.85, zorder=3,
        )

    # 标注转弯点
    ax.scatter(
        [TURN_POINT[0]], [TURN_POINT[1]],
        marker="D", s=120, color="crimson", zorder=10,
        label=f"Turn point ({TURN_POINT[0]}, {TURN_POINT[1]})",
    )

    # 图例代理
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="steelblue", lw=1.0, alpha=0.5,
               label="Expert trajectory"),
        Line2D([0], [0], color="steelblue", lw=2.0, ls="--",
               label="Reference trajectory"),
    ]
    if show_expert:
        ax.legend(handles=legend_handles, fontsize=10)
    else:
        ax.legend(handles=[legend_handles[1]], fontsize=10)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(
        f"Reference trajectories  ({len(ep_indices)} episodes)\n"
        f"Straight: start → {tuple(TURN_POINT.tolist())}  |  Arc: turn → goal (circular arc)",
        fontsize=11,
    )
    ax.set_aspect("equal", "datalim")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[relevant_traj] 已保存 → {save_path}")
    else:
        plt.show()

    return fig


# =============================================================
# 主入口
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize reference trajectories")
    parser.add_argument("--ep", type=int, nargs="+", default=None,
                        help="Episode indices to show (default: all)")
    parser.add_argument("--save", action="store_true",
                        help="Save figure to evaluate/ref_trajs.png")
    parser.add_argument("--no-expert", action="store_true",
                        help="Hide expert trajectories, show only reference")
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    print("Loading data ...")
    trajs, dfs = load_all_with_dataframes(args.data_dir)
    print(f"Loaded {len(dfs)} trajectories")

    save_path = "./evaluate/ref_trajs.png" if args.save else None
    plot_reference_trajs(
        dfs,
        ep_indices=args.ep,
        save_path=save_path,
        show_expert=not args.no_expert,
    )