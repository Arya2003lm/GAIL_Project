# =============================================================
# relevant_traj03.py  —  左转参考轨迹（turn 点相切版本）
# =============================================================
# 参考轨迹由两段构成：
#   1. 直线段：从轨迹起点 → 转弯点 TURN_POINT
#   2. 圆弧段：从 TURN_POINT → 轨迹终点（goal）
#
# 与 relevant_traj.py 的区别：
#   - 圆弧段要求与“前一段直线”在 TURN_POINT 处相切
#   - 不再要求在 goal 点与 +Y 方向相切
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
TURN_POINT  = np.array([-2, 14.54], dtype=np.float64)
N_PTS_LINE  = 100
N_PTS_ARC   = 150
CTRL_ARM_RATIO = 0.4  # 兼容旧接口保留


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)
    return v / n


def _build_tangent_arc_from_turn_to_goal(
    start: np.ndarray,
    turn_point: np.ndarray,
    goal: np.ndarray,
    n_arc: int,
) -> tuple[np.ndarray, np.ndarray]:
    """构造“在 turn_point 与入弯直线相切，并经过 goal”的圆弧。"""
    t = np.asarray(turn_point, dtype=np.float64)
    g = np.asarray(goal, dtype=np.float64)
    s = np.asarray(start, dtype=np.float64)

    # 入弯切向方向：start -> turn
    v_tan = _unit(t - s)
    # 法向（左法向）；若几何需要，lambda 可为负值，等价于在另一侧法向
    n = np.array([-v_tan[1], v_tan[0]], dtype=np.float64)

    d = g - t
    denom = 2.0 * float(np.dot(n, d))
    num = float(np.dot(d, d))

    # 几何退化：goal 接近切线方向，圆心退化到无穷远，使用直线兜底
    if abs(denom) < 1e-9:
        arc_x = np.linspace(t[0], g[0], n_arc)
        arc_y = np.linspace(t[1], g[1], n_arc)
        return arc_x, arc_y

    lam = num / denom
    c = t + lam * n
    r = float(np.linalg.norm(t - c))

    if r < 1e-9 or not np.isfinite(r):
        arc_x = np.linspace(t[0], g[0], n_arc)
        arc_y = np.linspace(t[1], g[1], n_arc)
        return arc_x, arc_y

    th0 = float(np.arctan2(t[1] - c[1], t[0] - c[0]))
    th1 = float(np.arctan2(g[1] - c[1], g[0] - c[0]))

    # 在 turn 点选择旋转方向，使圆弧切向与入弯方向最一致
    tan_ccw_t = np.array([-np.sin(th0), np.cos(th0)], dtype=np.float64)
    tan_cw_t = -tan_ccw_t
    use_ccw = float(np.dot(tan_ccw_t, v_tan)) >= float(np.dot(tan_cw_t, v_tan))

    if use_ccw:
        dtheta = th1 - th0
        while dtheta <= 0.0:
            dtheta += 2.0 * np.pi
    else:
        dtheta = th1 - th0
        while dtheta >= 0.0:
            dtheta -= 2.0 * np.pi

    theta = th0 + np.linspace(0.0, dtheta, n_arc)
    arc_x = c[0] + r * np.cos(theta)
    arc_y = c[1] + r * np.sin(theta)
    return arc_x, arc_y


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
    """生成从 start 经 turn_point 到 goal 的左转参考轨迹。"""
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    turn_point = np.asarray(turn_point, dtype=np.float64)

    # 1) 直线段：start -> turn_point
    t_line = np.linspace(0.0, 1.0, n_line)
    line_x = start[0] + t_line * (turn_point[0] - start[0])
    line_y = start[1] + t_line * (turn_point[1] - start[1])

    # 2) 圆弧段：turn_point -> goal，且在 turn_point 与直线相切
    arc_x, arc_y = _build_tangent_arc_from_turn_to_goal(
        start=start,
        turn_point=turn_point,
        goal=goal,
        n_arc=n_arc,
    )

    # 合并（去掉直线末点，与弧线首点重复）
    xs = np.concatenate([line_x[:-1], arc_x])
    ys = np.concatenate([line_y[:-1], arc_y])
    return xs, ys


# =============================================================
# 批量生成
# =============================================================
def make_all_reference_trajs(dfs: list) -> list[tuple[np.ndarray, np.ndarray]]:
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

    ax.scatter(
        [TURN_POINT[0]], [TURN_POINT[1]],
        marker="D", s=120, color="crimson", zorder=10,
        label=f"Turn point ({TURN_POINT[0]}, {TURN_POINT[1]})",
    )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="steelblue", lw=1.0, alpha=0.5, label="Expert trajectory"),
        Line2D([0], [0], color="steelblue", lw=2.0, ls="--", label="Reference trajectory (tangent at turn)"),
    ]
    if show_expert:
        ax.legend(handles=legend_handles, fontsize=10)
    else:
        ax.legend(handles=[legend_handles[1]], fontsize=10)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(
        f"Reference trajectories v03  ({len(ep_indices)} episodes)\n"
        f"Straight: start → {tuple(TURN_POINT.tolist())}  |  Arc: tangent at turn point only",
        fontsize=11,
    )
    ax.set_aspect("equal", "datalim")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[relevant_traj03] 已保存 → {save_path}")
    else:
        plt.show()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and visualize reference trajectories (v03 tangent-at-turn)")
    parser.add_argument("--ep", type=int, nargs="+", default=None, help="Episode indices to show (default: all)")
    parser.add_argument("--save", action="store_true", help="Save figure to evaluate/ref_trajs03.png")
    parser.add_argument("--no-expert", action="store_true", help="Hide expert trajectories, show only reference")
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    print("Loading data ...")
    trajs, dfs = load_all_with_dataframes(args.data_dir)
    print(f"Loaded {len(dfs)} trajectories")

    save_path = "./evaluate/ref_trajs03.png" if args.save else None
    plot_reference_trajs(
        dfs,
        ep_indices=args.ep,
        save_path=save_path,
        show_expert=not args.no_expert,
    )
