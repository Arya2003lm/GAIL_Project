# =============================================================
# plot_trajs.py  —  绘制所有实验数据的 HV 轨迹
# =============================================================

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

try:
    from Convert_expert_data import load_all_with_dataframes
except ModuleNotFoundError:
    from scripts.Convert_expert_data import load_all_with_dataframes

DATA_DIR = "./data/left_turn/"
SAVE_PATH = "./evaluate/all_trajs.png"
TOP10_SAVE_PATH = "./evaluate/top10_trajs_hvy20_maxx.png"
TARGET_HV_Y = 20.0
HV_Y_TOL = 1.0
TOP_K = 10


def print_top_trajs_near_hv_y(dfs, data_files, target_hv_y=TARGET_HV_Y, tol=HV_Y_TOL, top_k=TOP_K):
    """打印在 HV_Y≈target_hv_y 区域内 HV_X 最大的若干条轨迹。"""
    rows = []

    for i, ((df, _), file_path) in enumerate(zip(dfs, data_files)):
        mask = np.abs(df["HV_Y"].to_numpy(dtype=np.float32) - target_hv_y) <= tol
        if not np.any(mask):
            continue

        sub_df = df.loc[mask, ["HV_X", "HV_Y"]]
        best_idx = sub_df["HV_X"].idxmax()
        best_x = float(df.loc[best_idx, "HV_X"])
        best_y = float(df.loc[best_idx, "HV_Y"])
        rows.append({
            "rank_key": best_x,
            "traj_idx": i,
            "file": Path(file_path).name,
            "HV_X_at_target": best_x,
            "HV_Y_at_target": best_y,
            "traj_len": int(len(df)),
        })

    rows.sort(key=lambda x: x["rank_key"], reverse=True)
    top_rows = rows[:top_k]

    print(
        f"\nHV_Y≈{target_hv_y:.1f}±{tol:.1f} 区域内，HV_X 最大的前 {min(top_k, len(top_rows))} 条轨迹:"
    )
    if not top_rows:
        print("  没有轨迹经过该 HV_Y 区域。")
        return []

    for rank, row in enumerate(top_rows, start=1):
        print(
            f"  #{rank:02d} traj_idx={row['traj_idx']:3d} | "
            f"HV_X={row['HV_X_at_target']:.3f}, HV_Y={row['HV_Y_at_target']:.3f} | "
            f"len={row['traj_len']:3d} | file={row['file']}"
        )
    return top_rows


def plot_selected_trajs(dfs, selected_rows, save_path: str):
    """将筛选出的轨迹单独绘制成一张图。"""
    if not selected_rows:
        print("未生成 Top10 轨迹图：没有满足条件的轨迹。")
        return

    colors = cm.tab10(np.linspace(0, 1, len(selected_rows)))
    fig, ax = plt.subplots(figsize=(10, 9))

    for color, row in zip(colors, selected_rows):
        traj_idx = row["traj_idx"]
        df, goal = dfs[traj_idx]
        xs = df["HV_X"].to_numpy(dtype=np.float32)
        ys = df["HV_Y"].to_numpy(dtype=np.float32)

        ax.plot(xs, ys, color=color, lw=1.8, alpha=0.9, label=f"#{traj_idx}  X@Y20={row['HV_X_at_target']:.2f}")
        ax.scatter(xs[0], ys[0], color=color, s=24, zorder=3)
        ax.scatter(goal[0], goal[1], color=color, marker="x", s=36, zorder=3)

    ax.set_xlabel("HV_X (m)", fontsize=12)
    ax.set_ylabel("HV_Y (m)", fontsize=12)
    ax.set_title(
        f"Top {len(selected_rows)} Trajectories by HV_X near HV_Y≈{TARGET_HV_Y:.1f}±{HV_Y_TOL:.1f}",
        fontsize=13,
    )
    ax.set_aspect("equal", "datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"已保存 Top10 轨迹图 → {out}")


def main():
    print("加载数据 ...")
    trajs, dfs = load_all_with_dataframes(DATA_DIR)
    data_files = sorted(Path(DATA_DIR).glob("*.asc"))
    n = len(dfs)
    print(f"共 {n} 条轨迹")
    top_rows = print_top_trajs_near_hv_y(dfs, data_files)

    colors = cm.tab20(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(10, 9))

    for i, (df, goal) in enumerate(dfs):
        xs = df["HV_X"].to_numpy(dtype=np.float32)
        ys = df["HV_Y"].to_numpy(dtype=np.float32)
        ax.plot(xs, ys, color=colors[i], lw=1.0, alpha=0.7)
        # 起点圆点
        ax.scatter(xs[0], ys[0], color=colors[i], s=20, zorder=3)
        # 终点/目标用叉
        ax.scatter(goal[0], goal[1], color=colors[i], marker="x", s=30, zorder=3)

    ax.set_xlabel("HV_X (m)", fontsize=12)
    ax.set_ylabel("HV_Y (m)", fontsize=12)
    ax.set_title(f"All HV Trajectories  (n={n})", fontsize=13)
    ax.set_aspect("equal", "datalim")
    ax.grid(True, alpha=0.3)

    # 图例说明
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", lw=1.2, label="HV trajectory"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="Start point"),
        Line2D([0], [0], marker="x", color="gray", markersize=7, label="Goal point"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

    plt.tight_layout()

    out = Path(SAVE_PATH)
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"已保存 → {out}")

    plot_selected_trajs(dfs, top_rows, TOP10_SAVE_PATH)
    plt.show()


if __name__ == "__main__":
    main()
