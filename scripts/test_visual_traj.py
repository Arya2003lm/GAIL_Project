from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    from visualize_trajectory import visualize_single, split_train_val
    from Convert_expert_data import load_all_with_dataframes
except ModuleNotFoundError:
    from scripts.visualize_trajectory import visualize_single, split_train_val
    from scripts.Convert_expert_data import load_all_with_dataframes


SAVE_DIR = Path("./checkpoints")
BC_PATH = SAVE_DIR / "bc_policy_hv_ax_ay"
GAIL_PATH = SAVE_DIR / "gail_policy_conservative"

def main():
    # 加载数据
    trajs, dfs = load_all_with_dataframes("./data/left_turn/")
    _, train_dfs, _, val_dfs = split_train_val(trajs, dfs)

    # 加载 BC / GAIL 策略来源
    bc_policy = ActorCriticPolicy.load(str(BC_PATH))
    gail_policy = ActorCriticPolicy.load(str(GAIL_PATH))

    # 画验证集第 3 条轨迹
    visualize_single(
        ep_idx=9,
        all_dfs=val_dfs,
        policies={"BC": bc_policy, "GAIL": gail_policy},
        show_reference=True,
    )
    plt.show()


if __name__ == "__main__":
    main()