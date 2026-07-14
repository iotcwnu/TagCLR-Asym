import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# --- 配置 ---
PROCESSED_DATA_PATH = 'data/final_dataset_corrected.npz'
STATS_SAVE_PATH = 'data/norm_stats_corrected.npz'

TEST_SPLIT_RATIO = 0.2
FIXED_SEED = 42  # 必须与划分测试集时使用的种子一致


def run_stats_computation_per_tag():
    print("================================")
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: {PROCESSED_DATA_PATH} not found.")
        return

    data = np.load(PROCESSED_DATA_PATH, allow_pickle=True)
    rssi_matrices = data['rssi_matrices']
    phase_matrices = data['phase_matrices']
    doppler_matrices = data['doppler_matrices']
    mask_matrices = data['mask_matrices']

    # 加载标签用于分层抽样
    labels = data['labels']
    tag_ids = data['tag_ids']
    num_tags = len(tag_ids)

    indices = np.arange(len(labels))


    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SPLIT_RATIO,
        random_state=FIXED_SEED,  # 锁死种子
        stratify=labels  # 锁死分层逻辑
    )

    train_rssi = rssi_matrices[train_val_indices]
    train_phase = phase_matrices[train_val_indices]
    train_doppler = doppler_matrices[train_val_indices]
    train_mask = mask_matrices[train_val_indices]

    print(f"总样本数: {len(labels)}")
    print(f"用于统计的样本数 (Train+Val): {len(train_val_indices)} (排除测试集)")

    rssi_means = np.zeros(num_tags);
    rssi_stds = np.zeros(num_tags)
    phase_means = np.zeros(num_tags);
    phase_stds = np.zeros(num_tags)
    doppler_means = np.zeros(num_tags);
    doppler_stds = np.zeros(num_tags)

    print("正在计算每个标签的统计量...")
    for i in tqdm(range(num_tags)):
        mask_tag = train_mask[:, i, :]

        valid = train_rssi[:, i, :][mask_tag > 0]
        if valid.size > 1:
            rssi_means[i] = np.mean(valid)
            rssi_stds[i] = np.std(valid)

        valid = train_phase[:, i, :][mask_tag > 0]
        if valid.size > 1:
            phase_means[i] = np.mean(valid)
            phase_stds[i] = np.std(valid)

        valid = train_doppler[:, i, :][mask_tag > 0]
        if valid.size > 1:
            doppler_means[i] = np.mean(valid)
            doppler_stds[i] = np.std(valid)

    np.savez(
        STATS_SAVE_PATH,
        rssi_means=rssi_means, rssi_stds=rssi_stds,
        phase_means=phase_means, phase_stds=phase_stds,
        doppler_means=doppler_means, doppler_stds=doppler_stds
    )
    print(f"统计数据已保存至: {STATS_SAVE_PATH}")


if __name__ == '__main__':
    run_stats_computation_per_tag()