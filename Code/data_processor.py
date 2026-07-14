import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DATA_DIR = '../../data/RFID'
PROCESSED_DATA_PATH = 'data/final_dataset_corrected.npz'
SAMPLING_RATE = 25
EXPECTED_DURATION = 4.0
MIN_READS_THRESHOLD = 1
DEBUG_MODE = False


def get_all_tag_ids(root_dir):
    print("--- 预扫描所有文件以获取全局标签ID列表 ---")
    all_tags = set()
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                file_path = os.path.join(dirpath, filename)
                try:
                    df_tags = pd.read_csv(file_path, header=None, sep='\s+', usecols=[1],
                                          names=['tag_id'], engine='python')
                    all_tags.update(df_tags['tag_id'].unique())
                except:
                    pass
    sorted_tags = sorted(list(all_tags))
    tag_id_to_idx_map = {tag: i for i, tag in enumerate(sorted_tags)}
    print(f"扫描完成. 共发现 {len(sorted_tags)} 个唯一标签.")
    return sorted_tags, tag_id_to_idx_map


def parse_filename_for_action_id(filename):
    match = re.search(r'rfid_(\d+)\(', filename)
    if match: return int(match.group(1))
    return None


def process_single_file(file_path, all_tag_ids, tag_id_to_idx_map, params):
    try:
        col_names = ['reader_id', 'tag_id', 'timestamp', 'doppler', 'rssi', 'phase_raw', 'old_label']
        df = pd.read_csv(file_path, header=None, sep='\s+', names=col_names, engine='python')
    except Exception:
        col_names = ['reader_id', 'tag_id', 'timestamp', 'doppler', 'rssi', 'phase_raw']
        try:
            df = pd.read_csv(file_path, header=None, sep='\s+', names=col_names, engine='python', usecols=range(6))
        except Exception:
            return None

    if df.empty or len(df) < params['threshold']: return None

    sample_time = df['timestamp'].iloc[0]
    if sample_time > 1e14:
        df['timestamp'] = df['timestamp'] / 1000000.0
    else:
        df['timestamp'] = df['timestamp'] / 1000.0

    def unwrap_group(group):
        group = group.sort_values('timestamp')
        group['phase_unwrapped'] = np.unwrap(group['phase_raw'])
        return group

    try:
        df = df.groupby('tag_id').apply(unwrap_group, include_groups=False).reset_index()
    except TypeError:
        df = df.groupby('tag_id').apply(unwrap_group).reset_index()

    if 'level_1' in df.columns:
        df = df.drop(columns=['level_1'])

    start_time = df['timestamp'].min()
    end_time = start_time + params['duration']
    new_time_index = np.linspace(start_time, end_time, int(params['duration'] * params['rate']))

    num_global_tags = len(all_tag_ids)
    window_len = len(new_time_index)

    rssi_matrix = np.zeros((num_global_tags, window_len))
    phase_matrix = np.zeros((num_global_tags, window_len))
    doppler_matrix = np.zeros((num_global_tags, window_len))
    mask_matrix = np.zeros((num_global_tags, window_len))

    for tag_id in df['tag_id'].unique():
        if tag_id not in tag_id_to_idx_map: continue
        tag_data = df[df['tag_id'] == tag_id]
        if len(tag_data) < params['threshold']: continue

        row_idx = tag_id_to_idx_map[tag_id]

        rssi_interp = np.interp(new_time_index, tag_data['timestamp'], tag_data['rssi'])
        phase_interp = np.interp(new_time_index, tag_data['timestamp'], tag_data['phase_unwrapped'])
        doppler_interp = np.interp(new_time_index, tag_data['timestamp'], tag_data['doppler'])

        rssi_matrix[row_idx, :] = rssi_interp
        phase_matrix[row_idx, :] = phase_interp
        doppler_matrix[row_idx, :] = doppler_interp
        mask_matrix[row_idx, :] = 1.0

    return rssi_matrix, phase_matrix, mask_matrix, doppler_matrix


def run_processing():
    params = {'rate': SAMPLING_RATE, 'duration': EXPECTED_DURATION, 'threshold': MIN_READS_THRESHOLD}
    all_tag_ids, tag_id_to_idx_map = get_all_tag_ids(ROOT_DATA_DIR)
    if not all_tag_ids: return

    all_rssi, all_phase, all_doppler, all_mask = [], [], [], []
    all_labels, all_vols = [], []

    print("--- 开始处理文件 ---")
    file_paths = []
    for r, _, fs in os.walk(ROOT_DATA_DIR):
        for f in sorted(fs):
            if f.endswith('.txt'): file_paths.append(os.path.join(r, f))

    for fpath in tqdm(file_paths):
        filename = os.path.basename(fpath)
        vol_dir = os.path.basename(os.path.dirname(fpath))
        action_label = parse_filename_for_action_id(filename)

        if action_label is None: continue

        res = process_single_file(fpath, all_tag_ids, tag_id_to_idx_map, params)
        if res:
            r, p, m, d = res
            all_rssi.append(r)
            all_phase.append(p)
            all_doppler.append(d)
            all_mask.append(m)
            all_labels.append(action_label)
            all_vols.append(int(vol_dir))

    print("\n--- 保存数据 ---")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    np.savez_compressed(
        PROCESSED_DATA_PATH,
        rssi_matrices=np.array(all_rssi),
        phase_matrices=np.array(all_phase),
        doppler_matrices=np.array(all_doppler),
        mask_matrices=np.array(all_mask),
        labels=np.array(all_labels),
        volunteer_ids=np.array(all_vols),
        tag_ids=np.array(all_tag_ids, dtype=object)
    )
    print(f"完成! 数据已保存至: {PROCESSED_DATA_PATH}")


if __name__ == '__main__':
    run_processing()