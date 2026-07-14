import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class AugmentationMixin:
    def generate_random_state(self, num_timesteps):
        state = {}
        state['do_resize'] = np.random.rand() < 0.5
        if state['do_resize']:
            crop_len = int(num_timesteps * (0.8 + np.random.rand() * 0.2))
            start = np.random.randint(0, num_timesteps - crop_len + 1)
            state['resize_params'] = (start, crop_len)
        return state

    def apply_shared_transforms(self, matrix, state):
        if state.get('do_resize', False):
            start, crop_len = state['resize_params']
            T = matrix.shape[1]
            crop = matrix[:, start:start + crop_len]
            crop = crop.unsqueeze(0)
            matrix = F.interpolate(crop, size=T, mode='linear', align_corners=False).squeeze(0)

        return matrix

    def augment_rssi_pretrain(self, matrix):
        if np.random.rand() < 0.3:
            num_tags = matrix.shape[0]
            num_drop = np.random.randint(1, max(2, int(num_tags * 0.2)))
            mask_idx = np.random.choice(num_tags, num_drop, replace=False)
            matrix[mask_idx, :] = matrix[mask_idx, :] * 0.2
        if np.random.rand() < 0.3:
            matrix = matrix + torch.randn_like(matrix) * 0.01

        return matrix

    def augment_phase_pretrain(self, matrix):
        if np.random.rand() < 0.8:
            shift = (np.random.rand() - 0.5) * 1.0
            matrix = matrix + shift
        return matrix

    def augment_doppler_pretrain(self, matrix):
        if np.random.rand() < 0.3:
            matrix = matrix + torch.randn_like(matrix) * 0.01

        return matrix

    def augment_rssi_finetune(self, matrix):
        if np.random.rand() < 0.3:
            matrix = matrix + torch.randn(matrix.shape) * 0.01
        if np.random.rand() < 0.3:
            matrix = matrix * (1 + (np.random.rand() - 0.5) * 0.1)
        return matrix

    def augment_phase_finetune(self, matrix):

        if np.random.rand() < 0.3:
            shift = (np.random.rand() - 0.5) * 0.05
            matrix = matrix + shift

        return matrix

    def augment_doppler_finetune(self, matrix):

        if np.random.rand() < 0.3:
            scale = 1.0 + (np.random.rand() - 0.5) * 0.05
            matrix = matrix * scale

        if np.random.rand() < 0.2:
            matrix = matrix + torch.randn_like(matrix) * 0.003

        return matrix

    def get_norm_data(self, raw, mask, mod, stats):
        mean = torch.tensor(stats[f'{mod}_means'], dtype=torch.float32).unsqueeze(1)
        std = torch.tensor(stats[f'{mod}_stds'], dtype=torch.float32).unsqueeze(1)
        std[std == 0] = 1e-5
        return (torch.tensor(raw, dtype=torch.float32) - mean) / std * torch.tensor(mask, dtype=torch.float32)


class ContrastiveDataset(Dataset, AugmentationMixin):
    def __init__(self, npz_path, stats_path, modalities, is_train=True, indices=None):
        print(f"[Dataset] Loading for Contrastive Pre-training: {modalities}")
        with np.load(npz_path, allow_pickle=True) as f:
            full_data = {k: np.array(f[k]) for k in f.files}
            if indices is not None:
                indices = np.asarray(indices)
                subset_data = {}
                for k, v in full_data.items():
                    if isinstance(v, np.ndarray) and v.shape[0] == full_data['labels'].shape[0]:
                        subset_data[k] = v[indices]
                    else:
                        subset_data[k] = v
                self.data = subset_data
                print(f"[Dataset] Using subset indices: {len(indices)} samples")
            else:
                self.data = full_data
                print(f"[Dataset] Using full dataset: {len(self.data['labels'])} samples")

        with np.load(stats_path, allow_pickle=True) as f:
            self.stats = {k: np.array(f[k]) for k in f.files}
        self.modalities = modalities
        self.is_train = is_train

    def __len__(self):
        return len(self.data['labels'])

    def get_view(self, idx, mod, state=None):
        # 获取基础数据
        raw = self.data[f'{mod}_matrices'][idx]
        mask = self.data['mask_matrices'][idx]
        norm = self.get_norm_data(raw, mask, mod, self.stats)

        if self.is_train:
            if state is not None:
                norm = self.apply_shared_transforms(norm, state)
            if mod == 'rssi':
                return self.augment_rssi_pretrain(norm.clone())
            elif mod == 'phase':
                return self.augment_phase_pretrain(norm.clone())
            elif mod == 'doppler':
                return self.augment_doppler_pretrain(norm.clone())
        return norm

    def __getitem__(self, idx):
        L = self.data[f'{self.modalities[0]}_matrices'][idx].shape[1]

        if len(self.modalities) == 1:
            state1 = self.generate_random_state(L)
            state2 = self.generate_random_state(L)

            m = self.modalities[0]
            view1 = self.get_view(idx, m, state=state1)
            view2 = self.get_view(idx, m, state=state2)
            return view1, view2
        else:
            shared_state = self.generate_random_state(L)

            m1 = self.modalities[0]
            m2 = self.modalities[1]

            view1 = self.get_view(idx, m1, state=shared_state)
            view2 = self.get_view(idx, m2, state=shared_state)
            return view1, view2


class ActivityClassificationDataset(Dataset, AugmentationMixin):
    def __init__(self, data_dict, stats, modalities, is_train=True):
        self.data = data_dict

        if isinstance(stats, np.lib.npyio.NpzFile):
            self.stats = {k: np.array(stats[k]) for k in stats.files}
        else:
            self.stats = stats

        self.modalities = modalities
        self.is_train = is_train

        self.labels_raw = self.data['labels']
        ul = sorted(np.unique(self.labels_raw))
        self.label_map = {str(l): i for i, l in enumerate(ul)}
        self.labels = torch.tensor([self.label_map[str(l)] for l in self.labels_raw], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = {}
        for m in self.modalities:
            raw = self.data[f'{m}_matrices'][idx]
            mask = self.data['mask_matrices'][idx]
            norm = self.get_norm_data(raw, mask, m, self.stats)

            if self.is_train:
                if m == 'rssi':
                    norm = self.augment_rssi_finetune(norm)
                elif m == 'phase':
                    norm = self.augment_phase_finetune(norm)
                else:
                    norm = self.augment_doppler_finetune(norm)

            inputs[m] = norm

        return inputs, self.labels[idx]