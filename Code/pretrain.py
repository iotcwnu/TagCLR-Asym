import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import ContrastiveDataset
from models import TagCLR


def info_nce_loss(f1, f2, t):
    f1 = nn.functional.normalize(f1, dim=1)
    f2 = nn.functional.normalize(f2, dim=1)
    labels = torch.arange(f1.shape[0], device=f1.device)
    sim = torch.matmul(f1, f2.T) / t
    return (nn.CrossEntropyLoss()(sim, labels) + nn.CrossEntropyLoss()(sim.T, labels)) / 2

def run_pretraining(args):
    mod_name = "_".join(args.modalities)
    print(f"=== Pretrain Task: {mod_name} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with np.load(args.data_path, allow_pickle=True) as f:
        labels = np.array(f['labels'])
    indices = np.arange(len(labels))

    dataset = ContrastiveDataset(args.data_path, args.stats_path, modalities=args.modalities, is_train=True, indices=indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    num_tags = dataset.data['rssi_matrices'].shape[1]
    model = TagCLR(num_tags=num_tags, modalities=args.modalities, feature_dim=args.feature_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    loss_hist = []
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total = 0
        loop = tqdm(loader, desc=f"Ep {epoch + 1}", leave=True)
        for v1, v2 in loop:
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(v1, v2)
            loss = info_nce_loss(z1, z2, args.temperature)
            loss.backward()
            optimizer.step()
            total += loss.item()
            loop.set_postfix(loss=loss.item())

        avg = total / len(loader)
        loss_hist.append(avg)
        scheduler.step()

        if avg < best_loss:
            best_loss = avg
            state = {'loss': best_loss, 'epoch': epoch}
            for m in args.modalities:
                state[f'encoder_{m}'] = model.encoders[m].state_dict()
            torch.save(state, os.path.join(args.checkpoint_dir, f"tagclr_best_model_{mod_name}.pth"))

    plt.figure();
    plt.plot(loss_hist);
    plt.savefig(os.path.join(args.checkpoint_dir, f"loss_{mod_name}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 核心：使用 nargs='+' 接受列表
    parser.add_argument('--modalities', nargs='+', required=True, help="e.g. rssi phase")
    parser.add_argument('--data_path', default='datasets/final_dataset_corrected.npz')
    parser.add_argument('--stats_path', default='datasets/norm_stats_corrected_trainonly.npz')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--feature_dim', type=int, default=128)
    args = parser.parse_args()
    run_pretraining(args)