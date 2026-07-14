import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import glob
import json
import random

from datasets import ActivityClassificationDataset
from models import ClassificationModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, data_loader, device, criterion, num_classes):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    return avg_loss, accuracy, f1, cm


def plot_performance_curves(history, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(history['train_loss'], label='Training Loss', marker='.', markersize=5)
    ax1.plot(history['val_loss'], label='Validation Loss', marker='.', markersize=5)
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(history['val_acc'], label='Val Accuracy', color='C1', marker='.')
    ax2.plot(history['val_f1'], label='Val F1-Score', color='green', marker='.')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Performance')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, normalize=True):
    plt.figure(figsize=(max(14, len(class_names) * 0.6), max(12, len(class_names) * 0.5)))

    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = np.true_divide(cm, cm_sum)
            cm_normalized[~np.isfinite(cm_normalized)] = 0
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
        sns.heatmap(cm_normalized, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)

    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_finetuning(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.linear_probe:
        mode = "LinearProbe"
    else:
        mode = "From_Scratch" if args.from_scratch else "Finetuned"

    pt_str = "_".join(args.pretrain_modalities) if not args.from_scratch else "None"
    ft_str = "_".join(args.finetune_modalities)
    ratio_str = f"{int(args.sample_ratio * 100)}pct"
    lr_str = f"lr_{args.lr}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_dir_name = f"{ratio_str}_{mode}_PT-{pt_str}_FT-{ft_str}_{lr_str}_seed{args.seed}_{timestamp}"
    results_dir = os.path.join(args.results_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f"=== Experiment Start ===")
    print(f"Task Mode : {mode}")
    print(f"Data Ratio: {ratio_str} (Seed: {args.seed})")
    print(f"Pipeline  : Pretrain[{pt_str}] -> Finetune[{ft_str}]")
    print(f"Output Dir: {results_dir}")
    print("=" * 60 + "\n")

    with open(os.path.join(results_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    print("[Step 1] Loading Data...")
    try:
        with np.load(args.data_path, allow_pickle=True) as f:
            data = {k: f[k] for k in f.files}
        with np.load(args.stats_path, allow_pickle=True) as f:
            stats = {k: f[k] for k in f.files}
    except FileNotFoundError as e:
        print(f"Error: {e}");
        return

    indices = np.arange(len(data['labels']))
    labels_for_split = data['labels']

    train_val_idx, test_idx = train_test_split(
        indices, test_size=args.test_split, stratify=labels_for_split, random_state=42
    )

    train_val_labels = labels_for_split[train_val_idx]
    train_full_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.125, stratify=train_val_labels, random_state=42
    )

    if args.sample_ratio < 1.0:
        labels_subset = labels_for_split[train_full_idx]
        train_idx, _ = train_test_split(
            train_full_idx, train_size=args.sample_ratio, stratify=labels_subset, random_state=42
        )
    else:
        train_idx = train_full_idx

    needed_keys = [f"{m}_matrices" for m in args.finetune_modalities] + ['mask_matrices', 'labels']

    def get_subset_dict(indices):
        return {k: data[k][indices] for k in needed_keys}

    train_dataset = ActivityClassificationDataset(get_subset_dict(train_idx), stats, args.finetune_modalities,
                                                  is_train=True)
    val_dataset = ActivityClassificationDataset(get_subset_dict(val_idx), stats, args.finetune_modalities,
                                                is_train=False)
    test_dataset = ActivityClassificationDataset(get_subset_dict(test_idx), stats, args.finetune_modalities,
                                                 is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    unique_labels = sorted(np.unique(data['labels']))
    class_names = [str(l) for l in unique_labels]
    num_classes = len(unique_labels)
    first_mod_key = f"{args.finetune_modalities[0]}_matrices"
    num_tags = data[first_mod_key].shape[1]

    print(f"Dataset Sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # 6. 初始化模型
    print("[Step 2] Initializing Model...")
    model = ClassificationModel(num_tags, num_classes, args.finetune_modalities, args.feature_dim).to(device)

    # 7. 加载预训练权重
    if not args.from_scratch:
        if args.pretrained_path and os.path.exists(args.pretrained_path):
            checkpoint_path = args.pretrained_path
            print(f"Using specified checkpoint: {checkpoint_path}")
        else:
            pt_mode_str = "_".join(args.pretrain_modalities)
            search_pattern = f"checkpoints_3_newAug_Mask/tagclr_best_model_{pt_mode_str}*.pth"
            candidates = glob.glob(search_pattern)

            valid_paths = []
            all_known_modes = ['rssi', 'phase', 'doppler']

            for p in candidates:
                filename = os.path.basename(p)
                is_valid = True

                for m in all_known_modes:
                    if m in args.pretrain_modalities:
                        if m not in filename:
                            is_valid = False
                            break
                    else:
                        if m in filename:
                            is_valid = False
                            break

                if is_valid:
                    valid_paths.append(p)

            if valid_paths:
                checkpoint_path = max(valid_paths, key=os.path.getctime)
                print(f"Auto-found Exact Checkpoint: {checkpoint_path}")
            else:
                print(f"Warning: No strict match found for '{pt_mode_str}'. Candidates were: {candidates}")
                print("Switching to From Scratch.")
                args.from_scratch = True
                checkpoint_path = None

        if checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                print(f"Loading weights from {os.path.basename(checkpoint_path)}...")

                for m in args.finetune_modalities:
                    key_in_ckpt = f"encoder_{m}"

                    if key_in_ckpt in checkpoint:
                        model.encoders[m].load_state_dict(checkpoint[key_in_ckpt])
                        print(f"  -> Successfully loaded encoder: {m}")
                    else:
                        print(f"  [Warning] Encoder for '{m}' not found in checkpoint! It will remain random.")

            except Exception as e:
                print(f"Error loading weights: {e}")
                return

    if args.linear_probe:
        print("--- Mode: Linear Probe (Freezing Backbone) ---")
        for m in args.finetune_modalities:
            for p in model.encoders[m].parameters(): p.requires_grad = False

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.patience)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    do_warmup = (not args.from_scratch) and (not args.linear_probe)
    warmup_epochs = 3

    print("\n[Step 3] Start Training Loop...")
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}
    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):

        if do_warmup:
            if epoch < warmup_epochs:
                for m in args.finetune_modalities:
                    for p in model.encoders[m].parameters(): p.requires_grad = False
                for p in model.classifier.parameters(): p.requires_grad = True

                if epoch == 0: print(f">>> [Warmup Start] Backbone Frozen for {warmup_epochs} epochs.")
            elif epoch == warmup_epochs:
                for m in args.finetune_modalities:
                    for p in model.encoders[m].parameters(): p.requires_grad = True
                print(">>> [Warmup End] Backbone Unfrozen. Full Fine-tuning starts.")

        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch + 1}/{args.epochs}", leave=True)
        for inputs, labels in loop:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, device, criterion, num_classes)
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)

        print_str = f"Ep {epoch + 1} | T_Loss:{avg_train_loss:.4f} | V_Loss:{avg_val_loss:.4f} | V_F1:{val_f1:.4f} | LR:{current_lr:.1e}"

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            print_str += " (*)"
        print(print_str)

    print("\n--- Training Finished ---")
    print(f"Best Val F1: {best_val_f1:.4f} @ Ep {best_epoch}")

    if os.path.exists(os.path.join(results_dir, "best_model.pth")):
        model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))

    test_loss, test_acc, test_f1, test_cm = evaluate(model, test_loader, device, criterion, num_classes)
    print(f"Final Test F1: {test_f1:.4f}")

    pd.DataFrame(history).to_csv(os.path.join(results_dir, "epoch_metrics.csv"), index=False)

    # 保存 Summary
    with open(os.path.join(results_dir, "summary.json"), 'w') as f:
        json.dump({
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'test_f1': test_f1,
            'test_acc': test_acc,
            'config': vars(args)
        }, f, indent=4)

    plot_performance_curves(history, os.path.join(results_dir, "performance_curves.png"))

    plot_confusion_matrix(test_cm, class_names, os.path.join(results_dir, "final_confusion_matrix_normalized.png"),
                          normalize=True)

    print(f"Done! All results saved to: {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_modalities', nargs='+', default=['rssi', 'phase'],
                        help="List of modalities used in pretraining (e.g. rssi phase)")
    parser.add_argument('--finetune_modalities', nargs='+', required=True,
                        help="List of modalities to use for finetuning (e.g. rssi)")

    parser.add_argument('--from_scratch', action='store_true')
    parser.add_argument('--linear_probe', action='store_true')
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', default='data/final_dataset_corrected.npz')
    parser.add_argument('--stats_path', default='data/norm_stats_corrected.npz')
    parser.add_argument('--pretrained_path', type=str, default='')
    parser.add_argument('--results_dir', default='evaluation_results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--test_split', type=float, default=0.2)
    parser.add_argument('--feature_dim', type=int, default=128)
    parser.add_argument('--ablation', type=str, default=None)

    args = parser.parse_args()

    # 互斥检查
    if args.linear_probe and args.from_scratch:
        print("Error: Cannot set both --linear_probe and --from_scratch")
        exit()

    run_finetuning(args)