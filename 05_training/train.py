
import os
import csv
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

from dataset import get_dataloaders
from model import get_model

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "resnet50_baseline")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    return os.path.join(OUTPUT_DIR, filename)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = (scaler is not None)

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] Loss: {running_loss/total:.4f}, Acc: {correct/total*100:.2f}%")

    return running_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    return running_loss / total, correct / total * 100, np.array(all_preds), np.array(all_targets)


def plot_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"],   "s--", label="Val Loss",   color="#ff7f0e", lw=2)
    ax1.set_title("CrossEntropy Loss vs Epochs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6); ax1.legend(fontsize=11)
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, history["val_acc"],   "s--", label="Val Accuracy",   color="#d62728", lw=2)
    ax2.set_title("Accuracy (%) vs Epochs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6); ax2.legend(fontsize=11)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"Saved training curves  -> {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title("Confusion Matrix - Untouched Test Set", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class",      fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"Saved confusion matrix -> {save_path}")


def save_history_csv(history, save_path):
    keys = ["train_loss", "train_acc", "val_loss", "val_acc"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(len(history["train_loss"])):
            writer.writerow([i + 1] + [history[k][i] for k in keys])
    print(f"Saved training history -> {save_path}")


def run():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = False
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"  GPU  : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")
    print(f"  AMP  : {'enabled' if use_amp else 'disabled'}")

    # Hyperparameters
    BATCH_SIZE          = 16
    MAX_EPOCHS          = 20
    EARLY_STOP_PATIENCE = 5
    LEARNING_RATE       = 1e-4
    WEIGHT_DECAY        = 1e-4
    CHECKPOINT_PATH     = out("best_resnet50_baseline.pth")

    # Data
    train_loader, val_loader, test_loader, splits = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=0
    )
    classes     = splits["classes"]
    num_classes = len(classes)

    n_train = len(splits["train"])
    n_val   = len(splits["val"])
    n_test  = len(splits["test"])
    total   = n_train + n_val + n_test

    print("\n" + "="*60)
    print("DATASET SPLIT VERIFICATION")
    print("="*60)
    print(f"  Train     : {n_train:,}  (expected 6,172)")
    print(f"  Validation: {n_val:,}  (expected 1,323)")
    print(f"  Test      : {n_test:,}  (expected 1,322)")
    print(f"  Total     : {total:,}  (expected 8,817)")
    assert n_train == 6172, f"Train count mismatch: {n_train}"
    assert n_val   == 1323, f"Val   count mismatch: {n_val}"
    assert n_test  == 1322, f"Test  count mismatch: {n_test}"
    print("  All split counts VERIFIED")
    print(f"\n  Classes ({num_classes}): {classes}")

    # Model
    print("\nInstantiating pretrained ResNet-50 baseline (ImageNet weights)...")
    model = get_model(num_classes=num_classes, pretrained=True).to(device)
    print(f"  Final FC : in_features={model.fc.in_features}, out_features={num_classes}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    scaler = GradScaler("cuda") if use_amp else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss        = float("inf")
    best_val_acc_at_best = 0.0
    patience_counter     = 0
    start_time           = time.time()

    print("\n" + "="*60)
    print(f"TRAINING | Epochs={MAX_EPOCHS}, Batch={BATCH_SIZE}, LR={LEARNING_RATE}, WD={WEIGHT_DECAY}")
    print(f"         | Early stop patience={EARLY_STOP_PATIENCE} (on val loss)")
    print("="*60)

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n--- Epoch [{epoch}/{MAX_EPOCHS}] ---")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        duration = time.time() - epoch_start
        print(f"Epoch [{epoch}/{MAX_EPOCHS}] ({duration:.1f}s) | Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            print(f"  --> Val loss improved {best_val_loss:.4f} -> {val_loss:.4f}. Saving checkpoint...")
            best_val_loss        = val_loss
            best_val_acc_at_best = val_acc
            patience_counter     = 0
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "best_val_loss":       best_val_loss,
                "best_val_acc":        best_val_acc_at_best,
                "classes":             classes,
                "class_to_idx":        splits["class_to_idx"],
                "history":             history,
            }, CHECKPOINT_PATH)
        else:
            patience_counter += 1
            print(f"  --> No improvement for {patience_counter}/{EARLY_STOP_PATIENCE} epoch(s).")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping triggered after epoch {epoch}.")
                break

    total_train_time = time.time() - start_time
    print(f"\nTraining complete: {total_train_time/60:.2f} min | Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc_at_best:.2f}%")

    plot_curves(history, out("training_curves.png"))
    save_history_csv(history, out("training_history.csv"))
    with open(out("training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set
    print("\n" + "="*60)
    print(f"TEST SET EVALUATION  ({n_test:,} samples, untouched)")
    print("="*60)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded best checkpoint from epoch {checkpoint['epoch']} (val loss={checkpoint['best_val_loss']:.4f}, val acc={checkpoint['best_val_acc']:.2f}%)")

    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)

    precision_macro,    recall_macro,    f1_macro,    _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    cm          = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=classes, digits=4)

    print(f"\nTest Loss      : {test_loss:.4f}")
    print(f"Test Accuracy  : {test_acc:.2f}%")
    print(f"Macro   P/R/F1 : {precision_macro*100:.2f}% / {recall_macro*100:.2f}% / {f1_macro*100:.2f}%")
    print(f"Weighted P/R/F1: {precision_weighted*100:.2f}% / {recall_weighted*100:.2f}% / {f1_weighted*100:.2f}%")
    print("\n--- Per-Class Classification Report ---")
    print(report_text)

    print("\n--- Confusion Matrix (Rows=True, Cols=Predicted) ---")
    header = f"{'True \\ Pred':<25}" + "".join([f"{c[:10]:>12}" for c in classes])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        print(f"{classes[i]:<25}" + "".join([f"{v:>12}" for v in row]))

    plot_confusion_matrix(cm, classes, out("confusion_matrix.png"))

    results = {
        "model":          "ResNet-50 (ImageNet pretrained)",
        "split_file":     "dataset_splits_final.json",
        "split_counts":   {"train": n_train, "val": n_val, "test": n_test, "total": total},
        "best_epoch":     checkpoint["epoch"],
        "best_val_loss":  float(best_val_loss),
        "best_val_acc":   float(best_val_acc_at_best),
        "test_loss":      float(test_loss),
        "test_accuracy":  float(test_acc),
        "macro_metrics":  {"precision": float(precision_macro), "recall": float(recall_macro), "f1_score": float(f1_macro)},
        "weighted_metrics":{"precision": float(precision_weighted), "recall": float(recall_weighted), "f1_score": float(f1_weighted)},
        "per_class":      {cls: report_dict[cls] for cls in classes},
        "confusion_matrix": cm.tolist(),
        "classes":        classes,
        "hyperparameters": {
            "batch_size":    BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": EARLY_STOP_PATIENCE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer":     "AdamW", "scheduler": "CosineAnnealingLR", "amp": use_amp,
        },
    }
    with open(out("test_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved test results     -> {out('test_results.json')}")

    with open(out("classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved class. report    -> {out('classification_report.txt')}")

    config = {
        "classes": classes, "class_to_idx": splits["class_to_idx"],
        "num_classes": num_classes, "input_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    with open(out("config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config           -> {out('config.json')}")

    print("\n" + "="*60)
    print(f"ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print("="*60)
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        print(f"  {fname:<45} {os.path.getsize(fpath):>12,} bytes")


if __name__ == "__main__":
    run()
