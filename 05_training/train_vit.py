import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from model_vit import get_vit_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == len(dataloader):
            current_loss = running_loss / total
            current_acc = correct / total * 100
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%")
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc

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
        
    val_loss = running_loss / total
    val_acc = correct / total * 100
    return val_loss, val_acc, np.array(all_preds), np.array(all_targets)

def plot_curves(history, save_path="training_curves_vit.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", lw=2)
    ax1.set_title("ViT-B/16 CrossEntropy Loss vs Epochs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11)
    
    # Accuracy curve
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, history["val_acc"], "s--", label="Val Accuracy", color="#d62728", lw=2)
    ax2.set_title("ViT-B/16 Accuracy (%) vs Epochs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ViT training curves to {save_path}")

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix_vit.png"):
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True
    )
    plt.title("ViT-B/16 Confusion Matrix on Untouched Test Set", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved ViT confusion matrix heatmap to {save_path}")

def run(eval_only=False):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MB")
        
    # Hyperparameters (matching ResNet baseline)
    batch_size = 32
    max_epochs = 10
    early_stopping_patience = 3
    learning_rate = 1e-4
    weight_decay = 1e-2
    checkpoint_path = "best_vit_b16.pth"
    
    # Load dataset splits (same untouched dataset_splits.json)
    num_workers = 0  # Windows-safe
    train_loader, val_loader, test_loader, splits = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers
    )
    classes = splits["classes"]
    num_classes = len(classes)
    print(f"Loaded dataset splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    print(f"Target classes ({num_classes}): {classes}")
    
    criterion = nn.CrossEntropyLoss()

    if not eval_only:
        # Model
        print("Instantiating pretrained ViT-B/16 baseline...")
        model = get_vit_model(num_classes=num_classes, pretrained=True).to(device)
        
        # Loss, Optimizer & Scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        start_time = time.time()
        
        print("\n" + "="*65)
        print(f"Starting ViT-B/16 Training: Epochs={max_epochs}, Batch={batch_size}, LR={learning_rate}")
        print("="*65)
        
        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()
            print(f"\n--- Epoch [{epoch}/{max_epochs}] ---")
            
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            duration = time.time() - epoch_start
            print(f"Epoch [{epoch}/{max_epochs}] ({duration:.1f}s) - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Checkpoint saving & Early stopping
            if val_acc > best_val_acc:
                print(f"  --> Validation accuracy improved from {best_val_acc:.2f}% to {val_acc:.2f}%. Saving best model to '{checkpoint_path}'...")
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "classes": classes,
                    "class_to_idx": splits["class_to_idx"]
                }, checkpoint_path)
            else:
                patience_counter += 1
                print(f"  --> No improvement in val accuracy for {patience_counter} consecutive epoch(s).")
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs (patience={early_stopping_patience}).")
                    break
                    
        total_train_time = time.time() - start_time
        print(f"\nViT-B/16 Training finished in {total_train_time/60:.2f} minutes. Best Val Acc: {best_val_acc:.2f}%")
        
        # Save training curves
        plot_curves(history, "training_curves_vit.png")
    else:
        print(f"Evaluating existing checkpoint '{checkpoint_path}' without training...")
        
    # Evaluate on Untouched Test Set
    print("\n" + "="*65)
    print(f"Evaluating Best ViT-B/16 Model on Untouched Test Set ({len(splits['test'])} samples)...")
    print("="*65)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    best_epoch = checkpoint.get("epoch", None)
    
    model = get_vit_model(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    
    # Metrics computation
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=classes, digits=4)
    
    plot_confusion_matrix(cm, classes, "confusion_matrix_vit.png")
    
    results = {
        "model": "ViT-B/16",
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "macro_metrics": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1_score": float(f1_macro)
        },
        "weighted_metrics": {
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "f1_score": float(f1_weighted)
        },
        "per_class": {cls: report_dict[cls] for cls in classes},
        "confusion_matrix": cm.tolist(),
        "classes": classes
    }
    
    with open("test_results_vit.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nViT-B/16 Test Set Loss:     {test_loss:.4f}")
    print(f"ViT-B/16 Test Set Accuracy: {test_acc:.2f}%")
    print(f"Macro Metrics    - Precision: {precision_macro*100:.2f}% | Recall: {recall_macro*100:.2f}% | F1: {f1_macro*100:.2f}%")
    print(f"Weighted Metrics - Precision: {precision_weighted*100:.2f}% | Recall: {recall_weighted*100:.2f}% | F1: {f1_weighted*100:.2f}%")
    print("\n--- Per-Class Classification Report ---")
    print(report_text)
    
    print("\n--- Confusion Matrix (Rows: True, Cols: Predicted) ---")
    header = f"{'True \\ Pred':<25}" + "".join([f"{c[:10]:>12}" for c in classes])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{classes[i]:<25}" + "".join([f"{val:>12}" for val in row])
        print(row_str)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate ViT-B/16 model on chilli leaf disease dataset.")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate checkpoint on test set without retraining")
    args = parser.parse_args()
    run(eval_only=args.eval_only)
