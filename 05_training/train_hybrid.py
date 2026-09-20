import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders
from model_hybrid import get_hybrid_model


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = os.path.join("outputs", "hybrid_cnn_vit")

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "best_hybrid_cnn_vit.pth"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "training_history.json"
)

CURVES_PATH = os.path.join(
    OUTPUT_DIR,
    "training_curves.png"
)

CONFUSION_MATRIX_PATH = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "test_results.json"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

CONFIG_PATH = os.path.join(
    OUTPUT_DIR,
    "config.json"
)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    # Keep frozen backbones in evaluation mode.
    # This prevents BatchNorm statistics in ResNet from changing.
    model.cnn.eval()
    model.vit.eval()
    model.classifier.train()

    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, labels) in enumerate(dataloader):

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Forward pass
        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        # Statistics
        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if (
            (batch_idx + 1) % 50 == 0
            or
            (batch_idx + 1) == len(dataloader)
        ):

            current_loss = (
                running_loss / total
            )

            current_acc = (
                correct / total * 100
            )

            print(
                f"  Batch [{batch_idx + 1}/"
                f"{len(dataloader)}] "
                f"Loss: {current_loss:.4f}, "
                f"Acc: {current_acc:.2f}%"
            )

    epoch_loss = running_loss / total

    epoch_acc = (
        correct / total * 100
    )

    return epoch_loss, epoch_acc


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_targets = []

    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

    epoch_loss = running_loss / total

    epoch_acc = (
        correct / total * 100
    )

    return (
        epoch_loss,
        epoch_acc,
        np.array(all_predictions),
        np.array(all_targets)
    )


# ============================================================
# TRAINING CURVES
# ============================================================

def plot_training_curves(
    history,
    save_path
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    # Loss
    axes[0].plot(
        epochs,
        history["train_loss"],
        "o-",
        label="Train Loss",
        linewidth=2
    )

    axes[0].plot(
        epochs,
        history["val_loss"],
        "s--",
        label="Validation Loss",
        linewidth=2
    )

    axes[0].set_title(
        "Hybrid CNN + ViT Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Cross-Entropy Loss"
    )

    axes[0].grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    axes[0].legend()

    # Accuracy
    axes[1].plot(
        epochs,
        history["train_acc"],
        "o-",
        label="Train Accuracy",
        linewidth=2
    )

    axes[1].plot(
        epochs,
        history["val_acc"],
        "s--",
        label="Validation Accuracy",
        linewidth=2
    )

    axes[1].set_title(
        "Hybrid CNN + ViT Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy (%)"
    )

    axes[1].grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved training curves to: {save_path}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    cm,
    class_names,
    save_path
):

    plt.figure(
        figsize=(9, 7)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )

    plt.title(
        "Hybrid CNN + ViT Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved confusion matrix to: {save_path}"
    )


# ============================================================
# MAIN
# ============================================================

def run():

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("HYBRID CNN + VISION TRANSFORMER TRAINING")
    print("=" * 70)

    print(
        f"Using device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        total_vram = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU VRAM: {total_vram:.2f} GB"
        )

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------

    batch_size = 16

    max_epochs = 10

    learning_rate = 1e-4

    weight_decay = 1e-2

    early_stopping_patience = 3

    num_workers = 0

    pretrained = True

    dropout = 0.3

    print("\nConfiguration:")
    print(
        f"  Batch size:              {batch_size}"
    )

    print(
        f"  Max epochs:              {max_epochs}"
    )

    print(
        f"  Learning rate:           {learning_rate}"
    )

    print(
        f"  Weight decay:            {weight_decay}"
    )

    print(
        f"  Early stopping patience: {early_stopping_patience}"
    )

    print(
        f"  AMP:                     DISABLED"
    )

    print(
        f"  Pretrained:              {pretrained}"
    )

    print(
        f"  Backbones:               FROZEN"
    )

    print(
        f"  Trainable component:     Fusion classifier only"
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "model": "Hybrid ResNet-50 + ViT-B/16",
        "pretrained": pretrained,
        "backbones_frozen": True,
        "trainable_component": "fusion classifier",
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
        "num_workers": num_workers,
        "amp": False,
        "image_size": 224,
        "dropout": dropout
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("\nLoading verified dataset split...")

    train_loader, val_loader, test_loader, splits = (
        get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers
        )
    )

    classes = splits["classes"]

    num_classes = len(classes)

    print(
        f"\nTrain samples: {len(splits['train'])}"
    )

    print(
        f"Validation samples: {len(splits['val'])}"
    )

    print(
        f"Test samples: {len(splits['test'])}"
    )

    print(
        f"Classes: {classes}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading pretrained Hybrid CNN + ViT model...")

    model = get_hybrid_model(
        num_classes=num_classes,
        pretrained=True,
        dropout_p=dropout
    ).to(device)

    # --------------------------------------------------------
    # Verify trainable parameters
    # --------------------------------------------------------

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    trainable_count = sum(
        p.numel()
        for p in trainable_parameters
    )

    total_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nTotal parameters:     {total_count:,}"
    )

    print(
        f"Trainable parameters: {trainable_count:,}"
    )

    if trainable_count != 16902:

        raise RuntimeError(
            "Unexpected number of trainable parameters. "
            "Expected 16,902. Check model_hybrid.py."
        )

    print(
        "Parameter verification passed."
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_epochs,
        eta_min=1e-6
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")

    best_val_acc = 0.0

    best_epoch = 0

    patience_counter = 0

    training_start = time.time()

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        max_epochs + 1
    ):

        epoch_start = time.time()

        print(
            f"\n{'=' * 25} "
            f"Epoch {epoch}/{max_epochs} "
            f"{'=' * 25}"
        )

        # Train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # Validation
        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        # Scheduler
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Store history
        history["train_loss"].append(
            train_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )

        epoch_time = (
            time.time() - epoch_start
        )

        print(
            f"\nEpoch {epoch}/{max_epochs}"
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

        print(
            f"Val Loss:   {val_loss:.4f} | "
            f"Val Acc:    {val_acc:.2f}%"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        print(
            f"Time: {epoch_time:.1f}s"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            improvement = (
                best_val_loss - val_loss
                if best_val_loss != float("inf")
                else 0
            )

            best_val_loss = val_loss

            best_val_acc = val_acc

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                    "classes": classes,
                    "class_to_idx": splits["class_to_idx"]
                },
                CHECKPOINT_PATH
            )

            print(
                f"\n*** BEST MODEL SAVED ***"
            )

            print(
                f"Best Val Loss: {best_val_loss:.4f}"
            )

            print(
                f"Val Accuracy:  {best_val_acc:.2f}%"
            )

        else:

            patience_counter += 1

            print(
                f"\nNo validation-loss improvement "
                f"for {patience_counter} epoch(s)."
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= early_stopping_patience:

            print(
                f"\nEarly stopping triggered."
            )

            print(
                f"Best epoch: {best_epoch}"
            )

            break

        # GPU memory cleanup
        if device.type == "cuda":

            torch.cuda.empty_cache()

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    total_training_time = (
        time.time() - training_start
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Training time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.2f}%"
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history["best_epoch"] = best_epoch

    history["best_val_loss"] = best_val_loss

    history["best_val_acc"] = best_val_acc

    history["training_time_minutes"] = (
        total_training_time / 60
    )

    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Plot curves
    # --------------------------------------------------------

    plot_training_curves(
        history,
        CURVES_PATH
    )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    best_model = get_hybrid_model(
        num_classes=num_classes,
        pretrained=False,
        dropout_p=dropout
    ).to(device)

    best_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_model.eval()

    print(
        f"Loaded best model from epoch "
        f"{checkpoint['epoch']}"
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 70)

    print(
        f"Test samples: {len(splits['test'])}"
    )

    test_loss, test_acc, y_pred, y_true = evaluate(
        best_model,
        test_loader,
        criterion,
        device
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    plot_confusion_matrix(
        cm,
        classes,
        CONFUSION_MATRIX_PATH
    )

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Hybrid CNN + ViT Classification Report\n"
        )

        f.write(
            "=" * 55 + "\n\n"
        )

        f.write(
            report_text
        )

        f.write(
            "\n\nConfusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

    # --------------------------------------------------------
    # Save JSON results
    # --------------------------------------------------------

    results = {

        "model":
            "Hybrid ResNet-50 + ViT-B/16",

        "architecture":
            "ResNet-50 features + ViT-B/16 features + concatenation + linear classifier",

        "total_parameters":
            total_count,

        "trainable_parameters":
            trainable_count,

        "best_epoch":
            int(checkpoint["epoch"]),

        "best_validation_loss":
            float(checkpoint["best_val_loss"]),

        "best_validation_accuracy":
            float(checkpoint["best_val_acc"]),

        "test_loss":
            float(test_loss),

        "test_accuracy":
            float(test_acc),

        "macro_metrics": {
            "precision":
                float(precision_macro),

            "recall":
                float(recall_macro),

            "f1_score":
                float(f1_macro)
        },

        "weighted_metrics": {
            "precision":
                float(precision_weighted),

            "recall":
                float(recall_weighted),

            "f1_score":
                float(f1_weighted)
        },

        "per_class": {
            cls: report_dict[cls]
            for cls in classes
        },

        "confusion_matrix":
            cm.tolist(),

        "classes":
            classes,

        "test_samples":
            len(splits["test"])
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL HYBRID MODEL RESULTS")
    print("=" * 70)

    print(
        f"Best Epoch:        {checkpoint['epoch']}"
    )

    print(
        f"Best Val Accuracy: {checkpoint['best_val_acc']:.2f}%"
    )

    print(
        f"Test Loss:         {test_loss:.4f}"
    )

    print(
        f"Test Accuracy:     {test_acc:.2f}%"
    )

    print(
        f"\nMacro Precision:   {precision_macro * 100:.2f}%"
    )

    print(
        f"Macro Recall:      {recall_macro * 100:.2f}%"
    )

    print(
        f"Macro F1:          {f1_macro * 100:.2f}%"
    )

    print(
        f"\nWeighted Precision: {precision_weighted * 100:.2f}%"
    )

    print(
        f"Weighted Recall:    {recall_weighted * 100:.2f}%"
    )

    print(
        f"Weighted F1:        {f1_weighted * 100:.2f}%"
    )

    print("\n" + "-" * 70)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("-" * 70)

    print(report_text)

    print("\n" + "-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(cm)

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Checkpoint:          {CHECKPOINT_PATH}"
    )

    print(
        f"Training history:    {HISTORY_PATH}"
    )

    print(
        f"Training curves:     {CURVES_PATH}"
    )

    print(
        f"Confusion matrix:    {CONFUSION_MATRIX_PATH}"
    )

    print(
        f"Classification report:{REPORT_PATH}"
    )

    print(
        f"Test results:        {RESULTS_PATH}"
    )

    print(
        "\nHybrid training and evaluation completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run()