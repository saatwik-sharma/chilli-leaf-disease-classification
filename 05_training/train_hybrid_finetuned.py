import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from dataset import get_dataloaders
from model_hybrid_finetuned import get_hybrid_model


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 6

# RTX 3050 6 GB
BATCH_SIZE = 2

# Effective batch size = 2 x 8 = 16
ACCUMULATION_STEPS = 8

MAX_EPOCHS = 10
PATIENCE = 3

# Differential learning rates
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4

WEIGHT_DECAY = 1e-2
DROPOUT = 0.3

NUM_WORKERS = 0

OUTPUT_DIR = "outputs/hybrid_cnn_vit_finetuned"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE INFORMATION
# ============================================================

print("=" * 70)
print("FULLY FINE-TUNED CNN + ViT HYBRID")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

train_loader, val_loader, test_loader, split = get_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

print(
    f"Train samples: {len(train_loader.dataset)}"
)

print(
    f"Validation samples: {len(val_loader.dataset)}"
)

print(
    f"Test samples: {len(test_loader.dataset)}"
)

print(
    f"Train batches: {len(train_loader)}"
)

print(
    f"Validation batches: {len(val_loader)}"
)

print(
    f"Test batches: {len(test_loader)}"
)


# ============================================================
# VERIFY DATASET SPLIT
# ============================================================

total_samples = (
    len(train_loader.dataset)
    + len(val_loader.dataset)
    + len(test_loader.dataset)
)

print(
    f"Total samples: {total_samples}"
)

if total_samples != 8817:

    raise RuntimeError(
        f"Unexpected dataset size: {total_samples}. "
        "Expected 8817."
    )

print("Dataset size verification passed.")


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("=" * 70)
print("CREATING FULLY FINE-TUNED CNN + ViT MODEL")
print("=" * 70)

model = get_hybrid_model(
    num_classes=NUM_CLASSES,
    pretrained=True,
    dropout_p=DROPOUT
).to(DEVICE)


# ============================================================
# PARAMETER VERIFICATION
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print(
    f"Total parameters:     {total_params:,}"
)

print(
    f"Trainable parameters: {trainable_params:,}"
)

if total_params != trainable_params:

    raise RuntimeError(
        "ERROR: Some parameters are frozen. "
        "Full fine-tuning requires all parameters "
        "to be trainable."
    )

print()
print("ALL PARAMETERS ARE TRAINABLE.")
print("ResNet-50 -> TRAINABLE")
print("ViT-B/16  -> TRAINABLE")
print("Classifier -> TRAINABLE")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    [
        {
            "params": model.cnn.parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": model.vit.parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCHS
)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    optimizer.zero_grad(
        set_to_none=True
    )

    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Gradient accumulation
        scaled_loss = (
            loss / ACCUMULATION_STEPS
        )

        scaled_loss.backward()

        # Update weights
        if (
            (batch_idx + 1) % ACCUMULATION_STEPS == 0
            or
            (batch_idx + 1) == total_batches
        ):

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        # Track loss
        running_loss += (
            loss.item() * images.size(0)
        )

        # Predictions
        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        # Progress
        if (batch_idx + 1) % 50 == 0:

            current_accuracy = accuracy_score(
                all_labels,
                all_predictions
            )

            print(
                f"  Batch [{batch_idx + 1}/{total_batches}] "
                f"Loss: {loss.item():.4f} "
                f"Acc: {current_accuracy * 100:.2f}%"
            )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = (
        running_loss /
        len(loader.dataset)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(
    f"Batch size:              {BATCH_SIZE}"
)

print(
    f"Accumulation steps:      {ACCUMULATION_STEPS}"
)

print(
    f"Effective batch size:    "
    f"{BATCH_SIZE * ACCUMULATION_STEPS}"
)

print(
    f"Backbone LR:             {BACKBONE_LR}"
)

print(
    f"Classifier LR:           {CLASSIFIER_LR}"
)

print(
    f"Weight decay:            {WEIGHT_DECAY}"
)

print(
    f"Maximum epochs:          {MAX_EPOCHS}"
)

print(
    f"Early stopping patience: {PATIENCE}"
)

print()
print("AMP: OFF")
print("ResNet-50: FULLY TRAINABLE")
print("ViT-B/16: FULLY TRAINABLE")
print("Classifier: TRAINABLE")

print("=" * 70)


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_loss = float("inf")
best_val_accuracy = 0.0

epochs_without_improvement = 0

history = []


for epoch in range(MAX_EPOCHS):

    print()
    print("=" * 70)
    print(
        f"EPOCH {epoch + 1}/{MAX_EPOCHS}"
    )
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_loss, val_accuracy = validate(
        model,
        val_loader,
        criterion
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step()

    epoch_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print(
        f"Train Loss:      {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:  "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss:        {val_loss:.4f}"
    )

    print(
        f"Val Accuracy:    "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Epoch Time:      "
        f"{epoch_time / 60:.2f} minutes"
    )

    print(
        f"Backbone LR:     "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )

    print(
        f"Classifier LR:   "
        f"{optimizer.param_groups[2]['lr']:.8f}"
    )


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history.append(
        {
            "epoch": epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,

            "backbone_lr":
                optimizer.param_groups[0]["lr"],

            "classifier_lr":
                optimizer.param_groups[2]["lr"],

            "epoch_time_seconds":
                epoch_time
        }
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_val_accuracy = val_accuracy

        epochs_without_improvement = 0

        best_model_path = os.path.join(
            OUTPUT_DIR,
            "best_hybrid_cnn_vit_finetuned.pth"
        )

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print()
        print("🔥 NEW BEST MODEL SAVED!")

        print(
            f"Best Val Loss: "
            f"{best_val_loss:.4f}"
        )

        print(
            f"Best Val Accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )

    else:

        epochs_without_improvement += 1

        print()
        print(
            f"No improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.json"
)

with open(
    history_path,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

best_model_path = os.path.join(
    OUTPUT_DIR,
    "best_hybrid_cnn_vit_finetuned.pth"
)

if not os.path.exists(best_model_path):

    raise RuntimeError(
        "Best model checkpoint was not created."
    )

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)

print(
    "Best model loaded successfully."
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

model.eval()

all_predictions = []
all_labels = []

test_loss_total = 0.0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        test_loss_total += (
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# TEST METRICS
# ============================================================

test_loss = (
    test_loss_total /
    len(test_loader.dataset)
)

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)


(
    precision_macro,
    recall_macro,
    f1_macro,
    _
) = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


(
    precision_weighted,
    recall_weighted,
    f1_weighted,
    _
) = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "Bacterial_Spot",
    "Cercospora_Leaf_Spot",
    "Curl_Virus",
    "Healthy_Leaf",
    "Nutrition_Deficiency",
    "Powdery_Mildew"
]


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    digits=4,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Test Loss:       {test_loss:.4f}"
)

print(
    f"Test Accuracy:   "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("MACRO METRICS")

print(
    f"Precision:       "
    f"{precision_macro * 100:.2f}%"
)

print(
    f"Recall:          "
    f"{recall_macro * 100:.2f}%"
)

print(
    f"F1 Score:        "
    f"{f1_macro * 100:.2f}%"
)

print()
print("WEIGHTED METRICS")

print(
    f"Precision:       "
    f"{precision_weighted * 100:.2f}%"
)

print(
    f"Recall:          "
    f"{recall_weighted * 100:.2f}%"
)

print(
    f"F1 Score:        "
    f"{f1_weighted * 100:.2f}%"
)

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE TEST RESULTS
# ============================================================

test_results = {
    "test_loss":
        test_loss,

    "test_accuracy":
        test_accuracy,

    "macro_precision":
        precision_macro,

    "macro_recall":
        recall_macro,

    "macro_f1":
        f1_macro,

    "weighted_precision":
        precision_weighted,

    "weighted_recall":
        recall_weighted,

    "weighted_f1":
        f1_weighted,

    "confusion_matrix":
        cm.tolist()
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "test_results.json"
    ),
    "w"
) as f:

    json.dump(
        test_results,
        f,
        indent=4
    )


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {

    "model":
        "Hybrid ResNet-50 + ViT-B/16",

    "full_finetuning":
        True,

    "pretrained":
        True,

    "num_classes":
        NUM_CLASSES,

    "batch_size":
        BATCH_SIZE,

    "gradient_accumulation_steps":
        ACCUMULATION_STEPS,

    "effective_batch_size":
        BATCH_SIZE * ACCUMULATION_STEPS,

    "max_epochs":
        MAX_EPOCHS,

    "patience":
        PATIENCE,

    "backbone_learning_rate":
        BACKBONE_LR,

    "classifier_learning_rate":
        CLASSIFIER_LR,

    "weight_decay":
        WEIGHT_DECAY,

    "dropout":
        DROPOUT,

    "amp":
        False,

    "dataset_total":
        total_samples,

    "test_set_used_only_for_final_evaluation":
        True
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "config.json"
    ),
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("🔥 FULL FINE-TUNED HYBRID TRAINING COMPLETE 🔥")
print("=" * 70)

print()
print(
    f"FINAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("Output directory:")
print(
    os.path.abspath(OUTPUT_DIR)
)

print()
print("Saved:")
print("  best_hybrid_cnn_vit_finetuned.pth")
print("  training_history.json")
print("  test_results.json")
print("  classification_report.txt")
print("  config.json")

print("=" * 70)