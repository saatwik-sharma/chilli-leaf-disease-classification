import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model_hybrid_finetuned import get_hybrid_model


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = r"outputs\hybrid_cnn_vit_finetuned\best_hybrid_cnn_vit_finetuned.pth"
SPLIT_FILE = r"dataset_splits_final.json"

OUTPUT_DIR = r"outputs\xai_gradcam"

NUM_CLASSES = 6

CLASS_NAMES = [
    "Bacterial_Spot",
    "Cercospora_Leaf_Spot",
    "Curl_Virus",
    "Healthy_Leaf",
    "Nutrition_Deficiency",
    "Powdery_Mildew"
]

# Number of correctly classified examples per class
EXAMPLES_PER_CLASS = 3


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING FINAL FINE-TUNED HYBRID MODEL")
print("=" * 60)

model = get_hybrid_model(
    num_classes=NUM_CLASSES,
    pretrained=False
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# LOAD TEST SPLIT
# ============================================================

print("\nLoading locked test split...")

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

test_samples = splits["test"]

print("Test images:", len(test_samples))


# ============================================================
# HELPER: MODEL PREDICTION
# ============================================================

def predict(image_tensor):

    with torch.no_grad():
        output = model(image_tensor)

    prediction = output.argmax(dim=1).item()

    return prediction


# ============================================================
# FIND CORRECTLY CLASSIFIED IMAGES
# ============================================================

print("\nSearching for correctly classified test images...")

selected = {
    class_name: []
    for class_name in CLASS_NAMES
}

for sample in test_samples:

    image_path = sample["path"]
    true_label = sample["label"]

    if not os.path.exists(image_path):
        continue

    try:

        image = Image.open(image_path).convert("RGB")

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        prediction = predict(input_tensor)

        if prediction == true_label:

            class_name = CLASS_NAMES[true_label]

            if len(selected[class_name]) < EXAMPLES_PER_CLASS:

                selected[class_name].append({
                    "path": image_path,
                    "label": true_label,
                    "prediction": prediction
                })

    except Exception as e:

        print("Skipping:", image_path)
        print("Error:", e)


# ============================================================
# PRINT SELECTION SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SELECTED XAI EXAMPLES")
print("=" * 60)

total_selected = 0

for class_name in CLASS_NAMES:

    count = len(selected[class_name])

    print(f"{class_name}: {count}")

    total_selected += count

print("Total selected:", total_selected)


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

target_layers = [
    model.cnn.layer4[-1]
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

print("\n" + "=" * 60)
print("GENERATING GRAD-CAM")
print("=" * 60)

with GradCAM(
    model=model.cnn,
    target_layers=target_layers
) as cam:

    for class_name in CLASS_NAMES:

        examples = selected[class_name]

        for index, sample in enumerate(examples):

            image_path = sample["path"]
            true_label = sample["label"]

            image = Image.open(image_path).convert("RGB")

            # Resize for visualization
            image_resized = image.resize((224, 224))

            # Convert to float [0,1]
            rgb_image = np.array(image_resized).astype(np.float32) / 255.0

            # Model input
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            # ------------------------------------------------
            # Grad-CAM
            # ------------------------------------------------

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[
                    ClassifierOutputTarget(true_label)
                ]
            )

            grayscale_cam = grayscale_cam[0]

            # Overlay
            visualization = show_cam_on_image(
                rgb_image,
                grayscale_cam,
                use_rgb=True
            )

            # ------------------------------------------------
            # Plot
            # ------------------------------------------------

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(12, 4)
            )

            axes[0].imshow(rgb_image)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(grayscale_cam, cmap="jet")
            axes[1].set_title("Grad-CAM")
            axes[1].axis("off")

            axes[2].imshow(visualization)
            axes[2].set_title(
                f"Overlay\n{class_name}"
            )
            axes[2].axis("off")

            plt.tight_layout()

            # Safe filename
            safe_class_name = class_name.replace(" ", "_")

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{safe_class_name}_{index + 1}.png"
            )

            plt.savefig(
                output_path,
                dpi=200,
                bbox_inches="tight"
            )

            plt.close()

            print("Saved:", output_path)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("XAI COMPLETE")
print("=" * 60)

print("Output folder:")
print(OUTPUT_DIR)

print("\nGenerated Grad-CAM visualizations for correctly")
print("classified test images.")