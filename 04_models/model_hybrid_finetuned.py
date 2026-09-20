import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights
)


class HybridCNNViT(nn.Module):

    def __init__(self, num_classes=6, pretrained=True, dropout_p=0.3):
        super().__init__()

        # =========================
        # ResNet-50 CNN
        # =========================
        resnet_weights = ResNet50_Weights.DEFAULT if pretrained else None

        self.cnn = resnet50(weights=resnet_weights)

        cnn_in_features = self.cnn.fc.in_features

        self.cnn.fc = nn.Identity()

        # FULL FINE-TUNING:
        # Do NOT freeze CNN parameters.

        # =========================
        # ViT-B/16
        # =========================
        vit_weights = ViT_B_16_Weights.DEFAULT if pretrained else None

        self.vit = vit_b_16(weights=vit_weights)

        vit_in_features = self.vit.heads.head.in_features

        self.vit.heads = nn.Identity()

        # FULL FINE-TUNING:
        # Do NOT freeze ViT parameters.

        # =========================
        # Fusion Classifier
        # =========================
        self.combined_dim = cnn_in_features + vit_in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.combined_dim, num_classes)
        )

    def forward(self, x):

        # CNN features
        cnn_features = self.cnn(x)

        # ViT features
        vit_features = self.vit(x)

        # Feature fusion
        combined_features = torch.cat(
            [cnn_features, vit_features],
            dim=1
        )

        # Classification
        logits = self.classifier(combined_features)

        return logits


def get_hybrid_model(
    num_classes=6,
    pretrained=True,
    dropout_p=0.3
):

    return HybridCNNViT(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p
    )


if __name__ == "__main__":

    print("=" * 70)
    print("FULLY FINE-TUNED CNN + ViT HYBRID MODEL")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model = get_hybrid_model(
        num_classes=6,
        pretrained=True
    ).to(device)

    cnn_params = sum(
        p.numel() for p in model.cnn.parameters()
    )

    vit_params = sum(
        p.numel() for p in model.vit.parameters()
    )

    classifier_params = sum(
        p.numel() for p in model.classifier.parameters()
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(f"CNN parameters:          {cnn_params:,}")
    print(f"ViT parameters:          {vit_params:,}")
    print(f"Classifier parameters:   {classifier_params:,}")
    print(f"Total parameters:        {total_params:,}")
    print(f"Trainable parameters:    {trainable_params:,}")

    print()
    print("Expected:")
    print("All CNN parameters      -> TRAINABLE")
    print("All ViT parameters      -> TRAINABLE")
    print("Classifier parameters   -> TRAINABLE")

    # Safe verification batch for 6 GB GPU
    dummy_input = torch.randn(
        1,
        3,
        224,
        224,
        device=device
    )

    with torch.no_grad():
        output = model(dummy_input)

    print()
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (1, 6)

    assert trainable_params == total_params

    print()
    print("FULL FINE-TUNED MODEL VERIFICATION PASSED!")
    print("=" * 70)