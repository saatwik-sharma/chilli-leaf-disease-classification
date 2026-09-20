import torch
import torch.nn as nn
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights
)


class HybridCNNViT(nn.Module):
    """
    Hybrid CNN + Transformer architecture.

    ResNet-50 extracts 2048-dimensional CNN features.
    ViT-B/16 extracts 768-dimensional transformer features.
    Features are concatenated into 2816 dimensions.
    Only the fusion classifier is trained initially.
    """

    def __init__(self, num_classes=6, pretrained=True, dropout_p=0.3):
        super().__init__()

        # -------------------------------------------------
        # 1. ResNet-50 CNN feature extractor
        # -------------------------------------------------
        resnet_weights = ResNet50_Weights.DEFAULT if pretrained else None

        self.cnn = resnet50(weights=resnet_weights)

        cnn_in_features = self.cnn.fc.in_features  # 2048

        self.cnn.fc = nn.Identity()

        # Freeze ResNet-50
        for param in self.cnn.parameters():
            param.requires_grad = False

        # -------------------------------------------------
        # 2. ViT-B/16 Transformer feature extractor
        # -------------------------------------------------
        vit_weights = ViT_B_16_Weights.DEFAULT if pretrained else None

        self.vit = vit_b_16(weights=vit_weights)

        vit_in_features = self.vit.heads.head.in_features  # 768

        self.vit.heads = nn.Identity()

        # Freeze ViT-B/16
        for param in self.vit.parameters():
            param.requires_grad = False

        # -------------------------------------------------
        # 3. Feature fusion
        # -------------------------------------------------
        self.combined_dim = cnn_in_features + vit_in_features  # 2816

        # -------------------------------------------------
        # 4. Trainable fusion classifier
        # -------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.combined_dim, num_classes)
        )

    def forward(self, x):

        # Frozen CNN feature extraction
        with torch.no_grad():
            cnn_features = self.cnn(x)

        # Frozen Transformer feature extraction
        with torch.no_grad():
            vit_features = self.vit(x)

        # Concatenate CNN + ViT features
        combined_features = torch.cat(
            [cnn_features, vit_features],
            dim=1
        )

        # Trainable classifier
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

    print("=" * 65)
    print("Verifying Hybrid CNN (ResNet-50) + ViT (ViT-B/16) Model...")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    model = get_hybrid_model(
        num_classes=6,
        pretrained=True
    ).to(device)

    # Parameter counts
    cnn_params = sum(
        p.numel() for p in model.cnn.parameters()
    )

    vit_params = sum(
        p.numel() for p in model.vit.parameters()
    )

    head_params = sum(
        p.numel() for p in model.classifier.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"CNN (ResNet-50) parameters:   {cnn_params:,}")
    print(f"ViT (ViT-B/16) parameters:   {vit_params:,}")
    print(f"Classifier parameters:        {head_params:,}")
    print(f"Total parameters:             {total_params:,}")
    print(f"Trainable parameters:         {trainable_params:,}")

    # Dummy forward pass
    batch_size = 2

    dummy_input = torch.randn(
        batch_size,
        3,
        224,
        224,
        device=device
    )

    print(
        f"\nRunning forward pass with "
        f"dummy batch {dummy_input.shape}..."
    )

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")

    assert output.shape == (
        batch_size,
        6
    )

    print("\nHybrid model verification passed successfully!")