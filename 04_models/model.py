import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=6, pretrained=True):
    """
    ResNet-50 CNN baseline.
    """
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_vit_model(num_classes=6, pretrained=True):
    """
    ViT-B/16 standalone baseline.
    """
    if pretrained:
        weights = models.ViT_B_16_Weights.DEFAULT
    else:
        weights = None

    model = models.vit_b_16(weights=weights)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    return model


class HybridCNNViT(nn.Module):
    """
    Hybrid CNN + Vision Transformer model.

    ResNet-50 extracts local CNN features.
    ViT-B/16 extracts global transformer features.
    Their feature representations are concatenated and passed
    through a small fusion classifier.
    """

    def __init__(self, num_classes=6, pretrained=True, freeze_backbones=True):

        super().__init__()

        # --------------------------------------------------
        # ResNet-50 CNN branch
        # --------------------------------------------------
        if pretrained:
            resnet_weights = models.ResNet50_Weights.DEFAULT
        else:
            resnet_weights = None

        self.cnn = models.resnet50(weights=resnet_weights)

        # Remove ResNet classification layer
        self.cnn.fc = nn.Identity()

        # ResNet-50 feature size = 2048
        cnn_features = 2048

        # --------------------------------------------------
        # ViT-B/16 Transformer branch
        # --------------------------------------------------
        if pretrained:
            vit_weights = models.ViT_B_16_Weights.DEFAULT
        else:
            vit_weights = None

        self.vit = models.vit_b_16(weights=vit_weights)

        # Remove ViT classification head
        self.vit.heads.head = nn.Identity()

        # ViT-B/16 feature size = 768
        vit_features = 768

        # --------------------------------------------------
        # Optional freezing
        # --------------------------------------------------
        if freeze_backbones:
            for param in self.cnn.parameters():
                param.requires_grad = False

            for param in self.vit.parameters():
                param.requires_grad = False

        # --------------------------------------------------
        # Fusion classifier
        # --------------------------------------------------
        combined_features = cnn_features + vit_features

        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        # CNN features
        cnn_features = self.cnn(x)

        # ViT features
        vit_features = self.vit(x)

        # Feature fusion
        combined = torch.cat(
            [cnn_features, vit_features],
            dim=1
        )

        # Classification
        output = self.fusion(combined)

        return output


if __name__ == "__main__":

    print("Testing ResNet-50...")
    model_resnet = get_model(num_classes=6, pretrained=False)

    dummy_input = torch.randn(2, 3, 224, 224)

    output_resnet = model_resnet(dummy_input)

    assert output_resnet.shape == (2, 6)
    print("ResNet-50 test passed.")


    print("\nTesting ViT-B/16...")
    model_vit = get_vit_model(num_classes=6, pretrained=False)

    output_vit = model_vit(dummy_input)

    assert output_vit.shape == (2, 6)
    print("ViT-B/16 test passed.")


    print("\nTesting Hybrid CNN + ViT...")

    model_hybrid = HybridCNNViT(
        num_classes=6,
        pretrained=False,
        freeze_backbones=True
    )

    output_hybrid = model_hybrid(dummy_input)

    assert output_hybrid.shape == (2, 6)

    print("Hybrid CNN + ViT test passed.")
    print("\nAll model tests passed.")
