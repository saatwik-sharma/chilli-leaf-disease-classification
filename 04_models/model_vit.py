import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def get_vit_model(num_classes=6, pretrained=True):
    """
    Constructs a Vision Transformer (ViT-B/16) baseline with pretrained weights (ImageNet-1K)
    and replaces the final classification head for 6-class chilli leaf disease classification.
    """
    if pretrained:
        weights = ViT_B_16_Weights.DEFAULT
    else:
        weights = None
        
    model = vit_b_16(weights=weights)
    
    # In torchvision VisionTransformer, the classification head is located in `model.heads.head`
    # model.heads is nn.Sequential(nn.Linear(in_features=768, out_features=1000, bias=True))
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    print("Testing ViT-B/16 model definition and forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = get_vit_model(num_classes=6, pretrained=True).to(device)
    dummy_input = torch.randn(4, 3, 224, 224, device=device)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 6), f"Expected shape (4, 6), got {output.shape}"
    print("ViT-B/16 forward pass verification test passed successfully!")
