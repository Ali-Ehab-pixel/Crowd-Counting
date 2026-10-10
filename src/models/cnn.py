import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class PeopleCountingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the deeper ResNet50 backbone for advanced feature extraction
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # ResNet50 outputs 2048 features (compared to ResNet18's 512)
        input_features = self.backbone.fc.in_features
        
        # Map the 2048 features down to a single continuous regression value
        self.backbone.fc = nn.Linear(input_features, 1)

    def forward(self, x):
        # Squeeze removes the extra dimension: [batch_size, 1] -> [batch_size]
        return self.backbone(x).squeeze(1)