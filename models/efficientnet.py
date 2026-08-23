import torch
import torch.nn as nn
import timm

class EfficientNetB3Medical(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = False, dropout: float = 0.3):
        super(EfficientNetB3Medical, self).__init__()
        # Load EfficientNet-B3 from timm backbone
        self.backbone = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
