import torch
import torch.nn as nn

class Multimodal3DModel(nn.Module):
    def __init__(self, clinical_dim=5, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc_clinical = nn.Linear(clinical_dim, 32)
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, volume, clinical):
        x1 = self.cnn(volume).flatten(1)
        x2 = torch.relu(self.fc_clinical(clinical))
        x = torch.cat([x1, x2], dim=1)
        return self.classifier(x)