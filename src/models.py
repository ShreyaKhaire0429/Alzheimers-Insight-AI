import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvmodels

# 🔹 CNN BASE (Feature extractor)
def build_cnn(pretrained=True, output_dim=512):
    """
    Builds a ResNet18 base that returns a feature vector of size output_dim.
    """
    base = tvmodels.resnet18(weights=tvmodels.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = base.fc.in_features
    base.fc = nn.Identity()  # remove classification layer
    proj = nn.Linear(in_features, output_dim) if in_features != output_dim else None

    class CNNWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = base
            self.proj = proj

        def forward(self, x):
            feats = self.backbone(x)
            if self.proj is not None:
                feats = self.proj(feats)
            return feats

    return CNNWrapper()


# 🔹 MULTIMODAL MODEL (CNN + Clinical Data)
class MultimodalModel(nn.Module):
    def __init__(self, cnn_feature_dim=512, clinical_dim=3, fused_dim=256, num_classes=3):
        """
        Combines CNN features from MRI with tabular clinical data.
        """
        super(MultimodalModel, self).__init__()
        # CNN branch
        self.cnn = build_cnn(pretrained=True, output_dim=cnn_feature_dim)

        # Clinical branch
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Fusion + classifier
        self.fc = nn.Sequential(
            nn.Linear(cnn_feature_dim + 64, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, num_classes)
        )

    def forward(self, image, clinical):
        cnn_out = self.cnn(image)
        clin_out = self.clinical_fc(clinical)
        combined = torch.cat((cnn_out, clin_out), dim=1)
        out = self.fc(combined)
        return out


# SIMPLE CNN FOR MRI ONLY (Fallback Model)
class AlzheimerNet(nn.Module):
    def __init__(self, num_classes=3):
        super(AlzheimerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # adjust for input size
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# BUILDERS AND HELPERS
def build_multimodal_model(clinical_dim=3, num_classes=3):
    return MultimodalModel(
        cnn_feature_dim=512,
        clinical_dim=clinical_dim,
        fused_dim=256,
        num_classes=num_classes
    )

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model_class_fn, model_path, device=None):
    """
    model_class_fn: callable e.g. lambda: build_multimodal_model(clinical_dim=3, num_classes=3)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class_fn().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

import torch
import torch.nn as nn
from torchvision import models

class MultimodalModel(nn.Module):
    def __init__(self, num_clinical_features=5, num_classes=3, pretrained=True):
        super(MultimodalModel, self).__init__()
        
        #  CNN Branch (MRI) 
        backbone = models.resnet18(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # Remove FC
        self.cnn_fc = nn.Linear(backbone.fc.in_features, 512)  # Feature extractor
        
        #  MLP Branch (Clinical) 
        self.mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        #  Fusion + Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, mri, clinical):
        # MRI path
        x1 = self.cnn(mri)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.cnn_fc(x1)
        
        # Clinical path
        x2 = self.mlp(clinical)
        
        # Fusion
        fused = torch.cat((x1, x2), dim=1)
        return self.classifier(fused)