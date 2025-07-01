import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ResNetClassifier:
    def __init__(self, weights_path=None, device='cpu', num_classes=2):
        self.device = device
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, patch_img):
        # patch_img: PIL Image
        x = self.transform(patch_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        return prob 