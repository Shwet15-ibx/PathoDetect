import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b4
from PIL import Image

class EfficientNetClassifier:
    def __init__(self, weights_path=None, device='cpu', num_classes=2):
        self.device = device
        self.model = efficientnet_b4(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
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