from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def _make_model(num_classes):
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class MobileNetRoleClassifier:
    def __init__(self, weights_path, device=None):
        self.weights_path = Path(weights_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            self.weights_path,
            map_location=self.device,
            weights_only=False,
        )

        self.idx_to_class = {
            int(idx): label for idx, label in checkpoint["idx_to_class"].items()
        }
        num_classes = len(self.idx_to_class)
        self.model = _make_model(num_classes)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()

        img_size = int(checkpoint.get("cfg", {}).get("train", {}).get("img_size", 224))
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def predict_crop(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return "unknown", 0.0

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, dim=0)
        label = self.idx_to_class[int(pred_idx.item())]
        return label, float(confidence.item())
