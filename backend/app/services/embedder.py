"""Pretrained identity embedder (no training). Muzzle crop -> L2-normalised vector.
Grayscale + CLAHE so coat colour does not dominate. DINOv2 vector = [CLS | patch mean | 4 quadrant means]."""
from typing import List

import cv2
import numpy as np

from app.core.config import settings
from app.services.population import Population, set_active

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr: np.ndarray, size: int = settings.EMBED_SIZE) -> np.ndarray:
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    img = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32) / 255.0
    return np.ascontiguousarray(((img - _MEAN) / _STD).transpose(2, 0, 1))


class PretrainedEmbedder:
    def __init__(self, name: str = settings.EMBEDDER, device: str = None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kind, self.name = "resnet", name
        self.net = self._load(name)
        self.net.to(self.device).eval()
        self.pop = Population(self.name)
        set_active(self.pop)

    def _load(self, name: str):
        torch = self.torch
        if name.startswith("dinov2"):
            try:
                net = torch.hub.load("facebookresearch/dinov2", name)
                self.kind, self.name = "dinov2", f"{name}_gridgray"
                return net
            except Exception as e:
                print(f"[embedder] DINOv2 unavailable ({e}); falling back to ResNet50")
        import torchvision
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = torch.nn.Identity()
        self.kind, self.name = "resnet", "resnet50_gray"
        return net

    def embed_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 1), dtype=np.float32)
        torch = self.torch
        F = torch.nn.functional
        x = torch.from_numpy(np.stack([preprocess(c) for c in crops])).to(self.device)
        with torch.no_grad():
            if self.kind == "dinov2":
                o = self.net.forward_features(x)
                cls = F.normalize(o["x_norm_clstoken"], dim=1)
                pt = o["x_norm_patchtokens"]
                B, N, D = pt.shape
                g = int(round(N ** 0.5))
                pat = F.normalize(pt.mean(1), dim=1)
                grid = pt.reshape(B, g, g, D)
                h = g // 2
                quads = [grid[:, :h, :h], grid[:, :h, h:], grid[:, h:, :h], grid[:, h:, h:]]
                f = torch.cat([cls, pat] + [F.normalize(q.mean(dim=(1, 2)), dim=1) for q in quads], dim=1)
            else:
                f = self.net(x)
        f = f.cpu().numpy().astype(np.float32)
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
        self.pop.update(f.mean(axis=0))
        return f
