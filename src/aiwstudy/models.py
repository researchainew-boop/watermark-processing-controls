from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .io import read_image


class FrozenResNet18(nn.Module):
    """Frozen ImageNet ResNet-18 embedding extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        self.features = nn.Sequential(*list(model.children())[:-1])
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> np.ndarray:
        vectors = self.features(images).flatten(1)
        return vectors.cpu().numpy()


def load_clip(device: str = "cpu"):
    try:
        import open_clip
    except ImportError as error:
        raise RuntimeError("Install the optional [clip] extra to use a CLIP backbone.") from error
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess


def image_to_tensor(image: np.ndarray, size: int = 224) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(image, (size, size)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


@torch.no_grad()
def extract_embeddings(
    manifest: pd.DataFrame,
    backbone: str,
    pretrained: bool,
    device_name: str,
    batch_size: int,
) -> pd.DataFrame:
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    records = []
    if backbone == "resnet18":
        model = FrozenResNet18(pretrained=pretrained).to(device)
        tensors = []
        ids = []
        labels = []
        for row in manifest.to_dict("records"):
            tensors.append(image_to_tensor(read_image(row["path"])))
            ids.append(row["image_id"])
            labels.append(int(row["label"]))
            if len(tensors) == batch_size:
                batch = torch.stack(tensors).to(device)
                embeddings = model.embed(batch)
                for image_id, label, vector in zip(ids, labels, embeddings):
                    records.append({"image_id": image_id, "label": label, "embedding": vector})
                tensors, ids, labels = [], [], []
        if tensors:
            batch = torch.stack(tensors).to(device)
            embeddings = model.embed(batch)
            for image_id, label, vector in zip(ids, labels, embeddings):
                records.append({"image_id": image_id, "label": label, "embedding": vector})
        return pd.DataFrame(records)
    if backbone == "clip":
        model, preprocess = load_clip(str(device))
        from PIL import Image

        tensors, ids, labels = [], [], []
        for row in manifest.to_dict("records"):
            image = Image.open(row["path"]).convert("RGB")
            tensors.append(preprocess(image))
            ids.append(row["image_id"])
            labels.append(int(row["label"]))
            if len(tensors) == batch_size:
                batch = torch.stack(tensors).to(device)
                vectors = model.encode_image(batch).cpu().numpy()
                for image_id, label, vector in zip(ids, labels, vectors):
                    records.append({"image_id": image_id, "label": label, "embedding": vector})
                tensors, ids, labels = [], [], []
        if tensors:
            batch = torch.stack(tensors).to(device)
            vectors = model.encode_image(batch).cpu().numpy()
            for image_id, label, vector in zip(ids, labels, vectors):
                records.append({"image_id": image_id, "label": label, "embedding": vector})
        return pd.DataFrame(records)
    raise ValueError(f"Unknown backbone: {backbone}")


@dataclass
class LinearProbe:
    model: LogisticRegression

    @classmethod
    def fit(cls, embeddings: np.ndarray, labels: np.ndarray) -> LinearProbe:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
        model.fit(embeddings, labels)
        return cls(model)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(embeddings)[:, 1]

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            coefficients=self.model.coef_.astype(np.float64),
            intercept=self.model.intercept_.astype(np.float64),
            classes=self.model.classes_,
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearProbe:
        values = np.load(path)
        model = LogisticRegression()
        model.classes_ = values["classes"]
        model.coef_ = values["coefficients"]
        model.intercept_ = values["intercept"]
        model.n_features_in_ = model.coef_.shape[1]
        return cls(model)


@dataclass
class FusionCalibrator:
    model: LogisticRegression

    @classmethod
    def fit(
        cls, passive_scores: np.ndarray, active_scores: np.ndarray, labels: np.ndarray
    ) -> FusionCalibrator:
        features = np.column_stack((passive_scores, active_scores))
        model = LogisticRegression(random_state=0, class_weight="balanced")
        model.fit(features, labels)
        return cls(model)

    def predict_proba(self, passive_scores: np.ndarray, active_scores: np.ndarray) -> np.ndarray:
        features = np.column_stack((passive_scores, active_scores))
        return self.model.predict_proba(features)[:, 1]

    def save(self, path: str | Path) -> None:
        np.savez(
            path,
            coefficients=self.model.coef_.astype(np.float64),
            intercept=self.model.intercept_.astype(np.float64),
            classes=self.model.classes_,
        )

    @classmethod
    def load(cls, path: str | Path) -> FusionCalibrator:
        values = np.load(path)
        model = LogisticRegression()
        model.classes_ = values["classes"]
        model.coef_ = values["coefficients"]
        model.intercept_ = values["intercept"]
        model.n_features_in_ = model.coef_.shape[1]
        return cls(model)
