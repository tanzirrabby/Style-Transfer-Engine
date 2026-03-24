from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(-1, 1, 1)

STYLE_LAYERS = ["0", "5", "10", "19", "28"]
CONTENT_LAYERS = ["21"]


def load_image(path: Path, image_size: int = 512) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor


def save_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu().squeeze(0)
    image = image.clamp(0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(image)


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    b, c, h, w = features.shape
    feature_map = features.view(b * c, h * w)
    gram = feature_map @ feature_map.t()
    return gram / (b * c * h * w)


def extract_features(x: torch.Tensor, model: nn.Module) -> dict[str, torch.Tensor]:
    features: dict[str, torch.Tensor] = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in STYLE_LAYERS or name in CONTENT_LAYERS:
            features[name] = x
    return features


def run_style_transfer(
    content_image: Path,
    style_image: Path,
    style_weight: float = 1e6,
    content_weight: float = 1.0,
    num_steps: int = 200,
    image_size: int = 512,
) -> Image.Image:
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
    for param in vgg.parameters():
        param.requires_grad_(False)

    content = load_image(content_image, image_size=image_size)
    style = load_image(style_image, image_size=image_size)

    generated = content.clone().requires_grad_(True)

    content_features = extract_features(normalize(content), vgg)
    style_features = extract_features(normalize(style), vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in STYLE_LAYERS}

    optimizer = torch.optim.Adam([generated], lr=0.02)

    for _ in range(max(num_steps, 1)):
        gen_features = extract_features(normalize(generated), vgg)

        content_loss = F.mse_loss(gen_features[CONTENT_LAYERS[0]], content_features[CONTENT_LAYERS[0]])

        style_loss = 0.0
        for layer in STYLE_LAYERS:
            gen_gram = gram_matrix(gen_features[layer])
            style_loss += F.mse_loss(gen_gram, style_grams[layer])

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            generated.clamp_(0, 1)

    return save_image(generated)
