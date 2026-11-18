from functools import lru_cache
import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


class _CLIPSimilarityModel(torch.nn.Module):
    """Expose one image-text CLIP similarity score for Grad-CAM."""

    def __init__(
        self,
        model: CLIPModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=self.input_ids.expand(pixel_values.shape[0], -1),
            attention_mask=(
                self.attention_mask.expand(pixel_values.shape[0], -1)
                if self.attention_mask is not None
                else None
            ),
        )
        return outputs.logits_per_image


@lru_cache(maxsize=4)
def _load_clip(device: str) -> tuple[CLIPModel, CLIPProcessor]:
    """Load and cache the CLIP model and processor for a device."""
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def _reshape_clip_tokens(activation: torch.Tensor) -> torch.Tensor:
    """Convert CLIP's token activations into a CNN-style feature map."""
    if activation.ndim != 3:
        raise RuntimeError(
            "Expected CLIP vision activations with shape (batch, tokens, channels)"
        )

    patch_tokens = activation[:, 1:, :]
    grid_size = math.isqrt(patch_tokens.shape[1])
    if grid_size * grid_size != patch_tokens.shape[1]:
        raise RuntimeError("CLIP patch-token count does not form a square grid")

    return patch_tokens.reshape(
        patch_tokens.shape[0],
        grid_size,
        grid_size,
        patch_tokens.shape[2],
    ).permute(0, 3, 1, 2)


def _manual_activation_gradcam(
    similarity_model: _CLIPSimilarityModel,
    target_layer: torch.nn.Module,
    pixel_values: torch.Tensor,
) -> np.ndarray:
    """Compute a non-ReLU ViT Grad-CAM when standard Grad-CAM is empty."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_activation(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        activations.append(output)

    def save_gradient(
        _module: torch.nn.Module,
        _grad_inputs: tuple[torch.Tensor | None, ...],
        grad_outputs: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_outputs[0] is not None:
            gradients.append(grad_outputs[0])

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)
    try:
        similarity_model.zero_grad(set_to_none=True)
        output = similarity_model(pixel_values)
        output[:, 0].sum().backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    if not activations or not gradients:
        raise RuntimeError("Could not capture CLIP vision activations and gradients")

    activation = activations[-1]
    gradient = gradients[-1]
    weights = gradient.detach().abs().mean(dim=1, keepdim=True)
    token_map = (activation.detach().abs() * weights).sum(dim=-1)[:, 1:]
    grid_size = math.isqrt(token_map.shape[1])
    token_map = token_map.reshape(1, 1, grid_size, grid_size)
    attribution = F.interpolate(
        token_map,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )[0, 0].cpu().numpy()
    return attribution


def get_attribution_map(
    image_pil: Image.Image,
    text: str,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Return a smooth ViT Grad-CAM map for an image-text pair."""
    device_name = str(device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device_name}', but CUDA is unavailable")

    image = image_pil.convert("RGB").resize(
        (224, 224),
        resample=Image.Resampling.BILINEAR,
    )
    model, processor = _load_clip(device_name)
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
    )
    inputs = {name: value.to(device_name) for name, value in inputs.items()}

    similarity_model = _CLIPSimilarityModel(
        model,
        inputs["input_ids"],
        inputs.get("attention_mask"),
    )
    target_layer = model.vision_model.encoder.layers[-1]

    with GradCAM(
        model=similarity_model,
        target_layers=[target_layer],
        reshape_transform=_reshape_clip_tokens,
    ) as cam:
        attribution = cam(
            input_tensor=inputs["pixel_values"],
            targets=[ClassifierOutputTarget(0)],
        )[0]

    if float(attribution.max()) == 0.0:
        attribution = _manual_activation_gradcam(
            similarity_model,
            target_layer,
            inputs["pixel_values"],
        )

    attribution_min = float(attribution.min())
    attribution_max = float(attribution.max())
    if attribution_max > attribution_min:
        attribution = (attribution - attribution_min) / (
            attribution_max - attribution_min
        )
    else:
        attribution = np.zeros_like(attribution)

    return attribution.astype(np.float32)


def calcular_phi_base(
    image_pil: Image.Image,
    text: str,
    mask_array: np.ndarray,
    n_regioes: int,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Compute mean normalized saliency for each semantic region."""
    if n_regioes < 0:
        raise ValueError("n_regioes must be non-negative")

    mask = np.asarray(mask_array)
    if mask.ndim != 2:
        raise ValueError("mask_array must be a two-dimensional numpy array")
    if mask.shape != (224, 224):
        raise ValueError("mask_array must have shape (224, 224)")

    attribution_map = get_attribution_map(image_pil, text, device=device)
    phi_base = np.zeros(n_regioes, dtype=np.float32)

    for region_id in range(1, n_regioes + 1):
        region_pixels = mask == region_id
        if np.any(region_pixels):
            phi_base[region_id - 1] = float(attribution_map[region_pixels].mean())

    return phi_base
