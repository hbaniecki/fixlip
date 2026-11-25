from collections.abc import Sequence
from typing import Union

import numpy as np
import torch
from PIL import Image

from .attribution import _load_clip


def compute_mask_similarity(
    image_pil: Image.Image,
    text: str,
    mask_array: np.ndarray,
    visible_regions: Sequence[int],
    device: Union[str, torch.device] = "cpu",
) -> float:
    """Return CLIP cosine similarity for an image with only selected regions visible."""
    device_name = str(device)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device '{device_name}', but CUDA is unavailable")

    image = image_pil.convert("RGB")
    mask = np.asarray(mask_array)
    if mask.ndim != 2:
        raise ValueError("mask_array must be a two-dimensional numpy array")
    if mask.shape != (image.height, image.width):
        raise ValueError(
            "mask_array dimensions must match the image dimensions "
            f"({image.height}x{image.width})"
        )

    visible_mask = np.isin(mask, tuple(int(region_id) for region_id in visible_regions))
    masked_pixels = np.asarray(image).copy()
    masked_pixels[~visible_mask] = 0
    masked_image = Image.fromarray(masked_pixels)

    model, processor = _load_clip(device_name)
    inputs = processor(images=masked_image, text=text, return_tensors="pt")
    inputs = {name: value.to(device_name) for name, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        similarity = outputs.logits_per_image[0, 0] / model.logit_scale.exp()

    return float(similarity.item())


def predict_mask_similarity(
    phi_base: np.ndarray,
    M: np.ndarray,
    visible_regions: Sequence[int],
    phi_0: float = 0.0,
) -> float:
    """Predict mask similarity with the requested additive attribution approximation."""
    phi = np.asarray(phi_base, dtype=np.float64)
    matrix = np.asarray(M, dtype=np.float64)
    if phi.ndim != 1:
        raise ValueError("phi_base must be a one-dimensional array")
    if matrix.shape != (phi.size, phi.size):
        raise ValueError("M must be a square matrix matching the size of phi_base")

    region_ids = tuple(dict.fromkeys(int(region_id) for region_id in visible_regions))
    if any(region_id < 1 or region_id > phi.size for region_id in region_ids):
        raise ValueError("visible region IDs must be between 1 and len(phi_base)")

    indices = np.asarray([region_id - 1 for region_id in region_ids], dtype=int)
    regional_sum = float(phi[indices].sum()) if indices.size else 0.0
    interaction_sum = (
        float(matrix[np.ix_(indices, indices)].sum()) if indices.size else 0.0
    )
    return float(phi_0 + regional_sum + interaction_sum)


def quadratic_difference(v_real: float, v_pred: float) -> float:
    """Return the squared difference between real and predicted mask similarity."""
    return float((v_real - v_pred) ** 2)
