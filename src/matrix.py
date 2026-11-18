from typing import Union

import numpy as np
import torch
from PIL import Image

from .attribution import calcular_phi_base
from .perturbation import perturbar_regiao


def compute_directional_matrix(
    image_pil: Image.Image,
    text: str,
    mask_array: np.ndarray,
    n_regioes: int,
    device: Union[str, torch.device] = "cpu",
) -> np.ndarray:
    """Compute regional importance changes caused by perturbing each region."""
    if n_regioes < 0:
        raise ValueError("n_regioes must be non-negative")

    phi_base = calcular_phi_base(
        image_pil,
        text,
        mask_array,
        n_regioes,
        device=device,
    )
    directional_matrix = np.zeros((n_regioes, n_regioes), dtype=np.float32)

    for region_id in range(1, n_regioes + 1):
        image_perturbed = perturbar_regiao(
            image_pil,
            mask_array,
            region_id=region_id,
            method="zero",
        )
        phi_perturbed = calcular_phi_base(
            image_perturbed,
            text,
            mask_array,
            n_regioes,
            device=device,
        )
        directional_matrix[:, region_id - 1] = phi_base - phi_perturbed

    return directional_matrix
