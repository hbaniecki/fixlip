import numpy as np
from PIL import Image


def perturbar_regiao(
    image_pil: Image.Image,
    mask_array: np.ndarray,
    region_id: int,
    method: str = "zero",
) -> Image.Image:
    """Replace all pixels belonging to a region ID with a black or mean fill."""
    if method not in {"zero", "mean"}:
        raise ValueError("method must be 'zero' or 'mean'")

    mask = np.asarray(mask_array)
    if mask.ndim != 2:
        raise ValueError("mask_array must be a two-dimensional numpy array")

    image = image_pil.convert("RGB")
    if mask.shape != (image.height, image.width):
        raise ValueError(
            "mask_array dimensions must match the image dimensions "
            f"({image.height}x{image.width})"
        )

    perturbed = np.asarray(image).copy()
    region_pixels = mask == region_id

    if method == "zero":
        fill = np.array([0, 0, 0], dtype=np.uint8)
    else:
        fill = np.mean(perturbed, axis=(0, 1)).astype(np.uint8)

    perturbed[region_pixels] = fill
    return Image.fromarray(perturbed, mode="RGB")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dummy_image = Image.fromarray(
        np.full((224, 224, 3), 180, dtype=np.uint8),
        mode="RGB",
    )
    dummy_mask = np.zeros((224, 224), dtype=np.int32)
    dummy_mask[56:168, 56:168] = 1

    perturbed = perturbar_regiao(dummy_image, dummy_mask, region_id=1, method="zero")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(dummy_image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(perturbed)
    axes[1].set_title("Region 1 perturbed")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig("perturbation_test.png", dpi=150, bbox_inches="tight")
    print("Saved perturbation_test.png")
