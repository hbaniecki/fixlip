from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import json
from typing import Any, Optional


def load_image_and_mask(
    image_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]],
    target_size: int = 224,
) -> tuple[Image.Image, np.ndarray]:
    """Load an image and region mask, resizing both to a square target size."""
    if target_size <= 0:
        raise ValueError("target_size must be a positive integer")

    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB").resize(
            (target_size, target_size),
            resample=Image.Resampling.BILINEAR,
        )

    if mask_path is None:
        mask = np.zeros((target_size, target_size), dtype=np.int32)
    else:
        with Image.open(mask_path) as mask_file:
            mask = np.asarray(
                mask_file.resize(
                    (target_size, target_size),
                    resample=Image.Resampling.NEAREST,
                ),
                dtype=np.int32,
            )

    return image, mask


def load_mask_from_coco(
    json_path: Union[str, Path],
    image_size: tuple[int, int] = (224, 224),
    n_regions: Optional[int] = None,
    source_image_size: Optional[tuple[int, int]] = None,
) -> tuple[np.ndarray, int]:
    """Convert COCO polygon annotations into a nearest-neighbor region mask."""
    if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size must contain positive width and height")

    with open(json_path, "r", encoding="utf-8") as json_file:
        coco = json.load(json_file)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    source_width = source_height = None
    if source_image_size is not None:
        source_width, source_height = source_image_size
        matching_images = [
            item
            for item in images
            if item.get("width") == source_width
            and item.get("height") == source_height
        ]
        if len(matching_images) == 1:
            selected_image_id = matching_images[0].get("id")
            annotations = [
                annotation
                for annotation in annotations
                if annotation.get("image_id") == selected_image_id
            ]
    elif len(images) == 1:
        source_width = images[0].get("width")
        source_height = images[0].get("height")
    elif images:
        annotation_counts = {}
        for annotation in annotations:
            image_id = annotation.get("image_id")
            if image_id is not None:
                annotation_counts[image_id] = annotation_counts.get(image_id, 0) + 1
        if annotation_counts:
            selected_image_id = max(annotation_counts, key=annotation_counts.get)
            selected_image = next(
                (item for item in images if item.get("id") == selected_image_id),
                None,
            )
            if selected_image is not None:
                source_width = selected_image.get("width")
                source_height = selected_image.get("height")
                annotations = [
                    annotation
                    for annotation in annotations
                    if annotation.get("image_id") == selected_image_id
                ]

    if source_width is None or source_height is None:
        source_width, source_height = image_size

    mask_image = Image.new("I", (source_width, source_height), 0)
    draw = ImageDraw.Draw(mask_image)
    sequential_id = 1
    region_ids: set[int] = set()

    for annotation in annotations:
        if "image_id" not in annotation and len(images) > 1:
            continue

        category_id = annotation.get("category_id")
        region_id = int(category_id) if category_id is not None else sequential_id
        sequential_id += 1

        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            continue

        polygons = segmentation if segmentation and isinstance(segmentation[0], list) else [segmentation]
        for polygon in polygons:
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 6:
                continue
            points = [
                (float(polygon[index]), float(polygon[index + 1]))
                for index in range(0, len(polygon) - 1, 2)
            ]
            draw.polygon(points, fill=region_id)
            region_ids.add(region_id)

    resized_mask = mask_image.resize(
        image_size,
        resample=Image.Resampling.NEAREST,
    )
    mask_array = np.asarray(resized_mask, dtype=np.int32)

    if n_regions is not None:
        number_of_regions = n_regions
    else:
        number_of_regions = len(region_ids)

    return mask_array, number_of_regions


def visualize_alignment(
    image_pil: Image.Image,
    mask_array: np.ndarray,
    save_path: Union[str, Path] = "alignment_check.png",
) -> plt.Figure:
    """Save and return a side-by-side original, mask, and overlay plot."""
    mask = np.asarray(mask_array)
    if mask.ndim != 2:
        raise ValueError("mask_array must be a two-dimensional numpy array")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_pil)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="tab20")
    axes[1].set_title("Region mask")
    axes[1].axis("off")

    axes[2].imshow(image_pil)
    axes[2].imshow(mask, cmap="tab20", alpha=0.45)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.tight_layout()
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
