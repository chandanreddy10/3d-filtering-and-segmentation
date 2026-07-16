from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib
from transformers import Sam3Processor, Sam3Model


def load_sam3(device="cuda"):
    """
    Load SAM3 model and processor.
    """
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    model.eval()

    return model, processor


def segment_images(
    image_paths,
    model,
    processor,
    device="cuda",
    prompt="Objects",
    threshold=0.5,
    mask_threshold=0.5,
    save_dir=None,
):
    """
    Segment a batch of images using SAM3.

    Args:
        image_paths: list[Path]
        model: SAM3 model
        processor: SAM3 processor
        device: cuda/cpu
        prompt: text prompt
        threshold: detection threshold
        mask_threshold: mask threshold
        save_dir: optional directory to save masks

    Returns:
        dict:
            {
              image_path: masks
            }
    """

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    results_dict = {}

    for image_path in image_paths:

        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        masks = results["masks"]

        print(f"{image_path.name}: Found {len(masks)} objects")

        results_dict[image_path] = masks.cpu()

        #save masks
        if save_dir:

            mask_array = masks.cpu().numpy()

            for i, mask in enumerate(mask_array):

                mask_img = Image.fromarray((mask * 255).astype(np.uint8))

                mask_img.save(save_dir / f"{image_path.stem}_mask_{i}.png")

    return results_dict


def overlay_masks(image, masks, save_path=None):
    """
    Overlay segmentation masks on image.

    Args:
        image: PIL Image
        masks: torch tensor [N,H,W]
        save_path: optional output path

    Returns:
        PIL Image
    """

    image = image.convert("RGBA")

    masks = 255 * masks.numpy().astype(np.uint8)

    n_masks = masks.shape[0]

    if n_masks == 0:
        return image

    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)

    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks, colors):

        mask_img = Image.fromarray(mask)

        overlay = Image.new("RGBA", image.size, color + (0,))

        alpha = mask_img.point(lambda v: int(v * 0.5))

        overlay.putalpha(alpha)

        image = Image.alpha_composite(image, overlay)

    if save_path:
        image.save(save_path)

    return image
