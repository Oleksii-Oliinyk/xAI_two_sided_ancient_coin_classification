from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.xai.core.base import denormalize_gray


def overlay_heatmap(image, explanation, signed):
    """
    image:       PIL RGB image of any size (the map is stretched back to it, same as Resize squashed it to 224)
    explanation: map [H, W] from an explainer
    signed:      True -> bwr around 0 (red supports, blue against), False -> jet on 0..1
    Returns a PIL RGB image of the same size as `image` with the heatmap blended on top.
    """
    from PIL import Image

    width, height = image.size
    heat = Image.fromarray(np.asarray(explanation, dtype=np.float32)).resize((width, height), Image.BILINEAR)
    heat = np.asarray(heat)

    if signed:
        # same scaling as save_explanations_figure, alpha follows |value| so neutral pixels stay unpainted
        limit = np.percentile(np.abs(heat), 99) or 1e-8
        scaled = np.clip(heat / limit, -1, 1)
        colors = plt.get_cmap("bwr")((scaled + 1) / 2)[..., :3]
        alpha = 0.6 * np.abs(scaled)[..., None]
    else:
        colors = plt.get_cmap("jet")(np.clip(heat, 0, 1))[..., :3]
        alpha = 0.45

    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    blended = base * (1 - alpha) + colors * alpha
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


def save_explanations_figure(image_tensor, explanations, title, save_path):
    """
    image_tensor: [3, H, W] normalized input
    explanations: {method name: (map [H, W], signed)}
    One row: input + one overlay per method, each with its own color bar.
    """
    image = denormalize_gray(image_tensor)
    columns = 1 + len(explanations)

    fig, axes = plt.subplots(1, columns, figsize=(4.6 * columns, 5.2), squeeze=False)
    axes = axes[0]

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")

    for ax, (name, (explanation, signed)) in zip(axes[1:], explanations.items()):
        ax.imshow(image, cmap="gray")
        if signed:
            # symmetric scale, 99th percentile so a few extreme pixels don't wash everything out
            limit = np.percentile(np.abs(explanation), 99) or 1e-8
            heat = ax.imshow(explanation, cmap="bwr", vmin=-limit, vmax=limit, alpha=0.55)
            ax.set_title(f"{name}\nred = supports, blue = against")
        else:
            heat = ax.imshow(explanation, cmap="jet", vmin=0, vmax=1, alpha=0.45)
            ax.set_title(f"{name}\nimportance")
        fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path
