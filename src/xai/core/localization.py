import numpy as np
from scipy import ndimage


def attention_mask(explanation, top_percent=20.0, smooth_sigma=4.0):
    """
    explanation: map [H, W] from any explainer (signed or not)
    Returns the boolean mask of the `top_percent` % most important pixels that SUPPORT the class.

    Only the positive part is used: for SHAP / LIME negative values mean "against the class",
    for Grad-CAM(++) the map is already >= 0. The same percentile is taken for every method,
    so masks have (almost) the same area and IoU is not biased by how "spread" a method is.
    smooth_sigma: Gaussian blur in pixels before thresholding, turns pixel-level SHAP noise into
                  regions (Grad-CAM is already smooth, so it barely changes). 0 = no blur.
    """
    positive = np.clip(np.asarray(explanation, dtype=np.float32), 0, None)
    if smooth_sigma > 0:
        positive = ndimage.gaussian_filter(positive, sigma=smooth_sigma)
    if positive.max() <= 0:
        return np.zeros(positive.shape, dtype=bool)

    threshold = np.percentile(positive, 100 - top_percent)
    return (positive >= threshold) & (positive > 0)


def main_region_bbox(mask):
    """
    Bounding box [x0, y0, x1, y1] (inclusive pixels) of the largest connected region of the mask,
    i.e. the single coin element the method focuses on most. None for an empty mask.
    """
    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=range(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    ys, xs = np.nonzero(labels == largest)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    if union == 0:
        return np.nan
    return float(np.logical_and(a, b).sum() / union)


def bbox_iou(a, b):
    if a is None or b is None:
        return np.nan
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix1 - ix0 + 1) * max(0, iy1 - iy0 + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return float(intersection / (area_a + area_b - intersection))
