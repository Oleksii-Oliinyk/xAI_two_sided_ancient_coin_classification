import numpy as np
import torch
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

from src.xai.core.base import BaseExplainer, denormalize_gray, normalize_gray_batch


class LimeExplainer(BaseExplainer):
    """
    LIME (Ribeiro et al., 2016): splits the coin into superpixels, hides random subsets of them,
    and fits a linear model on how the class probability changes. Every superpixel gets one weight.
    """
    name = "LIME"
    signed = True

    num_samples = 1000   # perturbed images per explanation
    num_segments = 50    # superpixels (SLIC)
    batch_size = 64
    seed = 42

    @torch.no_grad()
    def _classifier_fn(self, images):
        # LIME passes [N, H, W, 3] images in 0..1, all 3 channels are equal (grayscale)
        probs = []
        for start in range(0, len(images), self.batch_size):
            batch = normalize_gray_batch(images[start:start + self.batch_size, :, :, 0]).to(self.device)
            probs.append(torch.softmax(self.model(batch), dim=1).cpu().numpy())
        return np.concatenate(probs)

    def explain(self, image_tensor, class_idx):
        gray = denormalize_gray(image_tensor)
        rgb = np.stack([gray] * 3, axis=-1).astype(np.float64)

        # start_label=0: LIME expects segment ids 0..n-1
        segmenter = SegmentationAlgorithm("slic", n_segments=self.num_segments, compactness=10,
                                          sigma=1, start_label=0, random_seed=self.seed)
        explanation = lime_image.LimeImageExplainer(random_state=self.seed).explain_instance(
            rgb,
            self._classifier_fn,
            labels=(class_idx,),
            top_labels=None,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            segmentation_fn=segmenter,
            random_seed=self.seed,
        )

        segment_weights = np.zeros(explanation.segments.max() + 1)
        for segment, weight in explanation.local_exp[class_idx]:
            segment_weights[segment] = weight
        return segment_weights[explanation.segments]
