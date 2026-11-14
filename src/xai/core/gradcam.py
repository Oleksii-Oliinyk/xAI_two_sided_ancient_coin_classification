import torch
import torch.nn.functional as F

from src.xai.core.base import BaseExplainer


class GradCAMExplainer(BaseExplainer):
    """
    Grad-CAM (Selvaraju et al., 2017): weights every channel of the last conv block by its
    average gradient w.r.t. the class score, sums the weighted channels and keeps the positive part.
    """
    name = "Grad-CAM"
    signed = False

    def _activations_and_gradients(self, image_tensor, class_idx):
        layer = self.model.get_submodule(self.target_layer)
        store = {}

        def forward_hook(module, inputs, output):
            store["activations"] = output
            output.register_hook(lambda grad: store.__setitem__("gradients", grad))

        handle = layer.register_forward_hook(forward_hook)
        try:
            with torch.enable_grad():
                self.model.zero_grad()
                logits = self.model(image_tensor.unsqueeze(0).to(self.device))
                logits[0, class_idx].backward()
        finally:
            handle.remove()
            self.model.zero_grad()

        # [C, h, w] each
        return store["activations"].detach()[0], store["gradients"].detach()[0]

    def _channel_weights(self, activations, gradients):
        return gradients.mean(dim=(1, 2))

    def explain(self, image_tensor, class_idx):
        activations, gradients = self._activations_and_gradients(image_tensor, class_idx)
        weights = self._channel_weights(activations, gradients)

        cam = F.relu((weights[:, None, None] * activations).sum(dim=0))
        cam = F.interpolate(cam[None, None], size=image_tensor.shape[1:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy()


class GradCAMPlusPlusExplainer(GradCAMExplainer):
    """
    Grad-CAM++ (Chattopadhyay et al., 2018): like Grad-CAM, but every spatial position gets its own
    weight (alpha) so several instances / parts of the object are highlighted more evenly.
    """
    name = "Grad-CAM++"

    def _channel_weights(self, activations, gradients):
        grads_2 = gradients ** 2
        grads_3 = gradients ** 3
        sum_activations = activations.sum(dim=(1, 2))

        denominator = 2 * grads_2 + sum_activations[:, None, None] * grads_3
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        alpha = grads_2 / denominator

        return (alpha * F.relu(gradients)).sum(dim=(1, 2))
