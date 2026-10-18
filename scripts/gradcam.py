"""
Explainability stage — Grad-CAM (closes gap G1: "predictions carry no evidence").

Standard Grad-CAM hooked onto the backbone's last conv block. Returns a
heatmap you overlay on the input image so a viewer can see WHICH pixels
(headlight cluster, grille slats, badge...) drove the prediction — this
is the evidence panel your proposal promises in Stage 3 ("Fuse & Explain").
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # (1,C,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)       # (1,1,H,W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, logits.softmax(dim=1)[0, class_idx].item()


def overlay_heatmap(orig_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (orig_bgr.shape[1], orig_bgr.shape[0]))
    return cv2.addWeighted(orig_bgr, 1 - alpha, heatmap, alpha, 0)
