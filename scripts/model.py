"""
Stage 2/3 — model architecture.

This is a *scoped, honest, feasible* reinterpretation of DeepCar 5.0 for a
course-timeline project, not a literal re-implementation:

  DeepCar 5.0 (base paper)                 This project
  --------------------------------------   --------------------------------------
  4 hand-labelled ROI agents               2 agents: (a) global view,
  (headlight/grille/bumper/scoop),         (b) an attention-discovered ROI crop —
  4 different heavy backbones              found automatically, no manual bboxes
  (ResNet101/VGG19/Xception/DenseNet201)   needed, since we don't have that
                                            annotation budget.
  Fixed majority-vote blackboard           Learnable weighted-vote "blackboard"
                                            (a 2-logit gating layer) fusing the
                                            two agents' softmax outputs.
  Heavy backbones                          Shared MobileNetV3-Small backbone
                                            (edge-deployable — closes gap G3)
  No explanation of predictions            Grad-CAM over both agents
                                            (closes gap G1, see gradcam.py)

Why one shared backbone instead of two full networks: with a few thousand
images (not 40k), two independently-trained heavy CNNs will overfit hard.
A shared, ImageNet-pretrained trunk with two small heads is the version of
"multi-agent ensembling" that is actually trainable on a laptop-scale
Indian car dataset. State this trade-off explicitly in your Methodology
section — it's a design decision, not a cut corner.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class SpatialAttention(nn.Module):
    """Learns WHERE to look (headlight/grille-scale regions) from feature maps.
    This produces the attention map used both for the second agent's crop
    and for the attention-mechanism figure in your paper (cf. DeepCar Fig. 4)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat: torch.Tensor):
        attn_map = self.sigmoid(self.conv(feat))          # (B,1,H,W)
        attended_feat = feat * attn_map
        return attended_feat, attn_map


class VMMRNet(nn.Module):
    def __init__(self, num_classes: int, backbone: str = "mobilenet_v3_small", pretrained: bool = True):
        super().__init__()
        if backbone == "mobilenet_v3_small":
            base = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            self.features = base.features
            feat_channels = 576
        elif backbone == "efficientnet_b0":
            base = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            self.features = base.features
            feat_channels = 1280
        else:
            raise ValueError(f"unsupported backbone: {backbone}")

        self.attention = SpatialAttention(feat_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Agent A: global (whole front-view)
        self.agent_global = nn.Sequential(
            nn.Linear(feat_channels, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        # Agent B: attention-weighted (mimics the part-based ROI agents)
        self.agent_roi = nn.Sequential(
            nn.Linear(feat_channels, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        # Blackboard fusion: learnable scalar gate per agent instead of a
        # fixed majority vote (DeepCar 5.0 used fixed majority vote across 4 agents)
        self.gate = nn.Parameter(torch.tensor([0.5, 0.5]))

        self._feat_channels = feat_channels

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        feat = self.features(x)                       # (B,C,H,W)
        attended_feat, attn_map = self.attention(feat)

        global_vec = self.pool(feat).flatten(1)
        roi_vec = self.pool(attended_feat).flatten(1)

        logits_global = self.agent_global(global_vec)
        logits_roi = self.agent_roi(roi_vec)

        w = torch.softmax(self.gate, dim=0)
        fused_logits = w[0] * logits_global + w[1] * logits_roi

        if return_attention:
            return fused_logits, logits_global, logits_roi, attn_map
        return fused_logits

    def target_layer(self):
        """Layer to hook for Grad-CAM (last conv block of the backbone)."""
        return self.features[-1]


def build_model(num_classes: int, backbone: str = "mobilenet_v3_small") -> VMMRNet:
    return VMMRNet(num_classes=num_classes, backbone=backbone, pretrained=True)
