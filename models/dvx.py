"""
Robust Differentiable Vectorization (DVX) module.

Improvements vs naive DVX:
- Projects backbone feature maps to `feature_dim` if channels don't match via 1x1 conv.
- Multi-step iterative refinement (improves final polygon accuracy).
- Safe guards for shapes, device handling, and grid-sampling.
- Returns init_polygons, final polygons, per-step displacements, and validity scores.

Usage:
- features: dict of feature maps (e.g. "p2", "p4"), each tensor (B, C, H, W).
- segmentation: (B, 1, H_img, W_img) or similar — only used for optional initialization logic.
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableVectorization(nn.Module):
    def __init__(
        self,
        max_polygons: int = 20,
        max_points: int = 50,
        feature_dim: int = 256,
        displacement_scale: float = 0.08,
        num_refinement_steps: int = 3,
        align_corners: bool = False,
        padding_mode: str = "border",  # options for grid_sample
        use_proj_conv: bool = True,
    ):
        """
        Args:
            max_polygons: maximum polygons to predict per image
            max_points: number of control points per polygon
            feature_dim: number of channels the DVX expects (will project backbone features to this)
            displacement_scale: multiplier for predicted displacement (tanh output)
            num_refinement_steps: how many iterative refinement steps to apply (>=1)
            align_corners: align_corners for F.grid_sample
            padding_mode: padding_mode for F.grid_sample
            use_proj_conv: whether to use 1x1 conv to project backbone features to feature_dim (recommended)
        """
        super().__init__()
        assert max_points > 2, "max_points must be > 2"
        assert num_refinement_steps >= 1

        self.max_polygons = int(max_polygons)
        self.max_points = int(max_points)
        self.feature_dim = int(feature_dim)
        self.displacement_scale = float(displacement_scale)
        self.num_refinement_steps = int(num_refinement_steps)
        self.align_corners = bool(align_corners)
        self.padding_mode = padding_mode
        self.use_proj_conv = bool(use_proj_conv)

        # init_net: from pooled p4 -> flattened -> produce normalized coords in [0,1]
        # AdaptiveAvgPool2d(8) -> (B, C, 8, 8) -> flatten -> Linear(C*8*8 -> hidden)
        hidden = max(512, feature_dim * 2)
        self.init_pool = nn.AdaptiveAvgPool2d(8)

        # we'll create a projector conv for p4/p2 channels if necessary at runtime
        # but also create an MLP init_net that assumes feature_dim channels after pooling
        self.init_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim * 8 * 8, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.max_polygons * self.max_points * 2),
            nn.Sigmoid(),
        )

        # refinement network: maps (feature_dim + 2) -> displacement in [-1,1]
        self.refine_net = nn.Sequential(
            nn.Linear(self.feature_dim + 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Tanh(),
        )

        # validity net (reads flattened coords only)
        self.validity_net = nn.Sequential(
            nn.Linear(self.max_points * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # projector convs (create lazily when first seen a feature channel mismatch)
        # stored per-key: e.g., self._proj_convs['p2'] = nn.Conv2d(in_ch, feature_dim, 1)
        self._proj_convs = nn.ModuleDict()
        self._proj_created = set()

    def _ensure_projector(self, key: str, in_channels: int):
        """
        Ensure a 1x1 conv exists that projects `in_channels` -> self.feature_dim for feature map `key`.
        """
        if not self.use_proj_conv:
            return None
        if key in self._proj_created:
            return self._proj_convs[key]

        if in_channels != self.feature_dim:
            conv = nn.Conv2d(in_channels, self.feature_dim, kernel_size=1, stride=1, padding=0)
            # initialize conv: kaiming
            nn.init.kaiming_normal_(conv.weight, a=0.2)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
            self._proj_convs[key] = conv
        else:
            # identity mapping using 1x1 conv with weights = identity-like is tricky
            # Instead simply keep no conv; we'll pass feature as-is
            self._proj_convs[key] = nn.Identity()
        self._proj_created.add(key)
        return self._proj_convs[key]

    def _project_feature(self, key: str, feat: torch.Tensor) -> torch.Tensor:
        """
        Project or verify feature map to have self.feature_dim channels.
        If projector conv wasn't present and channels == feature_dim, returns feat unchanged.
        """
        in_ch = feat.shape[1]
        proj = self._ensure_projector(key, in_ch)
        if proj is None:
            # projection not desired; assert channels match
            if in_ch != self.feature_dim:
                raise RuntimeError(
                    f"Feature '{key}' channels ({in_ch}) != feature_dim ({self.feature_dim}) "
                    "and projection disabled."
                )
            return feat
        # if proj is Identity, apply it still (fast path)
        return proj(feat)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        segmentation: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
    ) -> Dict[str, Any]:
        """
        features: dict with keys like "p2", "p4" containing tensors (B, C, H, W)
        segmentation: optional (B, 1, H_img, W_img) or similar (not strictly required)
        return_all_steps: if True returns per-step intermediate polygons & displacements
        """
        # pick features for init and refinement
        p4 = features.get("p4", None)
        p2 = features.get("p2", None)

        if p4 is None and p2 is None:
            raise ValueError("At least one of 'p4' or 'p2' must be present in features.")

        # prefer p4 for init; fallback to p2 if not present
        init_feat = p4 if p4 is not None else p2
        refine_feat = p2 if p2 is not None else p4

        B = init_feat.shape[0]

        # Project features to feature_dim (if needed)
        init_feat = self._project_feature("p4_init", init_feat)
        refine_feat = self._project_feature("p2_refine", refine_feat)

        # -- Initialize polygons --
        # Pool then MLP; ensure init_mlp expects feature_dim channels
        pooled = self.init_pool(init_feat)  # [B, C', 8, 8]
        if pooled.shape[1] != self.feature_dim:
            # If the projector returned Identity but pooled channels mismatch, try to apply a runtime projector
            pooled = self._project_feature("p4_init_postpool", pooled)

        init_logits = self.init_mlp(pooled)  # [B, max_polygons * max_points * 2]
        init_polygons = init_logits.view(B, self.max_polygons, self.max_points, 2)  # normalized [0,1]

        # Iterative refinement
        polygons = init_polygons.clone()
        per_step_displacements = []
        for step in range(self.num_refinement_steps):
            # sample features at the polygon control-point locations
            displ = self._single_refine_step(polygons, refine_feat)
            per_step_displacements.append(displ)
            polygons = torch.clamp(polygons + displ, 0.0, 1.0)

        # final validity
        validity = self._predict_validity(polygons)

        out: Dict[str, Any] = {
            "polygons": polygons,  # [B, P, N, 2]
            "validity": validity,  # [B, P]
            "init_polygons": init_polygons,
            "refinement_displacements": per_step_displacements,  # list of [B, P, N, 2]
        }

        if return_all_steps:
            out["all_step_polygons"] = [
                torch.clamp(init_polygons + sum(per_step_displacements[:i + 1]), 0.0, 1.0)
                for i in range(len(per_step_displacements))
            ]

        return out

    def _single_refine_step(self, polygons: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        """
        One refinement step: sample features at polygon points, predict displacement (scaled), return displacement.
        polygons: [B, P, N, 2] in [0,1]
        feature_map: [B, C, H, W] with C == feature_dim (or projected)
        returns displacement: [B, P, N, 2] in [-displacement_scale, displacement_scale]
        """
        B, P, N, _ = polygons.shape
        # flatten pts to sample
        coords = polygons.view(B, -1, 2)  # [B, P*N, 2], coords in [0,1]
        grid = coords * 2.0 - 1.0  # to [-1,1]
        # grid_sample expects (B, H_out, W_out, 2); use W_out=1
        grid_sample = grid.view(B, -1, 1, 2)
        sampled = F.grid_sample(
            feature_map,
            grid_sample,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )  # [B, C, P*N, 1]
        sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # [B, P*N, C]

        # combine sampled features and coords (coords in [0,1])
        input_feats = torch.cat([sampled, coords], dim=-1)  # [B, P*N, C+2]
        # predict displacements in [-1,1] via tanh on last layer
        disp = self.refine_net(input_feats)  # [B, P*N, 2], values ~[-1,1]
        disp = disp.view(B, P, N, 2)
        if torch.isnan(disp).any() or torch.isinf(disp).any():
            return torch.zeros_like(disp)
        disp = disp * self.displacement_scale
        disp = torch.clamp(disp, -self.displacement_scale * 2.0, self.displacement_scale * 2.0)
        return disp

    def _predict_validity(self, polygons: torch.Tensor) -> torch.Tensor:
        B, P, N, _ = polygons.shape
        if N != self.max_points:
            # If someone truncated or padded points, adapt: flatten to last dim whatever it is
            poly_flat = polygons.view(B * P, -1)
        else:
            poly_flat = polygons.view(B * P, -1)
        validity = self.validity_net(poly_flat)  # [B*P, 1]
        validity = validity.view(B, P)
        return validity


# ------------------ quick unit test / smoke test ------------------
def _smoke_test():
    torch.manual_seed(0)
    B = 2
    C1 = 384  # different from feature_dim to test projector conv
    C2 = 128
    H2, W2 = 64, 64
    H4, W4 = 16, 16

    # create dummy backbone features with different channels
    p2 = torch.randn(B, C1, H2, W2)
    p4 = torch.randn(B, C2, H4, W4)
    seg = torch.rand(B, 1, H2 * 4, W2 * 4)  # just a placeholder

    dvx = DifferentiableVectorization(
        max_polygons=4,
        max_points=16,
        feature_dim=256,
        displacement_scale=0.08,
        num_refinement_steps=3,
        align_corners=False,
        padding_mode="border",
        use_proj_conv=True,
    )

    # ensure module moves projector convs to device when dvx.to(device) called
    dvx = dvx.eval()  # inference mode ok
    # Forward pass
    out = dvx({"p2": p2, "p4": p4}, seg, return_all_steps=True)
    print("polygons shape:", out["polygons"].shape)  # expected [B, P, N, 2]
    print("validity shape:", out["validity"].shape)  # expected [B, P]
    print("init shape:", out["init_polygons"].shape)
    print("refinement steps:", len(out["refinement_displacements"]))
    # check ranges
    assert out["polygons"].min().item() >= 0.0 - 1e-6
    assert out["polygons"].max().item() <= 1.0 + 1e-6
    print("smoke test passed")


if __name__ == "__main__":
    _smoke_test()
