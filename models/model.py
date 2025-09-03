"""
Enhanced model definition with auxiliary heads for novel training strategies
Includes cross-modal consistency embeddings and graph structure interfaces
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import MultiScaleEncoder
from .heads import SegmentationHead, AttributeHead, SDFHead
from .dvx import DifferentiableVectorization
from .extrusion import DifferentiableExtrusion


class L2Normalize(nn.Module):
    """L2 normalization layer"""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)


class LatentEmbeddingHead(nn.Module):
    """Auxiliary head for cross-modal latent consistency"""

    def __init__(self, feature_dim: int, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # 2D embedding path
        self.embedding_2d = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            L2Normalize(dim=1),  # L2 normalize for cosine similarity
        )

        # 3D embedding path (from voxel features)
        self.embedding_3d = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            L2Normalize(dim=1),
        )

    def forward(
        self, features_2d: torch.Tensor, features_3d: torch.Tensor = None
    ) -> tuple:
        """
        Generate 2D and 3D embeddings for consistency loss

        Args:
            features_2d: [B, C, H, W] - 2D feature maps
            features_3d: [B, C, D, H, W] - 3D feature maps (optional)

        Returns:
            tuple: (embedding_2d, embedding_3d)
        """
        # 2D embedding
        emb_2d = self.embedding_2d(features_2d)

        # 3D embedding (if available, otherwise use 2D features reshaped)
        if features_3d is not None:
            emb_3d = self.embedding_3d(features_3d)
        else:
            # Create pseudo-3D from 2D features
            B, C, H, W = features_2d.shape
            pseudo_3d = features_2d.unsqueeze(2).expand(
                B, C, 4, H, W
            )  # Duplicate across depth
            emb_3d = self.embedding_3d(pseudo_3d)

        return emb_2d, emb_3d


class GraphStructureHead(nn.Module):
    """Head for predicting graph structure (room connectivity)"""

    def __init__(self, feature_dim: int, max_rooms: int = 10):
        super().__init__()
        self.max_rooms = max_rooms

        # Room detection branch
        self.room_detector = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, max_rooms, 3, padding=1),
            nn.Sigmoid(),  # Room probability maps
        )

        # Room feature extractor
        self.room_features = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # Pool to fixed size
            nn.Flatten(),
            nn.Linear(feature_dim * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Room feature vectors
        )

        # Adjacency predictor
        self.adjacency_net = nn.Sequential(
            nn.Linear(128 * 2, 64),  # Pairwise room features
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # Adjacency probability
        )

    def forward(self, features: torch.Tensor) -> dict:
        """
        Predict room graph structure

        Args:
            features: [B, C, H, W] - Feature maps

        Returns:
            dict with 'room_maps', 'room_features', 'adjacency_matrix'
        """
        B = features.shape[0]

        # Detect room probability maps
        room_maps = self.room_detector(features)  # [B, max_rooms, H, W]

        # Extract room features
        room_feats = self.room_features(features)  # [B, 128]

        # Create adjacency matrix for all room pairs
        adjacency_matrices = []

        for b in range(B):
            # Get room features for this batch item
            feat_b = room_feats[b : b + 1]  # [1, 128]

            # Create pairwise combinations
            adj_matrix = torch.zeros(
                (self.max_rooms, self.max_rooms), device=features.device
            )

            for i in range(self.max_rooms):
                for j in range(i + 1, self.max_rooms):
                    # Concatenate features for room pair
                    pair_feat = torch.cat([feat_b, feat_b], dim=1)  # [1, 256]

                    # Predict adjacency
                    adj_prob = self.adjacency_net(pair_feat)  # [1, 1]

                    # Fill symmetric matrix
                    adj_matrix[i, j] = adj_prob.squeeze()
                    adj_matrix[j, i] = adj_prob.squeeze()

            adjacency_matrices.append(adj_matrix)

        return {
            "room_maps": room_maps,
            "room_features": room_feats,
            "adjacency_matrices": torch.stack(adjacency_matrices),
        }


class NeuralGeometric3DGenerator(nn.Module):
    """
    Enhanced neural-geometric system with auxiliary heads for novel training strategies:
    - Cross-modal latent consistency
    - Graph structure prediction
    - Multi-view embeddings for dynamic curriculum
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=5,
        feature_dim=512,
        num_attributes=6,
        voxel_size=64,
        max_polygons=20,
        max_points=50,
        use_latent_consistency=True,
        use_graph_constraints=True,
        latent_embedding_dim=256,
    ):
        super().__init__()

        # Store configuration
        self.use_latent_consistency = use_latent_consistency
        self.use_graph_constraints = use_graph_constraints
        self.feature_dim = feature_dim

        # Core components
        self.encoder = MultiScaleEncoder(input_channels, feature_dim)
        self.seg_head = SegmentationHead(feature_dim, num_classes)
        self.attr_head = AttributeHead(feature_dim, num_attributes)
        self.sdf_head = SDFHead(feature_dim)
        self.dvx = DifferentiableVectorization(max_polygons, max_points, feature_dim)
        self.extrusion = DifferentiableExtrusion(voxel_size)

        # NEW: Auxiliary heads for novel training strategies
        if use_latent_consistency:
            self.latent_head = LatentEmbeddingHead(feature_dim, latent_embedding_dim)

        if use_graph_constraints:
            self.graph_head = GraphStructureHead(feature_dim)

        # Enhanced feature processing for multi-stage training
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(32, feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(32, feature_dim),
        )

        # lazy-created 3d voxel processor will be attached on first use
        self._voxel_processor = None

    def _select_spatial_feature(self, features):
        """
        Given encoder output (dict or tensor), select a spatial 4-D feature map
        Prefer high-resolution feature maps (p1) and avoid selecting 'global' vector.
        """
        # If encoder returned a tensor already, make sure it's 4D
        if not isinstance(features, dict):
            if features.dim() == 4:
                return features
            else:
                raise ValueError(
                    f"Encoder returned a tensor with shape {tuple(features.shape)}; "
                    "expected a 4D feature map [B, C, H, W]."
                )

        # Encoder returned dict: prefer p1,p2,p3,p4,high_res,out,main but NEVER 'global'
        preferred_keys = ["p1", "p2", "p3", "p4", "high_res", "out", "main"]
        for k in preferred_keys:
            if k in features:
                candidate = features[k]
                if isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
                    return candidate

        # As a last resort, scan dict values for the first 4D tensor that isn't 'global'
        for k, v in features.items():
            if k == "global":
                continue
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                return v

        # If nothing found, raise informative error rather than silently picking wrong shape
        raise RuntimeError(
            "No spatial 4D feature map found in encoder output. Encoder returned keys: "
            f"{list(features.keys())}. Ensure encoder provides at least one [B,C,H,W] tensor "
            "under keys like 'p1','p2','p3','p4','out', or 'high_res'."
        )

    def forward(self, image, return_aux=True):
        """
        Enhanced forward pass with auxiliary outputs

        Args:
            image: [B, C, H, W] input images
            return_aux: Whether to compute auxiliary outputs

        Returns:
            dict with all predictions including auxiliary outputs
        """
        # Multi-scale feature extraction
        features = self.encoder(image)

        # Enhance features
        # Use robust selector to avoid accidentally selecting the 'global' vector.
        spatial_feat = self._select_spatial_feature(features)
        enhanced_features = self.feature_enhancer(spatial_feat)
        # keep structured features dict for heads that expect multi-scale inputs
        if isinstance(features, dict):
            features["enhanced"] = enhanced_features
            main_features = enhanced_features
        else:
            # encoder returned 4D tensor
            features = {"main": enhanced_features, "enhanced": enhanced_features}
            main_features = enhanced_features

        # Core predictions
        segmentation = self.seg_head(features)
        attributes = self.attr_head(
            features.get("global")
            if "global" in features
            else main_features.mean(dim=[2, 3])
        )
        sdf = self.sdf_head(features)

        # DVX polygon fitting
        dvx_output = self.dvx(features, segmentation)
        polygons = dvx_output["polygons"]
        validity = dvx_output["validity"]

        # 3D extrusion
        voxels_pred = self.extrusion(polygons, attributes, validity)

        # Base outputs
        outputs = {
            "segmentation": segmentation,
            "attributes": attributes,
            "sdf": sdf,
            "polygons": polygons,
            "polygon_validity": validity,
            "voxels_pred": voxels_pred,
            "features": features,
        }

        # NEW: Auxiliary outputs for novel training strategies
        if return_aux:
            # Cross-modal consistency embeddings
            if self.use_latent_consistency:
                # Create 3D features from voxels for consistency
                voxel_features = self._create_3d_features_from_voxels(voxels_pred)
                latent_2d, latent_3d = self.latent_head(main_features, voxel_features)
                outputs["latent_2d_embedding"] = latent_2d
                outputs["latent_3d_embedding"] = latent_3d

            # Graph structure predictions
            if self.use_graph_constraints:
                graph_output = self.graph_head(main_features)
                outputs.update(graph_output)

        return outputs

    def get_latent_embeddings(self, image):
        """
        Convenience method to get just the latent embeddings
        Used by trainer for consistency loss
        """
        if not self.use_latent_consistency:
            return None, None

        with torch.no_grad():
            features = self.encoder(image)
            # reuse robust selector to derive main_features
            spatial_feat = self._select_spatial_feature(features)
            main_features = self.feature_enhancer(spatial_feat)

            # Quick forward to get voxels
            segmentation = self.seg_head(features)
            attributes = self.attr_head(
                features.get("global")
                if isinstance(features, dict) and "global" in features
                else main_features.mean(dim=[2, 3])
            )
            dvx_output = self.dvx(features, segmentation)
            voxels_pred = self.extrusion(
                dvx_output["polygons"], attributes, dvx_output["validity"]
            )

        # Get embeddings
        voxel_features = self._create_3d_features_from_voxels(voxels_pred)
        return self.latent_head(main_features, voxel_features)

    def _create_3d_features_from_voxels(self, voxels):
        """
        Create 3D feature representation from voxel predictions

        Args:
            voxels: [B, D, H, W] voxel predictions

        Returns:
            [B, C, D, H, W] 3D features
        """
        # Ensure expected shape
        if voxels.dim() != 4:
            raise ValueError(f"voxels must have shape [B,D,H,W], got {tuple(voxels.shape)}")

        B, D, H, W = voxels.shape

        # Expand voxels to have feature channels
        # Simple approach: repeat voxel values across feature dimension
        rep_ch = max(1, self.feature_dim // 4)
        voxel_features = voxels.unsqueeze(1).expand(B, rep_ch, D, H, W).contiguous()

        # Add some learned 3D processing
        if self._voxel_processor is None:
            # Build with correct device
            device = voxels.device
            self._voxel_processor = nn.Sequential(
                nn.Conv3d(rep_ch, max(rep_ch, self.feature_dim // 2), 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(max(rep_ch, self.feature_dim // 2), self.feature_dim, 3, padding=1),
            ).to(device)

        return self._voxel_processor(voxel_features)

    def get_stage_parameters(self, stage: int):
        """
        Get parameters for specific training stage
        Useful for stage-specific optimization
        """
        if stage == 1:
            # Stage 1: 2D components only
            params = []
            params.extend(list(self.encoder.parameters()))
            params.extend(list(self.seg_head.parameters()))
            params.extend(list(self.attr_head.parameters()))
            params.extend(list(self.sdf_head.parameters()))
            params.extend(list(self.feature_enhancer.parameters()))

            if self.use_latent_consistency:
                params.extend(list(self.latent_head.parameters()))

        elif stage == 2:
            # Stage 2: DVX components
            params = list(self.dvx.parameters())

        else:  # stage == 3
            # Stage 3: All parameters
            params = list(self.parameters())

        return params

    def freeze_stage_parameters(self, stages_to_freeze: list):
        """
        Freeze parameters for specific stages

        Args:
            stages_to_freeze: List of stage numbers to freeze
        """
        for stage in stages_to_freeze:
            stage_params = self.get_stage_parameters(stage)
            for param in stage_params:
                param.requires_grad = False

    def unfreeze_stage_parameters(self, stages_to_unfreeze: list):
        """
        Unfreeze parameters for specific stages

        Args:
            stages_to_unfreeze: List of stage numbers to unfreeze
        """
        for stage in stages_to_unfreeze:
            stage_params = self.get_stage_parameters(stage)
            for param in stage_params:
                param.requires_grad = True

    def get_curriculum_metrics(self):
        """
        Get metrics useful for curriculum learning decisions
        """
        metrics = {}

        # Parameter counts per stage
        for stage in [1, 2, 3]:
            stage_params = self.get_stage_parameters(stage)
            metrics[f"stage_{stage}_params"] = sum(p.numel() for p in stage_params)

        # Feature dimensions
        metrics["feature_dim"] = self.feature_dim
        metrics["has_latent_consistency"] = self.use_latent_consistency
        metrics["has_graph_constraints"] = self.use_graph_constraints

        return metrics
