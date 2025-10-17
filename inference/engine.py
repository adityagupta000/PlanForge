"""
Research-grade inference engine for 2D to 3D floorplan generation
Enhanced with robust model loading and checkpoint compatibility
"""

import torch
import cv2
import numpy as np
import json
import trimesh
from pathlib import Path

from models.model import NeuralGeometric3DGenerator
from config import DEFAULT_INFERENCE_CONFIG


class ResearchInferenceEngine:
    """
    Complete inference system that converts 2D floorplans to 3D models
    following the deterministic export pipeline
    """

    def __init__(self, model_path=None, device="cuda", config=None):
        if config is None:
            config = DEFAULT_INFERENCE_CONFIG

        self.device = device
        self.config = config
        self.model = NeuralGeometric3DGenerator()

        # Load trained model with robust state dict handling
        model_path = model_path or config.model_path
        self._load_model_safely(model_path)
        
        self.model.to(device)
        self.model.eval()

        print(f"Loaded trained model from {model_path}")

    def _load_model_safely(self, model_path):
        """
        Load model checkpoint with flexible state dict matching
        Handles keys that may not exist in current model architecture
        """
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model_state = checkpoint.get("model_state_dict", checkpoint)

        # Get current model keys
        current_keys = set(self.model.state_dict().keys())
        checkpoint_keys = set(model_state.keys())

        # Find compatible and incompatible keys
        compatible_state = {}
        incompatible_keys = []
        missing_keys = []

        # Load only compatible parameters
        for key, value in model_state.items():
            if key in current_keys:
                current_param = self.model.state_dict()[key]
                
                # Check shape compatibility
                if current_param.shape == value.shape:
                    compatible_state[key] = value
                else:
                    incompatible_keys.append(
                        f"{key} (shape mismatch: {value.shape} -> {current_param.shape})"
                    )
            else:
                # Skip keys that don't exist in current model (lazily-created modules)
                incompatible_keys.append(f"{key} (not in current model - lazy module)")

        # Check for missing keys
        for key in current_keys:
            if key not in model_state:
                missing_keys.append(key)

        # Load compatible state dict
        self.model.load_state_dict(compatible_state, strict=False)

        # Report loading status
        print(f"\n[INFO] Model Loading Status:")
        print(f"  ✓ Loaded {len(compatible_state)} compatible parameters")
        
        if incompatible_keys:
            print(f"  ! Skipped {len(incompatible_keys)} incompatible/lazy parameters")
            for key in incompatible_keys[:3]:  # Show first 3
                print(f"    - {key}")
            if len(incompatible_keys) > 3:
                print(f"    ... and {len(incompatible_keys) - 3} more")

        if missing_keys:
            print(f"  ! {len(missing_keys)} parameters will use random initialization")
            for key in missing_keys[:3]:  # Show first 3
                print(f"    - {key}")
            if len(missing_keys) > 3:
                print(f"    ... and {len(missing_keys) - 3} more")

    def generate_3d_model(
        self, image_path: str, output_path: str, export_intermediate: bool = None
    ):
        """
        Complete pipeline: Image -> Segmentation -> Polygons -> 3D Model
        """

        export_intermediate = export_intermediate or self.config.export_intermediate

        # Load and preprocess image
        image = self._load_image(image_path)

        with torch.no_grad():
            # Neural network inference
            predictions = self.model(image)

            # Extract predictions
            segmentation = predictions["segmentation"]
            attributes = predictions["attributes"]
            polygons = predictions["polygons"]
            validity = predictions["polygon_validity"]

            print("Neural network inference complete")

            # Convert to deterministic representations
            mask_np = self._extract_mask(segmentation)
            attributes_dict = self._extract_attributes(attributes)
            polygons_list = self._extract_polygons(polygons, validity)

            print(f"Extracted: {len(polygons_list)} valid polygons")

            # Export intermediate results if requested
            if export_intermediate:
                self._export_intermediates(
                    mask_np, attributes_dict, polygons_list, Path(output_path).parent
                )

            # Generate 3D model using deterministic pipeline
            success = self._generate_deterministic_3d(
                mask_np, attributes_dict, polygons_list, output_path
            )

            return success

    def _load_image(self, image_path):
        """Load and preprocess input image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = torch.from_numpy(image / 255.0).float()
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)

    def _extract_mask(self, segmentation):
        """Convert soft segmentation to hard mask"""
        seg_pred = torch.argmax(segmentation, dim=1)
        mask_np = seg_pred.squeeze().cpu().numpy().astype(np.uint8)
        return mask_np

    def _extract_attributes(self, attributes):
        """Convert normalized attributes back to physical values"""
        attr_np = attributes.squeeze().cpu().numpy()

        # Denormalize (reverse of normalization in dataset)
        attributes_dict = {
            "wall_height": float(attr_np[0] * 2.0),
            "wall_thickness": float(attr_np[1] * 0.5),
            "window_base_height": float(attr_np[2] * 0.5),
            "window_height": float(attr_np[3] * 0.5),
            "door_height": float(attr_np[4] * 2.0),
            "pixel_scale": float(attr_np[5] * 0.02),
        }

        return attributes_dict

    def _extract_polygons(self, polygons, validity, threshold=None):
        """Extract valid polygons from network predictions"""

        threshold = threshold or self.config.polygon_threshold
        batch_size, num_polys, num_points, _ = polygons.shape

        polygons_list = []

        for poly_idx in range(num_polys):
            if validity[0, poly_idx] > threshold:
                poly_points = polygons[0, poly_idx].cpu().numpy()

                # Remove zero-padded points
                valid_points = poly_points[poly_points.sum(axis=1) > 0]

                if len(valid_points) >= 3:
                    # Convert to image coordinates (assuming 256x256 input)
                    valid_points = valid_points * 256
                    polygons_list.append(
                        {
                            "points": valid_points.tolist(),
                            "class": "wall",
                        }
                    )

        return polygons_list

    def _export_intermediates(self, mask, attributes, polygons, output_dir):
        """Export intermediate results for debugging/analysis"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Export mask
        cv2.imwrite(str(output_dir / "predicted_mask.png"), mask * 50)

        # Export attributes
        with open(output_dir / "predicted_attributes.json", "w") as f:
            json.dump(attributes, f, indent=2)

        # Export polygons
        with open(output_dir / "predicted_polygons.json", "w") as f:
            json.dump(polygons, f, indent=2)

        # Visualize polygons on mask
        vis_img = np.zeros((256, 256, 3), dtype=np.uint8)
        vis_img[:, :, 0] = mask * 50

        for poly in polygons:
            points = np.array(poly["points"], dtype=np.int32)
            cv2.polylines(vis_img, [points], True, (0, 255, 0), 2)

        cv2.imwrite(str(output_dir / "polygon_visualization.png"), vis_img)

        print(f"Intermediate results exported to {output_dir}")

    def _generate_deterministic_3d(self, mask, attributes, polygons, output_path):
        """Generate 3D model using deterministic geometric operations"""
        try:
            # Initialize mesh components
            vertices = []
            faces = []
            vertex_count = 0

            # Extract geometric parameters
            wall_height = attributes.get("wall_height", 1.0)
            wall_thickness = attributes.get("wall_thickness", 0.15)
            pixel_scale = attributes.get("pixel_scale", 0.01)

            print(
                f"Generating 3D model with wall_height={wall_height:.2f}m, "
                f"thickness={wall_thickness:.2f}m"
            )

            # Process each polygon (walls, rooms, etc.)
            for poly_idx, polygon in enumerate(polygons):
                poly_vertices, poly_faces = self._extrude_polygon_3d(
                    polygon["points"],
                    wall_height,
                    wall_thickness,
                    pixel_scale,
                    vertex_count,
                )

                vertices.extend(poly_vertices)
                faces.extend(poly_faces)
                vertex_count += len(poly_vertices)

            # Add floor only (no ceiling/roof)
            floor_verts, floor_faces = self._generate_floor_only(
                mask, pixel_scale, vertex_count
            )
            vertices.extend(floor_verts)
            faces.extend(floor_faces)

            if len(vertices) == 0:
                print("No geometry generated")
                return False

            # Create mesh
            mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

            # Clean up mesh
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()

            # Export
            mesh.export(output_path)
            print(f"3D model exported to {output_path}")
            print(
                f"Mesh statistics: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
            )

            return True

        except Exception as e:
            print(f"Error generating 3D model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _extrude_polygon_3d(self, points, height, thickness, scale, vertex_offset):
        """Extrude a 2D polygon to create 3D wall geometry"""
        vertices = []
        faces = []

        # Convert points to 3D coordinates
        points_3d = []
        for point in points:
            x = (point[0] - 128) * scale
            z = (128 - point[1]) * scale
            points_3d.append([x, 0, z])

        # Create bottom vertices (y=0)
        bottom_outer = points_3d
        bottom_inner = self._inset_polygon(points_3d, thickness)

        # Create top vertices (y=height)
        top_outer = [[p[0], height, p[2]] for p in bottom_outer]
        top_inner = [[p[0], height, p[2]] for p in bottom_inner]

        # Combine all vertices
        all_vertices = bottom_outer + bottom_inner + top_outer + top_inner
        vertices.extend(all_vertices)

        n_points = len(points_3d)

        # Generate faces for walls
        for i in range(n_points):
            next_i = (i + 1) % n_points

            # Outer wall faces
            v1 = vertex_offset + i
            v2 = vertex_offset + next_i
            v3 = vertex_offset + 2 * n_points + next_i
            v4 = vertex_offset + 2 * n_points + i

            faces.extend([[v1, v2, v3], [v1, v3, v4]])

            # Inner wall faces (reverse winding)
            v1 = vertex_offset + n_points + i
            v2 = vertex_offset + n_points + next_i
            v3 = vertex_offset + 3 * n_points + next_i
            v4 = vertex_offset + 3 * n_points + i

            faces.extend([[v1, v3, v2], [v1, v4, v3]])

        # Top cap (between outer and inner)
        for i in range(n_points):
            next_i = (i + 1) % n_points

            v1 = vertex_offset + 2 * n_points + i
            v2 = vertex_offset + 2 * n_points + next_i
            v3 = vertex_offset + 3 * n_points + next_i
            v4 = vertex_offset + 3 * n_points + i

            faces.extend([[v1, v2, v3], [v1, v3, v4]])

        # Bottom cap (between outer and inner)
        for i in range(n_points):
            next_i = (i + 1) % n_points

            v1 = vertex_offset + i
            v2 = vertex_offset + next_i
            v3 = vertex_offset + n_points + next_i
            v4 = vertex_offset + n_points + i

            faces.extend([[v1, v3, v2], [v1, v4, v3]])

        return vertices, faces

    def _inset_polygon(self, points, inset_distance):
        """Create inset polygon for wall thickness"""
        if len(points) < 3:
            return points

        inset_points = []
        n = len(points)

        for i in range(n):
            prev_i = (i - 1) % n
            next_i = (i + 1) % n

            p_prev = np.array(points[prev_i])
            p_curr = np.array(points[i])
            p_next = np.array(points[next_i])

            # Vectors to adjacent points
            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            # Normalize vectors (in XZ plane, ignore Y)
            v1_norm = np.array([v1[0], 0, v1[2]])
            v2_norm = np.array([v2[0], 0, v2[2]])

            v1_len = np.linalg.norm(v1_norm)
            v2_len = np.linalg.norm(v2_norm)

            if v1_len > 1e-6:
                v1_norm /= v1_len
            if v2_len > 1e-6:
                v2_norm /= v2_len

            # Angle bisector
            bisector = v1_norm + v2_norm
            bisector_len = np.linalg.norm(bisector)

            if bisector_len > 1e-6:
                bisector /= bisector_len

            # Move point inward
            inset_point = p_curr - bisector * inset_distance
            inset_points.append([inset_point[0], inset_point[1], inset_point[2]])

        return inset_points

    def _generate_floor_only(self, mask, scale, vertex_offset):
        """Generate floor geometry only (no ceiling/roof)"""
        vertices = []
        faces = []

        # Find floor regions (assuming class 0 = floor/room)
        floor_mask = (mask == 0).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            # Convert to 3D coordinates
            floor_points = []
            for point in approx.reshape(-1, 2):
                x = (point[0] - 128) * scale
                z = (128 - point[1]) * scale
                floor_points.append([x, 0, z])

            # Add vertices
            n_points = len(floor_points)
            vertices.extend(floor_points)

            # Triangulate floor
            if n_points >= 3:
                for i in range(1, n_points - 1):
                    faces.append([vertex_offset, vertex_offset + i + 1, vertex_offset + i])

            vertex_offset += n_points

        return vertices, faces

    def _generate_floor_ceiling(self, mask, scale, wall_height, vertex_offset):
        """Generate floor and ceiling geometry from segmentation mask"""
        vertices = []
        faces = []

        # Find floor regions (assuming class 0 = floor/room)
        floor_mask = (mask == 0).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            # Convert to 3D coordinates
            floor_points = []
            for point in approx.reshape(-1, 2):
                x = (point[0] - 128) * scale
                z = (128 - point[1]) * scale
                floor_points.append([x, 0, z])

            ceiling_points = []
            for point in approx.reshape(-1, 2):
                x = (point[0] - 128) * scale
                z = (128 - point[1]) * scale
                ceiling_points.append([x, wall_height, z])

            # Add vertices
            n_points = len(floor_points)
            vertices.extend(floor_points)
            vertices.extend(ceiling_points)

            # Triangulate floor
            if n_points >= 3:
                for i in range(1, n_points - 1):
                    faces.append([vertex_offset, vertex_offset + i + 1, vertex_offset + i])

            # Triangulate ceiling (reverse winding)
            for i in range(1, n_points - 1):
                faces.append(
                    [
                        vertex_offset + n_points,
                        vertex_offset + n_points + i,
                        vertex_offset + n_points + i + 1,
                    ]
                )

            vertex_offset += 2 * n_points

        return vertices, faces

    def process_batch(self, image_paths, output_dir):
        """Process multiple images in batch"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        results = []

        for img_path in image_paths:
            img_path = Path(img_path)
            print(f"Processing: {img_path.name}")

            output_path = output_dir / f"{img_path.stem}_model.obj"

            try:
                success = self.generate_3d_model(
                    str(img_path), str(output_path), export_intermediate=True
                )

                results.append(
                    {
                        "input": str(img_path),
                        "output": str(output_path),
                        "success": success,
                    }
                )

                if success:
                    print(f"✓ Generated: {output_path}")
                else:
                    print(f"✗ Failed: {img_path.name}")

            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {str(e)}")
                results.append(
                    {
                        "input": str(img_path),
                        "output": str(output_path),
                        "success": False,
                        "error": str(e),
                    }
                )

        return results