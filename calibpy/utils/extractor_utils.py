from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class ImageProcessParams:
    """Parameters for image preprocessing"""

    # Image preprocessing parameters
    gaussian_kernel: int = 5
    adapt_thresh_block: int = 13
    adapt_thresh_c: int = 2

    # Grid analysis parameters
    grid_filter_size: int = 15
    feature_spacing_mm: float = 0.5

    # Conrtast parameters
    clip_limit: float = 3.0
    tile_grid_size: int = 8

    # Morphological operations
    morph_kernel: int = 3

    # Regularization thresholds
    min_dot_area: float = 30.0
    min_circularity: float = 0.7
    spacing_factor: float = 1.5

    # Visualization control
    show_intermediate: bool = False


@dataclass
class FeatureDetectParams:
    """Parameters for Subpixel refinement and feature alignment"""

    subpx_window: int = 5
    subpx_zero_zone: int = -1
    subpx_max_iter: int = 30
    subpx_epsilon: float = 0.001


@dataclass
class FeatureAlignParams:
    """Parameters for feature alignment to grid"""

    grid_spacing_mm = 1.0
    ref_point_dist_thresh = 0.5

    max_iterations: int = 400
    neighbor_count: int = 5
    distance_ratio_thresh: float = 1.3
    vec_round_thresh: float = 0.0
    grid_interp_step: float = 0.5


def load_image(image_path: Union[str, Path]) -> np.ndarray[np.uint8]:
    """Load and preprocess an image from file.

    Args:
        image_path: Path to image file (.jpg, .png, .npy)

    Returns:
        Image as uint8 numpy array

    Raises:
        ValueError: If image loading fails
        Exception: For other loading/processing errors
    """
    try:
        path = Path(image_path)
        if path.suffix == ".npy":
            image = np.load(str(path)).copy()
            if np.max(image) <= 1.0:
                image = (image * 255).astype(np.uint8)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image: {path}")

        return image.astype(np.uint8)
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        raise


class ImageProcessor:
    """Image processing utilities for feature detection.

    Handles image preprocessing, vignetting correction, and grid analysis.

    Args:
        image: Input image as uint8 numpy array
        params: Optional processing parameters
    """

    def __init__(
        self, image: np.ndarray, params: Optional[ImageProcessParams] = None
    ) -> None:
        self._raw_image = image.astype(np.uint8)
        self._params = params or ImageProcessParams()

    def _filter_image(self) -> np.ndarray:
        """Preprocess image for feature detection.

        Returns:
            Binary image with regularized feature dots
        """
        # Prepare image processing steps first
        processed_stages = []

        # Original image
        processed_stages.append(("Original", self._raw_image))

        # Contrast enhancement
        lab = cv2.cvtColor(self._raw_image, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self._params.clip_limit,
            tileGridSize=(
                self._params.tile_grid_size,
                self._params.tile_grid_size,
            ),
        )
        cl = clahe.apply(lightness)
        processed_img = cv2.merge((cl, a, b))
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_LAB2BGR)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        processed_stages.append(("CLAHE Enhancement", processed_img))

        # Gaussian blur
        blurred_img = cv2.GaussianBlur(
            processed_img,
            (self._params.gaussian_kernel, self._params.gaussian_kernel),
            1,
        )
        processed_stages.append(("Gaussian Blur", blurred_img))

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self._params.adapt_thresh_block,
            self._params.adapt_thresh_c,
        )
        processed_stages.append(("Adaptive Threshold", binary))

        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self._params.morph_kernel,
                self._params.morph_kernel,
            ),
        )
        morph_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        processed_stages.append(("Morphological Open", morph_open))

        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                self._params.morph_kernel,
                self._params.morph_kernel,
            ),
        )
        eroded = cv2.erode(morph_open, kernel_erode, iterations=1)

        # Add regularization step
        regularized = self._regularize_features(eroded)
        processed_stages.append(("Regularized Dots", regularized))

        # Show visualization if enabled
        if self._params.show_intermediate:
            self._visualize_processing_steps(processed_stages)

        return regularized

    def _regularize_features(
        self, binary_image: np.ndarray[np.uint8]
    ) -> np.ndarray[np.uint8]:
        """Regularize detected dots by making them uniform circles with improved filtering.

        Args:
            binary_image: Binary image containing the dots

        Returns:
            Regularized binary image with uniform circular dots
        """

        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # First pass: collect valid dots and their properties
        valid_centers = []
        areas = []

        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            # More stringent circularity check
            if (
                area > self._params.min_dot_area
                and circularity > self._params.min_circularity
            ):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    valid_centers.append((cx, cy))
                    areas.append(area)

        if not valid_centers:
            return np.zeros_like(binary_image)

        # Calculate grid properties
        centers = np.array(valid_centers)
        median_area = np.median(areas)

        # Find typical grid spacing using nearest neighbors
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=2).fit(centers)
        distances, _ = nbrs.kneighbors(centers)
        typical_spacing = np.median(distances[:, 1])

        # Filter dots based on grid alignment
        final_centers = []
        for center in centers:
            neighbors = centers[
                np.where(
                    np.linalg.norm(centers - center, axis=1)
                    < typical_spacing * self._params.spacing_factor
                )[0]
            ]
            if len(neighbors) >= 2:  # At least two nearby neighbors
                final_centers.append(center)

        # Create output image with uniform circles
        result = np.zeros_like(binary_image)
        dot_radius = int(np.sqrt(median_area / np.pi))

        for x, y in final_centers:
            cv2.circle(result, (int(x), int(y)), dot_radius, 255, -1)

        return result

    def _visualize_processing_steps(
        self, stages: List[Tuple[str, np.ndarray[np.uint8]]]
    ) -> None:
        """Visualize all intermediate processing steps."""
        plt.figure(figsize=(15, 10))

        for idx, (title, img) in enumerate(stages, 1):
            plt.subplot(3, 3, idx)
            plt.title(title)
            # Use jet colormap for distance transform, gray for others
            if "Distance Transform" in title:
                plt.imshow(img, cmap="jet")
            else:
                plt.imshow(img, cmap="gray")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


class FeatureDetector:
    """Extract and refine circular features from binary image.

    Args:
        bin_image: Binary input image
        raw_image: Original grayscale image
    """

    def __init__(
        self,
        bin_image: np.ndarray[np.uint8],
        raw_image: np.ndarray[np.uint8],
        params: Optional[FeatureDetectParams] = None,
    ) -> None:
        self._bin_image = bin_image
        self._raw_image = raw_image
        self._params = params or FeatureDetectParams()

    def _detect_features(self) -> np.ndarray:
        """Extract feature points from binary image.

        Returns:
            Array of refined feature point coordinates (Nx2)
        """
        contours, _ = cv2.findContours(
            self._bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        features = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                features.append({"contour": c, "center": (cx, cy)})

        # # Refine features
        # refined_points = self._refine_feature_points(features, self._raw_image)

        # # Create visualization
        # self._create_visualization(features, refined_points)

        features = np.array([feature["center"] for feature in features])
        features = np.array(features, dtype=np.float32).reshape(-1, 2)

        return features

    def _refine_feature_points(
        self, features: List[Dict[str, Any]], gray_image: np.ndarray[np.uint8]
    ) -> np.ndarray:
        """Refine feature points to subpixel accuracy."""
        if not features:
            return np.array([])

        points = np.array([feature["center"] for feature in features], dtype=np.float32)

        window_size = (self._params.subpx_window,) * 2
        zero_zone = (self._params.subpx_zero_zone,) * 2
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            self._params.subpx_max_iter,
            self._params.subpx_epsilon,
        )

        refined_points = cv2.cornerSubPix(
            gray_image, points, window_size, zero_zone, criteria
        )

        return refined_points

    def _create_visualization(
        self,
        features: List[Dict[str, Any]],
        refined_points: np.ndarray,
    ) -> None:
        """Create visualization of detected dots."""
        result_image = cv2.cvtColor(self._raw_image, cv2.COLOR_GRAY2BGR)

        for feature in features:
            cv2.drawContours(
                result_image,
                [feature["contour"]],
                -1,
                (0, 255, 0),
                1,
            )

        for point in refined_points:
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.drawMarker(
                result_image,
                (x, y),
                (255, 0, 0),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=8,
                thickness=1,
            )


class FeatureAligner:
    """Align detected features to a regular grid pattern.

    Args:
        FeatureAlignParams
    """

    def __init__(self, params: Optional[FeatureAlignParams] = None) -> None:
        self._params = params or FeatureAlignParams()

    def _align_points_to_grid(
        self,
        centroids: np.ndarray,
        ref_point: np.ndarray,
        folder_ind: int,
        step: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align detected feature points to regular grid and get their correspondly object points.

        Args:
            centroids: Centroid coordinates of detected features
            ref_point: Reference point (origin) for alignment
            folder_ind: Folder index to set z-coordinate
            step: Step size for z-coordinate

        Returns:
            valid_features: Array of shape (N,2) containing centroid coordinates of valid features
            valid_obj_points: Array of shape (N,3) containing corresponding 3d coordinates of valid features
            centered_features: Array of shape (N,2) containing centered feature points
        """

        # Initialize object points array with 3 dimensions (x, y, z)
        obj_points = np.empty((centroids.shape[0], 3))
        obj_points.fill(np.nan)

        # Set the origin point directly using ref_point
        _, origin_idx = self.get_dists(centroids, ref_point, k=1)
        obj_points[origin_idx] = [
            0,
            0,
            folder_ind * step,
        ]  # Set z=folder_ind * step for origin

        iterations = 0

        while (
            np.isnan(obj_points[:, :2].sum())
            and iterations < self._params.max_iterations
        ):
            new_assignments = False
            for i, c in enumerate(centroids):
                if not np.isnan(obj_points[i, 0]):
                    dists, neighbor_idx = self.get_dists(
                        centroids, c, k=self._params.neighbor_count
                    )
                    min_dist = np.min(dists[1 : self._params.neighbor_count])
                    vecs = centroids[neighbor_idx[1 : self._params.neighbor_count]] - c
                    norms = np.linalg.norm(vecs, axis=1)
                    vecs = vecs / np.column_stack((norms, norms))
                    vecs /= np.sqrt(2) * 0.5
                    for j in range(self._params.neighbor_count - 1):
                        neighbor = neighbor_idx[j + 1]
                        if np.isnan(obj_points[neighbor, 0]):
                            if (
                                dists[j + 1] / min_dist
                                < self._params.distance_ratio_thresh
                            ):
                                if (
                                    np.round(vecs[j, 1], 0) * np.round(vecs[j, 0], 0)
                                    == self._params.vec_round_thresh
                                ):
                                    # Assign x,y coords and set z=0
                                    new_xy = (
                                        np.round(vecs[j], 0)
                                        * self._params.grid_spacing_mm
                                        + obj_points[i, :2]
                                    )
                                    obj_points[neighbor] = [
                                        new_xy[0],
                                        new_xy[1],
                                        folder_ind * step,
                                    ]
                                    new_assignments = True
            if not new_assignments:
                break
            iterations += 1

        # Find nearest neighbors to establish grid directions
        dists, neighs = self.get_dists(
            centroids, centroids[origin_idx], k=self._params.neighbor_count
        )

        # Sort points by distance from origin
        sorted_indices = np.argsort(
            np.linalg.norm(centroids - centroids[origin_idx], axis=1)
        )

        # Assign grid coordinates
        for idx in sorted_indices:
            if np.isnan(obj_points[idx][0]):
                # Get nearest assigned neighbors
                assigned_mask = ~np.isnan(obj_points[:, 0])
                if np.sum(assigned_mask) >= 2:
                    dists, neighs = self.get_dists(
                        centroids[assigned_mask], centroids[idx], k=2
                    )
                    # Calculate grid position based on nearest assigned neighbors
                    assigned_coords = obj_points[assigned_mask][neighs]
                    relative_pos = np.round(
                        (centroids[idx] - centroids[assigned_mask][neighs[0]])
                        / self._params.grid_spacing_mm
                    )
                    new_xy = (
                        assigned_coords[0, :2]
                        + relative_pos * self._params.grid_spacing_mm
                    )
                    obj_points[idx] = [new_xy[0], new_xy[1], folder_ind * step]

        # Filter out points without assigned coordinates
        valid_mask = ~np.isnan(obj_points).any(axis=1)
        valid_features = centroids[valid_mask]
        valid_obj_points = obj_points[valid_mask]

        centered_features = valid_features - ref_point

        unique_coords, unique_indices = np.unique(
            valid_obj_points, axis=0, return_index=True
        )

        valid_features = valid_features[unique_indices]
        valid_obj_points = unique_coords
        centered_features = centered_features[unique_indices]

        return valid_features, valid_obj_points, centered_features

    @staticmethod
    def get_dists(points, target_point, k):
        """Get k nearest neighbors and distances for a target point.

        Args:
            points: Array of points to search
            target_point: Target point for nearest neighbors
            k: Number of neighbors to return
        Returns:
            distances: Array of distances to nearest neighbors
            indices: Array of indices of nearest neighbors
        """
        if len(points) < k:
            raise ValueError(f"Need at least {k} points for neighbor search")

        # Ensure points is 2D array
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Ensure target_point is 2D array
        target_point = np.asarray(target_point)
        if target_point.ndim == 1:
            target_point = target_point.reshape(1, -1)

        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, indices = nbrs.kneighbors(target_point)
        return distances[0], indices[0]

    @staticmethod
    def sample_deviation(
        path: np.ndarray,
        deviation_field: np.ndarray,
        grid_points: np.ndarray,
    ) -> np.ndarray:
        """Sample deviation field along a path."""
        from scipy.interpolate import griddata

        params = FeatureAlignParams()

        # Create grid for interpolation
        x = np.arange(
            np.min(grid_points[0]), np.max(grid_points[0]) + 1, params.grid_interp_step
        )
        y = np.arange(
            np.min(grid_points[1]), np.max(grid_points[1]) + 1, params.grid_interp_step
        )
        X, Y = np.meshgrid(x, y)

        # Interpolate deviation field
        deviation_interp = griddata(
            grid_points.T, deviation_field, (X, Y), method="nearest"
        )

        # Sample deviation along the path
        return griddata(
            (X.flatten(), Y.flatten()),
            deviation_interp.flatten(),
            (path[0], path[1]),
            method="nearest",
        )

    @staticmethod
    def _visualize_alignment(
        self,
        image: np.ndarray,
        detected_features: np.ndarray,
        aligned_features: np.ndarray,
    ) -> None:
        """Visualize the alignment results."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Before alignment plot
        axs[0].imshow(image, cmap="gray")
        axs[0].scatter(
            detected_features[:, 0],
            detected_features[:, 1],
            c="red",
            marker="o",
            s=50,
            label="Detected Features",
        )
        axs[0].set_title("Before Alignment")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # After alignment plot
        axs[1].scatter(
            aligned_features[0],
            aligned_features[1],
            c="blue",
            marker="x",
            s=50,
            label="Aligned Features",
        )
        axs[1].set_title("After Alignment (Grid)")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
