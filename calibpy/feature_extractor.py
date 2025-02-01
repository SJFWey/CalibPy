from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import KDTree
import cv2

from utils.extractor_utils import (
    FeatureAligner,
    FeatureDetector,
    ImageProcessor,
    ImageProcessParams,
)
from utils.helpers import adaptive_feature_extractor


def find_nearest_feature(point: np.ndarray, features: np.ndarray) -> np.ndarray:
    """
    Find the nearest feature point to the given point.

    Args:
        point: Target point coordinates
        features: Array of feature points

    Returns:
        Nearest feature point coordinates
    """
    distances = np.linalg.norm(features - point, axis=1)
    nearest_idx = np.argmin(distances)
    return features[nearest_idx]


def create_descriptors(points: np.ndarray, k: int = 4) -> np.ndarray:
    # Creates neighborhood-based descriptors for each point
    if len(points) < k + 1:
        return np.zeros((len(points), 2 * k))
    tree = KDTree(points)
    descriptors = []
    for i, point in enumerate(points):
        dists, indices = tree.query([point], k + 1)
        neighbor_pts = points[indices[0][1:]]
        rel_vectors = neighbor_pts - point
        sorted_idx = np.argsort(dists[0][1:])
        sorted_vecs = rel_vectors[sorted_idx]
        closest_dist = dists[0][1]
        if closest_dist < 1e-6:
            desc = np.zeros(k * 2)
        else:
            desc = (sorted_vecs / closest_dist).flatten()
        descriptors.append(desc)
    return np.array(descriptors)


# def track_ref_point(
#     first_pts: np.ndarray,
#     current_pts: np.ndarray,
#     ref_point: np.ndarray,
#     k: int = 4,
#     ransac_thresh: float = 5.0,
# ) -> Optional[Tuple[float, float]]:

#     desc_first = create_descriptors(first_pts, k)
#     desc_current = create_descriptors(current_pts, k)

#     matches = []
#     for i, desc in enumerate(desc_current):
#         distances = np.linalg.norm(desc_first - desc, axis=1)
#         matches.append((np.argmin(distances), i))
#     src_pts = first_pts[[m[0] for m in matches]].astype(np.float32)
#     dst_pts = current_pts[[m[1] for m in matches]].astype(np.float32)

#     if len(src_pts) < 4:
#         return None
#     H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

#     if H is None:
#         return None
#     proj = cv2.perspectiveTransform(
#         ref_point.reshape(1, 1, 2).astype(np.float32), H
#     ).reshape(-1)

#     dists = np.linalg.norm(current_pts - proj, axis=1)

#     return tuple(current_pts[np.argmin(dists)].tolist())


def track_ref_point(
    prev_pts: np.ndarray,
    current_pts: np.ndarray,
    ref_point: np.ndarray,
    k: int = 6,
    ransac_thresh: float = 5.0,
) -> Optional[Tuple[float, float]]:
    desc_prev = create_descriptors(prev_pts, k)
    desc_current = create_descriptors(current_pts, k)

    matches = []
    for i, desc in enumerate(desc_current):
        distances = np.linalg.norm(desc_prev - desc, axis=1)
        matches.append((np.argmin(distances), i))
    src_pts = prev_pts[[m[0] for m in matches]].astype(np.float32)
    dst_pts = current_pts[[m[1] for m in matches]].astype(np.float32)

    if len(src_pts) < 4:
        return None
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if H is None:
        return None
    proj = cv2.perspectiveTransform(
        ref_point.reshape(1, 1, 2).astype(np.float32), H
    ).reshape(-1)

    dists = np.linalg.norm(current_pts - proj, axis=1)

    return tuple(current_pts[np.argmin(dists)].tolist())


def extract_features(
    image: np.ndarray,
    folder_ind: int,
    is_ref_image: bool = False,
    manual_select: bool = False,
    prev_refpoint: Optional[np.ndarray] = None,
    prev_features: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Extract point features from a single image with point tracking.

    Args:
        image: Input image as numpy array (uint8)
        folder_ind: Image folder index
        is_ref_image: If image is the first image, where reference point should be selected
        manual_slect: If manually select reference point
        prev_refpoint: Previous reference point for tracking

    Returns:
        img_points: Array of 2D image points (Nx2)
        obj_points: Array of corresponding 3D object points (Nx3)
        refpoint_estimate: Reference point coordinates if is_ref_image=True
        features: Array of detected feature centroids
    """

    # Initialize variables
    refpoint_estimate = None

    # # Generate binary image
    # processor = ImageProcessor(image, ImageProcessParams(show_intermediate=False))
    # bin_image = processor._filter_image()

    # # Extract features
    # detector = FeatureDetector(bin_image, image)
    # features = detector._detect_features()
    features = adaptive_feature_extractor(image)

    if is_ref_image and manual_select:
        import matplotlib.pyplot as plt

        clicked = None

        while not clicked:
            plt.figure()
            plt.imshow(image, cmap="gray")
            plt.title(
                "Select which point should be tracked (or wait for auto-selection)"
            )

            clicked = plt.ginput(1, timeout=30)
            if clicked:
                clicked_point = np.array(clicked[0])
                refpoint_estimate = find_nearest_feature(clicked_point, features)
                plt.plot(
                    refpoint_estimate[0],
                    refpoint_estimate[1],
                    "rx",
                    markersize=10,
                    label="Selected Point",
                )
                plt.legend()
                plt.draw()
                plt.pause(1)
        plt.close()

    elif not is_ref_image and prev_refpoint is not None:
        # Attempt descriptor-based tracking
        tracked = track_ref_point(
            prev_pts=prev_features, current_pts=features, ref_point=prev_refpoint
        )
        if tracked is not None:
            refpoint_estimate = np.array(tracked)
        else:
            # Fallback to nearest-feature approach
            print(
                "Warning: descriptor-based tracking failed, falling back to nearest_feature"
            )
            refpoint_estimate = find_nearest_feature(prev_refpoint, features)

        # # Check tracking validity
        # tracking_dist = np.linalg.norm(refpoint_estimate - prev_refpoint)
        # feature_spacing = np.min(
        #     [
        #         np.linalg.norm(prev_features[i] - prev_features[j])
        #         for i in range(len(prev_features))
        #         for j in range(i + 1, len(prev_features))
        #     ]
        # )
        # if tracking_dist > 0.5 * feature_spacing:
        #     raise ValueError("Tracking failed, too much away from last points")
    else:
        raise ValueError("Try to select the reference point again")

    # Analyze grid points
    aligner = FeatureAligner()
    img_points, obj_points, _ = aligner._align_points_to_grid(
        features, refpoint_estimate, folder_ind
    )

    # Prepare output points
    img_points = np.reshape(img_points, (-1, 2)).astype(np.float64)

    return (
        img_points,
        obj_points,
        refpoint_estimate,
        features,
    )  # features before alignment
