import logging
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from utils.feature_extractor import FeatureExtractor


class FeatureTracker:
    @staticmethod
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

    @staticmethod
    def create_descriptors(points: np.ndarray, k: int = 4) -> np.ndarray:
        """
        Create descriptors for the given points.

        Args:
            points: Array of points
            k: Number of nearest neighbors to consider

        Returns:
            Array of descriptors
        """
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

    def track_ref_point(
        self,
        prev_pts: np.ndarray,
        current_pts: np.ndarray,
        ref_point: np.ndarray,
        k: int = 6,
        ransac_thresh: float = 5.0,
    ) -> Optional[Tuple[float, float]]:
        """
        Track a reference point using descriptor matching and homography estimation.

        Args:
            prev_pts: Previous feature points
            current_pts: Current feature points
            ref_point: Reference point coordinates
            k: Number of nearest neighbors to consider
            ransac_thresh: RANSAC threshold

        Returns:
            Tracked reference point coordinates or None if tracking fails
        """
        desc_prev = self.create_descriptors(prev_pts, k)
        desc_current = self.create_descriptors(current_pts, k)

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

    def select_reference_point(
        self, image: np.ndarray, features: np.ndarray
    ) -> np.ndarray:
        """
        Allow manual selection of the reference point.

        Args:
            image: Input image as numpy array (uint8)
            features: Array of feature points

        Returns:
            Selected reference point coordinates
        """
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
                refpoint_estimate = self.find_nearest_feature(clicked_point, features)
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
        return refpoint_estimate

    def track_features(
        self,
        image: np.ndarray,
        img_serial_num: int,
        feature_extractor: FeatureExtractor,
        manual_select: bool = False,
        prev_refpoint: Optional[np.ndarray] = None,
        prev_features: Optional[np.ndarray] = None,
        is_ref_image: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Extract features and track reference point.

        Args:
            image: Input image as numpy array (uint8)
            img_serial_num: Image folder index
            feature_extractor: Instance of FeatureExtractor class
            manual_select: If manually select reference point
            prev_refpoint: Previous reference point for tracking
            prev_features: Previous features for tracking
            is_ref_image: If image is the first image, where reference point should be selected

        Returns:
            img_points: Array of 2D image points (Nx2)
            obj_points: Array of corresponding 3D object points (Nx3)
            refpoint_estimate: Reference point coordinates
            features: Array of detected feature centroids
        """
        refpoint_estimate = None
        # Initial extraction without reference point
        _, _, features, _ = feature_extractor.get_2d_3d_points(image, img_serial_num, None)

        if is_ref_image and (manual_select or prev_refpoint is None):
            refpoint_estimate = self.select_reference_point(image, features)
        elif not is_ref_image and prev_refpoint is not None:
            # Attempt descriptor-based tracking
            tracked = self.track_ref_point(
                prev_pts=prev_features, current_pts=features, ref_point=prev_refpoint
            )
            if tracked is not None:
                refpoint_estimate = np.array(tracked)
            else:
                # Fallback to nearest-feature approach
                logging.warning(
                    "Descriptor-based tracking failed, falling back to nearest feature"
                )
                refpoint_estimate = self.find_nearest_feature(prev_refpoint, features)
        else:
            raise ValueError("Try to select the reference point again")

        # Re-extract features with the tracked reference point
        img_points, obj_points, _ = feature_extractor.get_2d_3d_points(
            image, img_serial_num, refpoint_estimate
        )

        return img_points, obj_points, refpoint_estimate, features
