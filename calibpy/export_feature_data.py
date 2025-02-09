from pathlib import Path
from scipy.io import savemat
import json
import numpy as np
import cv2

from utils.feature_extractor_utils import ImageProcessor, FeatureExtractor


def extract_and_save_features(
    image_folder: str | Path,
    output_path: str,
    if_save: bool = False,
    if_plot: bool = False,
):
    """
    Extract features from a series of images in a folder and save in MATLAB format.

    Args:
        image_folder (str | Path): Path to folder containing images
        output_path (str): Path to save the extracted features
        if_save (bool, optional): Flag to save features. Defaults to False.
        if_plot (bool, optional): Flag to plot features. Defaults to False.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    img_files = sorted([img for img in image_folder.iterdir() if img.suffix == ".npy"])

    for i in range(len(img_files)):
        img = ImageProcessor.load_image(img_files[i])
        print(f"Processing {img_files[i].name}")

        # Extract features
        feature_extractor = FeatureExtractor(img)
        img_points, obj_points, _, ref_point = feature_extractor._extract_3d_points(
            img_serial_num=i
        )

        # --- Start: Outlier filtering step ---
        from scipy.spatial import cKDTree

        radius = 400
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        def filter_outliers(points, tol=0.5, radius_multiplier=0.9):
            """
            Filters outliers based on distance to neighbors and proximity to the edge.

            Args:
                points: The input points (img_points in this case).
                tol: Tolerance for neighbor-based outlier removal.
                radius_multiplier:  A factor to shrink the radius for edge detection.  Values closer to 1.0
                                    remove fewer points near the edge.

            Returns:
                A boolean mask indicating the inliers (True for inliers, False for outliers).
            """
            # 1. Neighbor-based outlier removal
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=3)
            median_dist = np.median(distances[:, 1])
            neighbor_mask = abs(distances[:, 1] - median_dist) < tol * median_dist

            # 2. Edge proximity filtering
            #  Calculate distance of each point from the center
            point_distances = np.sqrt(
                (points[:, 0] - center[0]) ** 2 + (points[:, 1] - center[1]) ** 2
            )
            #  Create a mask for points near the edge (within the shrunk radius)
            edge_mask = point_distances <= radius * radius_multiplier

            # 3. Combine masks:  A point is an inlier if it passes BOTH checks
            combined_mask = neighbor_mask & edge_mask
            return combined_mask

        mask = filter_outliers(img_points)
        img_points = img_points[mask]
        obj_points = obj_points[mask]
        # --- End: New outlier filtering step ---

        # Visualize the features
        if if_plot:
            import matplotlib.pyplot as plt

            def on_key(event):
                if event.key == "q":
                    plt.close("all")
                    raise KeyboardInterrupt

            fig, ax = plt.subplots()
            ax.imshow(img, cmap="gray")
            ax.scatter(img_points[:, 0], img_points[:, 1], c="r", s=5)
            ax.scatter(ref_point[0], ref_point[1], c="g", s=5)
            plt.title(f"Features in {img_files[i].stem}")

            if if_save:
                plt.savefig(
                    str(output_path / f"features_{img_files[i].stem}.jpg"), dpi=300
                )

            fig.canvas.mpl_connect("key_press_event", on_key)
            # plt.show()
            plt.close()

        if if_save:
            # Save individual feature data in mat format
            savemat(
                output_path / f"feature_data_{img_files[i].stem}.mat",
                {
                    "img_points": img_points,
                    "obj_points": obj_points,
                },
            )


if __name__ == "__main__":
    src_folder = Path("calibpy/data/new_images/")

    for folder in src_folder.iterdir():
        output_path = f"calibpy/data/feature_data/test/{folder.name}"
        extract_and_save_features(folder, output_path, if_save=True, if_plot=True)
        # extract_and_save_features(folder, output_path, if_save=False, if_plot=True)
        print("\n")
