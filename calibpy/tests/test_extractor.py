import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from calibpy.src.feature_extractor import extract_features
from calibpy.utils.extractor_utils import (
    FeatureDetector,
    ImageProcessor,
    ImageProcessParams,
    load_image,
)
from calibpy.utils.helpers import adaptive_feature_extractor

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Unable to open default EUDC font")
warnings.filterwarnings("ignore", message="iCCP: known incorrect sRGB profile")


def test_feature_extraction():
    """Test feature extraction on multiple images and visualize the results."""
    # Test image paths
    test_image_folders = [str(i) for i in range(30)]
    # test_image_folders = [str(i) for i in [0, 6, 24]]
    base_path = Path("calibpy/tests/test_images")

    image_paths = []
    for folder in test_image_folders:
        img_path = Path(base_path / folder / "img0.npy")
        image_paths.append(img_path)

        # Load and process image
        raw_image = load_image(img_path)
        # params = ImageProcessParams(show_intermediate=True)
        # processor = ImageProcessor(raw_image, params)
        # binary_image = processor._filter_image()

        # detector = FeatureDetector(binary_image, raw_image)
        # features = detector._detect_features()
        img_points = adaptive_feature_extractor(raw_image)

        print(f"Detected features: {len(img_points)}")


def test_point_tracking():
    """Test point tracking on multiple images."""
    # Test image paths
    test_image_folders = [str(i) for i in range(0, 29)]
    test_image_indices = [0]
    base_path = Path("calibpy/tests/test_images")
    image_paths = []
    for folder in test_image_folders:
        image_paths.extend(
            [base_path / folder / f"img{ind}.npy" for ind in test_image_indices]
        )

    ref_points = []

    for img_path in image_paths:
        # Load and process image
        raw_image = load_image(img_path)

        is_ref_image = img_path.parent.stem == "0"

        img_points, obj_points, refpoint_estimate, centroids = extract_features(
            raw_image,
            is_ref_image=is_ref_image,
            manual_select=True,
            prev_refpoint=ref_points[-1] if ref_points else None,
        )

        ref_points.append(refpoint_estimate)

        plt.figure()
        plt.imshow(raw_image, cmap="gray")
        plt.plot(
            centroids[:, 0],
            centroids[:, 1],
            "go",
            markerfacecolor="none",
            label="Detected Features",
            markersize=7,
        )
        plt.plot(
            img_points[:, 0],
            img_points[:, 1],
            "bo",
            label="Valid Features",
            markersize=3,
        )
        plt.plot(
            refpoint_estimate[0],
            refpoint_estimate[1],
            "r+",
            markersize=10,
            label="Reference Point",
        )
        plt.title(f"{img_path.parent.stem}/{img_path.stem}")

        # plt.figure(2)
        # plt.scatter(obj_points[0], obj_points[1], c="r", marker="o", s=15)
        # plt.xlabel("X (mm)")
        # plt.ylabel("Y (mm)")
        # plt.title("Grid Coordinates")
        # plt.grid(True, alpha=0.5)
        # plt.axis("equal")

        plt.show()

        # print(f"Extracted {len(img_points)} points")


if __name__ == "__main__":
    test_feature_extraction()
    # test_point_tracking()
