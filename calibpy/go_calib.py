import json
from pathlib import Path

import numpy as np
from utils.calibrator_temp import InitParams, Optimizer

from export_feature_data import extract_and_save_features


def load_calib_data(feature_data_path: str | Path) -> dict:
    """
    Load calibration data from JSON files in the specified directory.
    (Using saved file naming pattern: "calib_data_*.json")

    Args:
        feature_data_path: Path to directory containing calibration data files

    Returns:
        dict: Dictionary containing object and image points for each frame
    """
    feature_path = Path(feature_data_path)
    json_files = sorted(feature_path.glob("*.json"))

    calib_data = {
        "obj_points": {},
        "img_points": {},
    }

    for i, json_file in enumerate(json_files):
        with open(json_file, "r") as f:
            data = json.load(f)
        calib_data["obj_points"][i] = np.array(data["obj_points"])
        calib_data["img_points"][i] = np.array(data["img_points"])

    return calib_data


def calibrate_camera(feature_data_path: str | Path, image_size=(1280, 720)) -> dict:
    """
    Perform camera calibration using feature data from calibrator_temp.
    First extract features from images and then calibrate.

    Args:
        feature_data_path: Path to directory where feature data will be saved/read.
        image_size: Tuple of (width, height) for the images.

    Returns:
        dict: Calibration results containing camera parameters.
    """
    # Load calibration data
    calib_data = load_calib_data(feature_data_path)

    if not calib_data["img_points"]:
        raise ValueError("No feature data found in the specified directory")

    # Initialize camera parameters with default values
    init_params = InitParams(
        IMG_WIDTH=image_size[0],
        IMG_HEIGHT=image_size[1],
        INITIAL_FOV=60,
        fx=703.0,  # Let the algorithm estimate initial focal length
        fy=697.0,
        cx=None,  # Let the algorithm use image center
        cy=None,
    )

    # Create optimizer and run calibration
    optimizer = Optimizer(init_params, calib_data)
    calib_results = optimizer.calibrate()

    # Format results to match expected output format
    results = {
        "camera_matrix": np.array(
            [
                [calib_results["fc"][0], calib_results["skew"], calib_results["pp"][0]],
                [0, calib_results["fc"][1], calib_results["pp"][1]],
                [0, 0, 1],
            ]
        ),
        "dist_coeffs": calib_results["dist_coeffs"],
        "rvecs": [ext["rvec"] for ext in calib_results["extrinsics"]],
        "tvecs": [ext["tvec"] for ext in calib_results["extrinsics"]],
        "residual": calib_results["residual"],
    }

    return results


if __name__ == "__main__":
    # src_folder = Path("calibpy/data/new_images/")

    # for folder in src_folder.iterdir():
    #     if folder.name == "d=2, s=0.25":
    #         output_path = f"calibpy/data/feature_data/test/{folder.name}"
    #         extract_and_save_features(folder, output_path, if_save=True, if_plot=False)

    data_path = Path("calibpy/data/feature_data/test/d=2, s=0.25")

    results = calibrate_camera(data_path)

    print("\nCalibration Results:")
    print("Camera Matrix:")
    print(results["camera_matrix"])
    print("\nDistortion Coefficients:")
    print(results["dist_coeffs"])
    print("\nReprojection Error:")
    print(results["residual"])
