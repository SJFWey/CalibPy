import json
from pathlib import Path

import numpy as np
from utils.calibrator import Optimizer
from utils.optimizer_params import OptimizerParams, CalibrationFlags, ParamsGuess
from utils.helpers import extract_and_save_features


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
        with open(json_file, "r", encoding="utf-8") as f:
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
    params_guess = ParamsGuess(
        image_size=image_size,
        fx=703.0,  # Initial focal length guess
        fy=697.0,
        cx=image_size[0] // 2,  # Image center
        cy=image_size[1] // 2,
    )

    # Create optimizer params with custom settings
    optimizer_params = OptimizerParams(
        max_iter=30,
        step=0.2,
        verbose=1,
        outlier_threshold=1.5,
        max_outlier_iter=3,
        opt_method="lm",
        loss="linear",
    )

    # Create calibration flags with correct parameter names
    flags = CalibrationFlags(
        fix_aspect_ratio=False,  # Allow fx and fy to be different
        estimate_principal=True,
        estimate_skew=False,
        estimate_distort=np.array([True, True, True, True, True]),
    )

    # Create optimizer and run calibration
    optimizer = Optimizer(
        calib_data,
        params_guess=params_guess,
        optimizer_params=optimizer_params,
        flags=flags,
    )

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
        "mean_error": calib_results["mean_reproj_error"],
        "std_error": calib_results["std_reproj_error"],
        "extrinsics": calib_results["extrinsics"],
        "param_correlations": calib_results.get("param_correlations", None),
    }

    return results


if __name__ == "__main__":
    # src_folder = Path("calibpy/data/new_images/")
    # for folder in src_folder.iterdir():
    #     if folder.name == "d=4, s=0.25":
    #         output_path = f"calibpy/data/feature_data/test/{folder.name}"
    #         extract_and_save_features(folder, output_path, if_save=True, if_plot=False)

    data_path = Path("calibpy/data/feature_data/test/d=5, s=0.25")

    results = calibrate_camera(data_path)

    print("\nCalibration Results:")
    print("Camera Matrix:")
    print(results["camera_matrix"])
    print("\nDistortion Coefficients:")
    print(results["dist_coeffs"])
    print("\nMean Reprojection Error:")
    print(results["mean_error"])
    print("\nStd of Reprojection Error:")
    print(results["std_error"])
    print("\nExtrinsics per Image:")
    for i, ext in zip(range(len(results["extrinsics"])), results["extrinsics"]):
        print(f"\nImage {i}:")
        print(
            f"  Roll, Pitch, Yaw (deg): {ext['euler_angles'][0]:.2f}, {ext['euler_angles'][1]:.2f}, {ext['euler_angles'][2]:.2f}"
        )
        print(
            f"  Translation (mm): {ext['tvec'][0]:.2f}, {ext['tvec'][1]:.2f}, {ext['tvec'][2]:.2f}"
        )

    # Improved printing of correlation matrix using descriptive labels and pandas formatting
    import pandas as pd

    # Example: define intrinsics and extrinsics labels based on optimized parameters
    intrinsics = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    num_extrinsics = len(results["extrinsics"])
    extrinsics = (
        [f"Img{i} rvec_x" for i in range(num_extrinsics)]
        + [f"Img{i} rvec_y" for i in range(num_extrinsics)]
        + [f"Img{i} rvec_z" for i in range(num_extrinsics)]
        + [f"Img{i} tvec_x" for i in range(num_extrinsics)]
        + [f"Img{i} tvec_y" for i in range(num_extrinsics)]
        + [f"Img{i} tvec_z" for i in range(num_extrinsics)]
    )
    param_labels = intrinsics + extrinsics

    if results["param_correlations"] is not None:
        n_params = len(param_labels)
        if results["param_correlations"].shape == (n_params, n_params):
            corr_df = pd.DataFrame(
                results["param_correlations"], index=param_labels, columns=param_labels
            )
            print("\nParameter Correlation Matrix:")
            print(corr_df.iloc[:9, :9].round(2))
        else:
            print(
                "Parameter correlation matrix dimensions do not match parameter labels."
            )
    else:
        print("No parameter correlations available.")
