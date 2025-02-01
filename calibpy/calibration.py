import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import cv2
from splib.calibration import stereoCalibration
from splib.utils import configLoader
from splib.utils.resourceManager import ResourceManager

from feature_extractor import extract_features
from utils.extractor_utils import load_image
from utils.helpers import export_features, getObjectPointsNNSingle, polyfit


def run_calibration(image_folders: Union[str, Path]):
    image_folders = Path(image_folders)

    if not image_folders.exists():
        raise FileNotFoundError(f"Path does not exist: {image_folders}")
    if not image_folders.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {image_folders}")

    cfg_file = "debugging.yaml"
    cfg_path = os.path.join("calibpy\\configs", cfg_file)
    configLoader.loadProjectConfig(cfg_path)
    resource_manager = ResourceManager.getInstance()
    CalShift = configLoader.initMeasurementClass(resource_manager["config"])

    ref_images = []
    calib_images = []

    cam_features = Path("calibpy\\data\\cam_feature_data_z=0.pkl")
    if cam_features.exists():
        with open(cam_features, "rb") as infile:
            cam_feature_data = pickle.load(infile)
    else:
        cam_feature_data = defaultdict(dict)
        ref_points = []  # Store reference points of each image for tracking

        folders = sorted(
            image_folders.iterdir(),
            key=lambda x: int(x.stem) if x.stem.isdigit() else -1,
        )

        # For test!!!
        # folders = folders[:10]

        prev_features = None
        for folder in folders:
            img_files = sorted(
                folder.iterdir(),
                key=lambda x: int(x.stem.replace("img", ""))
                if x.stem.startswith("img")
                else -1,
            )

            for img in img_files:
                if img.name == "img0.npy":
                    ref_img = load_image(img)
                    ref_images.append(ref_img)
                elif img.suffix == ".npy":
                    calib_img = load_image(img)
                    calib_images.append(calib_img)

            if ref_images and calib_images:
                CalShift.deleteImageStack()
                CalShift.addImages(calib_images)

                print(f"Getting projector map for {folder.stem}...")
                map = CalShift.getProjectorMap(calib_images)

                # img_points, obj_points = getObjectPointsNNSingle(
                #     ref_img, folder_ind=int(folder.stem), step=0.0
                # )
                is_ref_image = folder.stem == "0"
                manual_select = is_ref_image

                img_points, obj_points, ref_point, _ = extract_features(
                    ref_img,
                    folder_ind=int(folder.stem),
                    is_ref_image=is_ref_image,
                    manual_select=manual_select,
                    prev_refpoint=ref_points[-1] if ref_points else None,
                    prev_features=prev_features,
                )

                fitted_map, feature_mask = polyfit(
                    map, img_points, radius=50, dense_thresh=0.8
                )

                # plt.subplot(131)
                # plt.imshow(fitted_map[:, :, 0] + fitted_map[:, :, 1])
                # plt.title("Phasemap Synth")

                # plt.subplot(132)
                # plt.imshow(fitted_map[:, :, 0])
                # plt.title('Phasemap X')

                # plt.subplot(133)
                # plt.imshow(fitted_map[:, :, 1])
                # plt.title('Phasemap Y')

                # plt.show()
                # plt.close()

                filtered_img_points = img_points[feature_mask]
                filtered_obj_points = obj_points[feature_mask]

                # plt.figure(figsize=(12, 8))
                # plt.imshow(ref_img, cmap="gray", alpha=1.0)

                # if len(ref_images) > 1 and len(ref_points) > 0:
                #     plt.imshow(ref_images[-2], cmap="gray", alpha=0.2)
                #     plt.plot(
                #         ref_points[-1][0],
                #         ref_points[-1][1],
                #         "o",
                #         color="red",
                #         markersize=12,
                #         markerfacecolor="none",
                #         label="last ref point",
                #         linewidth=3.0,
                #     )

                # plt.plot(
                #     ref_point[0],
                #     ref_point[1],
                #     "o",
                #     color="deepskyblue",
                #     markersize=12,
                #     markerfacecolor="none",
                #     label="current ref point",
                #     linewidth=5.0,
                # )

                # plt.plot(
                #     img_points[:, 0],
                #     img_points[:, 1],
                #     "go",
                #     markersize=14,
                #     markerfacecolor="none",
                #     label="features",
                #     linewidth=3.0,
                # )

                # if len(ref_points) > 0:
                #     prev_ref = ref_points[-1]
                #     plt.arrow(
                #         prev_ref[0],
                #         prev_ref[1],
                #         ref_point[0] - prev_ref[0],
                #         ref_point[1] - prev_ref[1],
                #         color="red",
                #         width=0.5,
                #         head_width=3,
                #         label="tracking trajectory from last frame",
                #     )

                # plt.title(f"Features Tracking - Folder_{folder.stem}")
                # plt.legend()
                # plt.axis("image")
                # plt.show()

                ref_points.append(ref_point)
                prev_features = img_points

                cam_feature_data[folder.stem] = {
                    "projectorMap": fitted_map,
                    "img": ref_img,
                    "imagePoints": filtered_img_points,
                    "objectPoints": filtered_obj_points,
                }

        with open(cam_features, "wb") as outfile:
            pickle.dump(cam_feature_data, outfile)

    # Selecting some reliable camera feature data for calibration
    # selected_cam_feature_data = [cam_feature_data[str(i)] for i in range(15, 20)]
    selected_cam_feature_data = [cam_feature_data[str(i)] for i in [4, 5, 6, 7]]

    pro_K_guess = np.array(
        [
            [665, 0.0, 384],
            [0.0, 665, 384],
            [0.0, 0.0, 1.0],
        ]
    )
    pro_dist_guess = np.zeros((5, 1))

    proj_feature_data, mask_planes, _ = (
        stereoCalibration.get_projector_features(
            feature_data=selected_cam_feature_data,
            projector_resolution=(768, 768),
            pro_K_guess=pro_K_guess,
            pro_dist_guess=pro_dist_guess,
            removal_type="overall",
            removal_algorithm="3sigma",
        )
    )

    for i in range(len(selected_cam_feature_data)):
        if mask_planes[i]:
            cam_obj_pts = selected_cam_feature_data[i]["objectPoints"]
            cam_img_pts = selected_cam_feature_data[i]["imagePoints"]
            proj_obj_pts = proj_feature_data[i]["objectPoints"]

            # Keep only camera points that also exist in projector data
            keep_indices = []
            for idx, cam_pt in enumerate(cam_obj_pts):
                if any((proj_obj_pts == cam_pt).all(axis=1)):
                    keep_indices.append(idx)

            selected_cam_feature_data[i]["objectPoints"] = cam_obj_pts[keep_indices]
            selected_cam_feature_data[i]["imagePoints"] = cam_img_pts[keep_indices]

    # export_features(
    #     selected_cam_feature_data,
    #     proj_feature_data,
    #     mask_planes,
    #     "calibpy\\data\\feature_data/new",
    # )

    cam_list = []
    proj_list = []

    for num in range(len(selected_cam_feature_data)):
        if mask_planes[num] and len(proj_feature_data[num]["imagePoints"]) > 6:
            fdict_cam = selected_cam_feature_data[num]
            fdict_proj = proj_feature_data[num]

            proj_dict = {
                "imagePoints": fdict_proj["imagePoints"],
                "objectPoints": fdict_proj["objectPoints"],
            }
            cam_dict = {
                "imagePoints": fdict_cam["imagePoints"],
                "objectPoints": fdict_cam["objectPoints"],
            }

            cam_list.append(cam_dict)
            proj_list.append(proj_dict)

            # _ = selected_cam_feature_data[num]["projectorMap"]
            # xy = fdict_proj["imagePoints"]
            # obj = fdict_proj["objectPoints"]
            # plt.scatter(xy[:, 0], xy[:, 1])
            # plt.title(str(num))
            # for k in range(np.size(xy, axis=0)):
            #     plt.text(
            #         int(xy[k, 0]),
            #         int(xy[k, 1]),
            #         str(obj[k, 0:2]),
            #         c="black",
            #         fontsize=5.0,
            #     )
            # plt.show(block=True)

    init_cam_K = np.array([[943, 0, 640], [0, 943, 360], [0, 0, 1]], dtype=np.float32)
    init_cam_dist = np.zeros((5, 1), dtype=np.float32)
    init_cam_undist = -init_cam_dist

    init_proj_K = np.array([[665, 0, 384], [0, 665, 384], [0, 0, 1]], dtype=np.float32)
    init_proj_dist = np.zeros((5, 1), dtype=np.float32)
    init_proj_undist = -init_proj_dist

    calibration = stereoCalibration.stereo_calibration(
        cam_list,
        proj_list,
        (720, 1280),
        (768, 768),
        cam_calibration_1={
            "K": init_cam_K,
            "distortion": init_cam_dist,
            "undistortion": init_cam_undist,
        },
        cam_calibration_2={
            "K": init_proj_K,
            "distortion": init_proj_dist,
            "undistortion": init_proj_undist,
        },
        cam_name_1="camera",
        cam_name_2="projector",
        removal_type="overall",
        removal_algorithm="3sigma",
    )

    del calibration["date"]

    for cam_name in calibration:
        calibration[cam_name]["K"] = calibration[cam_name]["K"].tolist()
        calibration[cam_name]["distortion"] = calibration[cam_name][
            "distortion"
        ].tolist()
        calibration[cam_name]["undistortion"] = calibration[cam_name][
            "undistortion"
        ].tolist()
        calibration[cam_name]["Rt"] = calibration[cam_name]["Rt"].tolist()
        calibration[cam_name]["resolution"] = calibration[cam_name]["resolution"]

    import json

    output_file = "calibpy\\data\\results\\stereo_calib_result.json"
    with open(output_file, "w") as out:
        json.dump(calibration, out, sort_keys=True, indent=4)

    return cam_feature_data


if __name__ == "__main__":
    run_calibration("calibpy\\data\\test_images")
