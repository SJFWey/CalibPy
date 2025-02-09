import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import disk


def get_dists(points, origin, k=1):
    dists = np.linalg.norm(points - origin, axis=1, ord=2)
    inds = np.argsort(dists)
    dists = dists[inds]

    return dists[0:k], inds[0:k]


def polyfit(projector_map, features, radius=100, dense_thresh=0.5):
    """
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    """
    kx = 2
    ky = 2
    order = 2

    x_grid = np.arange(np.size(projector_map[:, :, 0], 0))
    y_grid = np.arange(np.size(projector_map[:, :, 0], 1))

    xx, yy = np.meshgrid(x_grid, y_grid)

    feature_mask = np.ones(np.size(features, 0), dtype=bool)

    for dir in range(2):
        for feature in range(np.size(features, 0)):
            xr = xx - features[feature, 1]
            yr = yy - features[feature, 0]

            radii = np.sqrt(np.power(xr, 2) + np.power(yr, 2))

            inds_r = np.where(radii <= radius)
            x0 = inds_r[1]
            y0 = inds_r[0]
            # z = phasemap[radii <= radius]
            z = projector_map[x0, y0, dir]

            inds_z = z > 0.0
            x = x0[inds_z]
            y = y0[inds_z]
            z = z[inds_z]

            if np.size(x) / np.size(x0) > dense_thresh:
                # coefficient array, up to x^kx, y^ky
                coeffs = np.ones((kx + 1, ky + 1))

                # solve array
                a = np.zeros((coeffs.size, x.size))

                # for each coefficient produce array x^i, y^j
                for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
                    # do not include powers greater than order
                    if order is not None and i + j > order:
                        arr = np.zeros_like(x)
                    else:
                        arr = coeffs[i, j] * x**i * y**j

                    a[index] = arr.ravel()

                fit_data = np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

                c = fit_data[0]

                Z = np.zeros((np.shape(x0)), dtype=float)

                for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
                    Z += c[index] * x0**i * y0**j

                projector_map[x0, y0, dir] = Z

                # print("feature " + str(feature) + " done")

            else:
                # print("Not enough points for feature " + str(feature))
                feature_mask[feature] = 0

        mask_x = projector_map[:, :, 0] > 1.0
        mask_y = projector_map[:, :, 1] > 1.0
        mask = mask_x * mask_y

        projector_map[:, :, 2] = mask

        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # X = np.arange(0, 1504)
        # Y = np.arange(0, 1504)
        # X, Y = np.meshgrid(X, Y)
        #
        # ax.plot_surface(X, Y, Z, cmap='turbo')
        #
        # plt.show()
        # time.sleep(10)

    return projector_map, feature_mask


def getObjectPointsNNSingle(img, grid_dist=0.5, folder_ind=0, step=0.25, vis=False):
    max_iterations = 400
    origin = [640, 360]

    # imagePoints, _, _, _ = extract_features(img)
    imagePoints, _ = adaptive_feature_extractor(img)
    _, origin_index = get_dists(imagePoints, origin, k=1)

    # imagePoints, _ = extract_features(img)
    # For Tracking:
    # _, origin_index = get_dists(imagePoints, origin_old, k=1)
    # No Tracking:
    # _, origin_index = get_dists(imagePoints, origin, k=1)

    dists, neighbours_index = get_dists(imagePoints, imagePoints[origin_index], k=5)

    objectPoints = np.empty(np.shape(imagePoints))
    objectPoints.fill(np.NAN)
    objectPoints[origin_index, :] = [0, 0]

    iterations = 0
    while np.isnan(objectPoints.sum()):
        if iterations > max_iterations:
            break
        iterations += 1

        for num, _ in enumerate(imagePoints):
            if not np.isnan(objectPoints[num, 0]):
                dists, neighbours_index = get_dists(
                    imagePoints, imagePoints[num, :], k=5
                )

                vecs = imagePoints[neighbours_index[1:5], :] - imagePoints[num, :]
                vecs = (
                    vecs
                    / np.asarray(
                        [np.linalg.norm(vecs, axis=1), np.linalg.norm(vecs, axis=1)]
                    ).T
                )
                vecs /= math.sqrt(2) * 0.5

                min_dist = np.min(dists[1:5])

                for dir in range(4):
                    if np.isnan(objectPoints[neighbours_index[dir + 1]].sum()):
                        if (
                            dists[dir + 1] / min_dist
                        ) < 1.3:  # abs((dists[dir + 1] - av_dist) / av_dist) < 0.5:
                            if (
                                np.round(vecs[dir, 1], 0) * np.round(vecs[dir, 0], 0)
                                == 0
                            ):
                                objectPoints[neighbours_index[dir + 1]] = (
                                    np.round(vecs[dir], 0) * grid_dist
                                    + objectPoints[num, :]
                                )

    imagePoints = imagePoints[~np.isnan(objectPoints).any(axis=1)]
    objectPoints = objectPoints[~np.isnan(objectPoints).any(axis=1)]

    # Set Z coordinate based on folder_ind
    z = np.ones((np.size(objectPoints, axis=0), 1)) * (step * folder_ind)
    objectPoints = np.hstack((objectPoints, z))

    if vis:
        plt.imshow(img)
        for num, coords in enumerate(imagePoints):
            plt.text(
                int(coords[0]),
                int(coords[1]),
                str(objectPoints[num, :]),
                c="black",
                fontsize=5.0,
            )
        # plt.text(0, 0, str(), c="red", fontsize=10.0)
        plt.scatter(imagePoints[:, 0], imagePoints[:, 1], c="red")
        plt.draw()
        plt.pause(1)
        plt.clf()

    return imagePoints, objectPoints


def adaptive_feature_extractor(
    img, min_area=250, max_area=3000, circularity_threshold=0.75
):
    """
    Extract feature points from a grayscale image.

    The function scales, blurs, applies CLAHE, and uses adaptive thresholding along with morphological
    operations to enhance features. It then finds contours, computes centroids, and filters them based
    on area and circularity.

    Parameters:
        img (numpy.ndarray): Grayscale input image with pixel values in [0, 1] or [0, 255].
        min_area (int): Minimum contour area (default 250).
        max_area (int): Maximum contour area (default 3000).
        circularity_threshold (float): Minimum circularity value (default 0.75).

    Returns:
        numpy.ndarray: Array of detected feature points, each as [cx, cy].
    """

    if np.max(img) <= 1.0:
        img *= 255
    img = img.astype(np.uint8)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)

    # _, thresh = cv2.threshold(
    #     img_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    # )

    # img_blurred = cv2.GaussianBlur(img_clahe, (9, 9), 0)
    img_blurred = cv2.medianBlur(img_clahe, 9)

    img_binary = cv2.adaptiveThreshold(
        img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 37, 7
    )

    ks = 9
    kernel = np.ones((ks, ks), dtype=np.uint8)
    rr, cc = disk((int(ks / 2), int(ks / 2)), int(ks / 2))
    kernel[rr, cc] = 1

    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    imgPoints = []

    for contour in contours:
        m = cv2.moments(contour)
        if m["m00"] == 0:  # avoid division by zero
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        circularity = 0
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter**2)

        if area > min_area and area < max_area and circularity > circularity_threshold:
            imgPoints.append([cx, cy])

    imgPoints = np.asarray(imgPoints)

    # _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(img_binary, cmap="gray")
    # axs[0].set_title("Threshold Image")

    # axs[1].imshow(img, cmap="gray")
    # axs[1].scatter(imgPoints[:, 0], imgPoints[:, 1], c="red", s=7)
    # axs[1].set_title("Detected Features")

    # plt.tight_layout()
    # plt.show()
    # plt.close()

    return np.asarray(imgPoints)


def export_features(camera_data, projector_data, planes, save_path):
    import json
    from pathlib import Path

    from scipy.io import savemat

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for num in range(len(camera_data)):
        if planes[num] and len(projector_data[num]["imagePoints"]) > 6:
            # np.save(save_path / f"cam_img_points_{num}.npy", camera_data[num]['imagePoints']) # (N, 2)
            # np.save(save_path / f"cam_obj_points_{num}.npy", camera_data[num]['objectPoints']) # (N, 3)
            # np.save(save_path / f"proj_img_points_{num}.npy", projector_data[num]['imagePoints']) # (N, 2)
            # np.save(save_path / f"proj_obj_points_{num}.npy", projector_data[num]['objectPoints']) # (N, 3)

            # print(f"num {num}:")
            # print(f"cam_img_points: {camera_data[num]['imagePoints'].shape}")
            # print(f"cam_obj_points: {camera_data[num]['objectPoints'].shape}")
            # print(f"proj_img_points: {projector_data[num]['imagePoints'].shape}")
            # print(f"proj_obj_points: {projector_data[num]['objectPoints'].shape}\n")

            # Check if camera and projector object points match for each point
            cam_obj = camera_data[num]["objectPoints"]
            proj_obj = projector_data[num]["objectPoints"]

            # Verify points match within small tolerance
            if not np.allclose(cam_obj, proj_obj, rtol=1e-5, atol=1e-8):
                print(f"Warning: Points mismatch in set {num}")
                continue

            savemat(
                save_path / f"calib_data_{num}.mat",
                {
                    "img_points": camera_data[num]["imagePoints"],
                    "obj_points": camera_data[num]["objectPoints"],
                    "proj_img_points": projector_data[num]["imagePoints"],
                    "proj_obj_points": projector_data[num]["objectPoints"],
                },
            )

            # Save calibration data in pretty JSON format
            json_data = {
                "img_points": camera_data[num]["imagePoints"].tolist(),
                "obj_points": camera_data[num]["objectPoints"].tolist(),
                "proj_img_points": projector_data[num]["imagePoints"].tolist(),
                "proj_obj_points": projector_data[num]["objectPoints"].tolist(),
            }

            with open(save_path / f"calib_data_{num}.json", "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=4, sort_keys=True)


def visualize_feature_tracking(
    ref_img,
    img_points,
    ref_point,
    prev_ref_point=None,
    prev_ref_image=None,
    folder_stem=None,
):
    """
    Visualize feature tracking results.

    Parameters:
    -----------
    ref_img : numpy.ndarray
        Current reference image
    img_points : numpy.ndarray
        Detected feature points
    ref_point : numpy.ndarray
        Current reference point
    prev_ref_point : numpy.ndarray, optional
        Previous reference point for tracking trajectory
    prev_ref_image : numpy.ndarray, optional
        Previous reference image for overlay
    folder_stem : str, optional
        Folder name for the title
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(ref_img, cmap="gray", alpha=1.0)

    if prev_ref_image is not None and prev_ref_point is not None:
        plt.imshow(prev_ref_image, cmap="gray", alpha=0.2)
        plt.plot(
            prev_ref_point[0],
            prev_ref_point[1],
            "o",
            color="red",
            markersize=12,
            markerfacecolor="none",
            label="last ref point",
            linewidth=3.0,
        )

    plt.plot(
        ref_point[0],
        ref_point[1],
        "o",
        color="deepskyblue",
        markersize=12,
        markerfacecolor="none",
        label="current ref point",
        linewidth=5.0,
    )

    plt.plot(
        img_points[:, 0],
        img_points[:, 1],
        "go",
        markersize=14,
        markerfacecolor="none",
        label="features",
        linewidth=3.0,
    )

    if prev_ref_point is not None:
        plt.arrow(
            prev_ref_point[0],
            prev_ref_point[1],
            ref_point[0] - prev_ref_point[0],
            ref_point[1] - prev_ref_point[1],
            color="red",
            width=0.5,
            head_width=3,
            label="tracking trajectory from last frame",
        )

    title = "Features Tracking"
    if folder_stem:
        title += f" - Folder_{folder_stem}"
    plt.title(title)
    plt.legend()
    plt.axis("image")
    plt.show()
