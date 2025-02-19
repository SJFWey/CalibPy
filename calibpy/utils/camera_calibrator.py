import math
from dataclasses import dataclass, field

import numpy as np
import cv2
from scipy.optimize import least_squares


@dataclass
class InitParams:
    """
    Holds initial intrinsics for the camera calibration.
    """

    IMG_WIDTH: int = 1280
    IMG_HEIGHT: int = 720
    INITIAL_FOV: int = 60  # Degree

    fx: float = 703.0
    fy: float = 697.0
    cx: float = None
    cy: float = None
    skew: float = 0.0
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=float))

    def __post_init__(self):
        if self.fx is None or self.fy is None:
            f = (self.IMG_WIDTH / 2.0) / math.tan(math.pi * self.INITIAL_FOV / 360.0)
            if self.fx is None:
                self.fx = f
            if self.fy is None:
                self.fy = f
        if self.cx is None:
            self.cx = self.IMG_WIDTH / 2.0
        if self.cy is None:
            self.cy = self.IMG_HEIGHT / 2.0


class Optimizer:
    """
    Performs optimization to minimize reprojection error.
    """

    def __init__(self, init_params: InitParams, calib_data):
        """
        Initialize the optimizer.

        Args:
            init_params: An InitParams object containing initial camera intrinsics.
            calib_data: A dictionary with calibration data (obj_points, img_points).
        """
        # Initialize parameters from InitParams
        self.IMG_WIDTH = init_params.IMG_WIDTH
        self.IMG_HEIGHT = init_params.IMG_HEIGHT
        self.fx = init_params.fx
        self.fy = init_params.fy
        self.cx = init_params.cx
        self.cy = init_params.cy
        self.skew = init_params.skew
        self.dist_coeffs = init_params.dist_coeffs.copy()

        # Initialize extrinsics and parameter optimization records
        self.rvecs = {}
        self.tvecs = {}
        self.params_history = None

        # Load data
        self.object_points = calib_data.get(
            "obj_points", {}
        )  # World coordinates for each image (n_img, (N, 3))
        self.image_points = calib_data.get(
            "img_points", {}
        )  # Pixel coordinates for each image (n_img, (N, 2))

        self.num_images = len(self.image_points)

        self.active_images = np.ones(self.num_images, dtype=bool)
        self.active_indices = []
        self.check_active_images()  # Update active images (placeholder method)
        self.num_active_images = len(self.active_indices)
        self.deactivated_images = []  # Deactivated images

        # When calibrating using a single image with a planar calibration board, disable principal point and skew estimation (with warning)
        self.estimate_skew = False

        self.estimate_principal = True
        self.init_pp = np.array([self.cx, self.cy])
        self.fix_aspect_ratio = False

        self.estimate_focal = np.array([True, True])
        if self.fix_aspect_ratio:
            self.init_fc = np.array(
                [(self.fx + self.fy) // 2, (self.fx + self.fy) // 2]
            )
        else:
            self.init_fc = np.array([self.fx, self.fy])

        self.estimate_distort = np.array([True, True, True, True, True])

        self.no_initial_guess = False  # Flags for 3D calibration

        self.max_iter = 30  # Maximum iterations for gradient descent
        self.alpha_smooth = (
            0.1  # Smoothing factor, [0, 1] convergence from slow to fast
        )

        # Check if the object points are 3D over all active images
        self.is_3d_calib = False
        for i in self.active_indices:
            pts = self.object_points.get(i, None)
            if pts is not None and pts.shape[1] == 3:
                self.is_3d_calib = True

        # If initial guess are not provided, set the flag
        if (
            not np.any(self.init_fc)
            or not np.any(self.init_pp)
            or not np.any(self.dist_coeffs)
        ):
            self.no_initial_guess = True

        if self.no_initial_guess:
            # Estimate initial intrinsics
            self.calc_init_intrinsics()
            # Estimate initial extrinsics
            self.calc_init_extrinsics()

            # Print initial estimates
            print(
                f"Initial estimates: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, skew={self.skew}, dist_coeffs={self.dist_coeffs}"
            )

        if (
            self.estimate_principal
            and (len(self.active_indices) < 2)
            and (not self.is_3d_calib)
        ):
            # Principal point shouldn't be optimized when using one image and planar rig."
            self.estimate_principal = False
            self.estimate_skew = False

        if (
            self.estimate_principal
            and (len(self.active_indices) < 5)
            and (not self.is_3d_calib)
        ):
            print(
                "WARNING: The principal point estimation may be unreliable (More images needed for calibration)."
            )

        # if not np.array_equal(self.estimate_focal, np.array([True, True])):
        #     self.fix_aspect_ratio = True
        # if not self.fix_aspect_ratio:
        #     self.estimate_focal = np.array([True, True])

        # Principal point estimation
        if (not self.estimate_principal) and self.estimate_skew:
            # optimize_principal=False, skew estimation should be set to False.
            self.estimate_skew = False

        # When not estimating the focal length, keep the principal point at the image center
        if np.array_equal(self.estimate_focal, np.array([False, False])):
            self.estimate_principal = False

        # If the distortion is not fully estimated, set the unestimated parts to zero
        if not np.all(self.estimate_distort):
            self.dist_coeffs = self.dist_coeffs * self.estimate_distort

        # Add flags for extrinsics, e.g. optimize all or none:
        self.estimate_extrinsics = True

        # --- New: Setup organized optimization flags ---
        self.setup_optim_flags()
        # --- End New ---

        self.params = self.init_parameter_vector()

        # Save parameter iteration history
        self.params_history = self.params.copy()

    # --- New method for setting optimization flags ---
    def setup_optim_flags(self):
        """
        Organize and store optimization flags into a dictionary.
        """
        self.optim_flags = {
            "focal": self.estimate_focal,  # [estimate_fx, estimate_fy]
            "principal": self.estimate_principal,  # bool
            "skew": self.estimate_skew,  # bool
            "distortion": self.estimate_distort,  # np.array of bool for each dist coeff
            "extrinsics": self.estimate_extrinsics,  # bool
        }

    # --- End New ---

    def init_parameter_vector(self):
        """
        Build the initial parameter vector using the index map.
        """
        # Index-map logic merged here
        self.param_flags = []
        # Focal
        if self.optim_flags["focal"][0]:
            self.param_flags.append("fx")
        if self.optim_flags["focal"][1]:
            self.param_flags.append("fy")
        # Principal
        if self.optim_flags["principal"]:
            self.param_flags.extend(["cx", "cy"])
        # Skew
        if self.optim_flags["skew"]:
            self.param_flags.append("skew")
        # Distortion
        for i, flag in enumerate(self.optim_flags["distortion"]):
            if flag:
                self.param_flags.append(f"dist_{i}")
        # Extrinsics
        if self.optim_flags["extrinsics"]:
            self.param_flags.extend(["extrinsics"] * (6 * self.num_active_images))

        param_list = []
        # Populate param_list in order of self.param_flags
        # Focal
        if "fx" in self.param_flags:
            param_list.append(self.init_fc[0])
        if "fy" in self.param_flags:
            param_list.append(self.init_fc[1])
        # Principal
        if "cx" in self.param_flags:
            param_list.append(self.init_pp[0])
        if "cy" in self.param_flags:
            param_list.append(self.init_pp[1])
        # Skew
        if "skew" in self.param_flags:
            param_list.append(self.skew)
        # Distortion
        for i, c in enumerate(self.dist_coeffs):
            if f"dist_{i}" in self.param_flags:
                param_list.append(c)
        # Extrinsics
        if self.estimate_extrinsics:
            for i, idx in enumerate(self.active_indices):
                rvec = self.rvecs.get(idx, np.zeros(3)).flatten()
                tvec = self.tvecs.get(idx, np.zeros(3)).flatten()
                param_list.extend(rvec)
                param_list.extend(tvec)
        return np.array(param_list)

    def calc_init_intrinsics(self):
        """
        Initialize intrinsic parameters using planar homography.
        For each valid image, we use the orthogonality constraint between the two
        principal directions. After removing the principal point offset, if H = [h1, h2, t]
        is the homography, then the constraint is:
        (h1_x * h2_x) / f_x^2 + (h1_y * h2_y) / f_y^2 = -h1_z * h2_z
        We accumulate these constraints from all valid images and solve for f_x and f_y.
        """
        n_active = len(self.active_indices)
        print(f"\nInitializing intrinsic parameters - Number of images: {n_active}\n")

        # Collect valid homographies in a list.
        valid_homographies = []
        for idx in self.active_indices:
            if self.active_images[idx]:
                image_pts = self.image_points.get(idx, None)
                object_pts = self.object_points.get(idx, None)
                if image_pts is None or np.isnan(image_pts).any():
                    print(
                        f"WARNING: Calibration with image {idx} failed. Check the image points."
                    )
                    self.active_images[idx] = False
                    continue
                if image_pts is not None and object_pts is not None:
                    H, _ = cv2.findHomography(object_pts[:, :2], image_pts, cv2.RANSAC)
                    if H is not None and not np.isnan(H).any():
                        valid_homographies.append(H)
                    else:
                        self.active_images[idx] = False

        self.check_active_images()

        # Create principal point offset matrix to subtract (cx, cy)
        pp_offset = np.array([[1, 0, -self.cx], [0, 1, -self.cy], [0, 0, 1]])

        A_list = []
        b_list = []
        for H in valid_homographies:
            # Remove principal point offset.
            H_offset = pp_offset @ H

            # Normalize H_offset so that H_offset[2,2] == 1 (if possible)
            if np.abs(H_offset[2, 2]) > 1e-8:
                H_offset = H_offset / H_offset[2, 2]

            # Extract the first two columns of H as h1 and h2.
            h1 = H_offset[:, 0]
            h2 = H_offset[:, 1]

            # Construct the orthogonality constraint:
            # (h1[0]*h2[0])/f_x^2 + (h1[1]*h2[1])/f_y^2 = -h1[2]*h2[2]
            A_list.append([h1[0] * h2[0], h1[1] * h2[1]])
            b_list.append(-h1[2] * h2[2])

        if len(A_list) == 0:
            print("No valid images available for intrinsic initialization.")
            return

        A = np.array(A_list)
        b = np.array(b_list)

        # Solve for [1/f_x^2, 1/f_y^2] using least squares: A * [1/f_x^2, 1/f_y^2]^T = b
        X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Ensure that the computed values are positive to obtain real focal lengths.
        inv_fx2 = np.abs(X[0])
        inv_fy2 = np.abs(X[1])
        f_x_init = 1.0 / np.sqrt(inv_fx2) if inv_fx2 > 1e-8 else 0.0
        f_y_init = 1.0 / np.sqrt(inv_fy2) if inv_fy2 > 1e-8 else 0.0

        if self.fix_aspect_ratio:
            # Use the average focal length if aspect ratio is fixed.
            f_scalar = (f_x_init + f_y_init) / 2.0
            f_x_init = f_y_init = f_scalar

        self.fx = f_x_init
        self.fy = f_y_init
        self.init_fc = np.array([self.fx, self.fy])

    def calc_init_extrinsics(self):
        """
        Initialize extrinsic parameters for each frame.
        """
        self.rvecs = {}
        self.tvecs = {}
        self.rot_matrices = {}

        # Define the intrinsic camera matrix once for consistency
        K_init = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        for idx in self.active_indices:
            if self.active_images[idx]:
                image_pts = self.image_points.get(idx)
                object_pts = self.object_points.get(idx)
                # Input validation is ignored as per instructions
                # Inline normalization logic
                image_pts = image_pts.reshape(-1, 1, 2).astype(np.float64)
                undistorted_pts = cv2.undistortPoints(
                    image_pts, K_init, np.array(self.dist_coeffs).flatten()
                )
                image_pts_norm = undistorted_pts.reshape(-1, 2)

                # Estimation logic (planar or PnP)
                num_pts = image_pts_norm.shape[0]
                obj_pts_mean = np.mean(object_pts, axis=0, keepdims=True)
                Y = object_pts - obj_pts_mean
                _, S, _ = np.linalg.svd(Y.T @ Y)
                planar_ratio = S[2] / S[1] if S[1] != 0 else 0

                if planar_ratio < 1e-6 or num_pts < 5:
                    H, _ = cv2.findHomography(
                        image_pts_norm, object_pts[:, :2], cv2.RANSAC
                    )
                    h1 = H[:, 0]
                    h2 = H[:, 1]
                    lambda_ = 2.0 / (np.linalg.norm(h1) + np.linalg.norm(h2))
                    r1 = h1 * lambda_
                    r2 = h2 * lambda_
                    r3 = np.cross(r1, r2)
                    rot_mat = np.column_stack([r1, r2, r3])
                    U, _, Vt = np.linalg.svd(rot_mat)
                    rot_mat = U @ Vt
                    if np.linalg.det(rot_mat) < 0:
                        rot_mat = -rot_mat
                    rvec, _ = cv2.Rodrigues(rot_mat)
                    tvec = H[:, 2] * lambda_
                else:
                    success, rvec, tvec = cv2.solvePnP(
                        object_pts,
                        image_pts,
                        K_init,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    if not success:
                        raise RuntimeError("PnP estimation failed")

                rvec, tvec = cv2.solvePnPRefineLM(
                    object_pts,
                    image_pts,
                    K_init,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
                        100,
                        1e-6,
                    ),
                )

                self.rvecs[idx] = rvec
                self.tvecs[idx] = tvec
                if not np.any(np.isnan(rvec)):
                    self.rot_matrices[idx] = cv2.Rodrigues(rvec)[0]
        self.check_active_images()

    def check_active_images(self):
        """
        Validate active images and refresh the active index list.

        Returns:
            bool: True if at least one image remains active.
        """
        # Validate n_ima
        if not isinstance(self.num_images, int) or self.num_images < 0:
            raise ValueError(
                "Number of images (num_images) must be a non-negative integer"
            )

        if self.num_images == 0:
            return False

        # Resize active_images array if needed
        n_active = len(self.active_images)
        if n_active < self.num_images:
            self.active_images = np.concatenate(
                [self.active_images, np.ones(self.num_images - n_active, dtype=bool)]
            )
        elif n_active > self.num_images:
            self.active_images = self.active_images[: self.num_images]

        # Check if any active images remain
        if not np.any(self.active_images):
            print(
                "Error: There is no active image. Run Add/Suppress images to add images"
            )
            return False

        self.active_indices = list(np.where(self.active_images)[0])
        return True

    def calibrate(self):
        """
        Conduct camera calibration by optimizing reprojection error.

        Returns:
            dict: Calibration results with intrinsics, extrinsics, and residual.
        """
        self.check_active_images()  # Update active images

        # Initial guess for intrinsic and extrinsic parameters
        params_opt = self.params.copy()

        res = least_squares(self.calc_reproj_error, params_opt, verbose=1)
        params_opt = res.x

        # --- New: Update internal parameters from the optimized vector ---
        self.update_params_from_vector(params_opt)
        # --- End New ---

        return {
            "fc": [self.fx, self.fy],
            "pp": [self.cx, self.cy],
            "skew": self.skew,
            "dist_coeffs": self.dist_coeffs,
            "extrinsics": [
                {"rvec": self.rvecs[i], "tvec": self.tvecs[i]}
                for i in self.active_indices
            ],
            "residual": res.cost,
        }

    # --- New method: update parameters from optimization vector ---
    def update_params_from_vector(self, params):
        """
        Update the internal parameters from the flat optimization vector.
        """
        idx = 0
        if "fx" in self.param_flags:
            self.fx = params[idx]
            idx += 1
        if "fy" in self.param_flags:
            self.fy = params[idx]
            idx += 1
        if "cx" in self.param_flags:
            self.cx = params[idx]
            idx += 1
        if "cy" in self.param_flags:
            self.cy = params[idx]
            idx += 1
        if "skew" in self.param_flags:
            self.skew = params[idx]
            idx += 1
        for i in range(5):
            if f"dist_{i}" in self.param_flags:
                self.dist_coeffs[i] = params[idx]
                idx += 1
        if self.estimate_extrinsics:
            for i in self.active_indices:
                rvec = params[idx : idx + 3]
                tvec = params[idx + 3 : idx + 6]
                self.rvecs[i] = np.array(rvec)
                self.tvecs[i] = np.array(tvec)
                idx += 6

    def calc_reproj_error(self, params):
        """
        Compute the reprojection error for a given parameter set.

        Args:
            params: Flattened array of intrinsic and extrinsic parameters.

        Returns:
            np.ndarray: Residual errors, flattened.
        """
        import cv2
        import numpy as np

        # Reconstruct each parameter in the same order used above
        idx = 0
        fc = [self.fx, self.fy]
        pp = [self.cx, self.cy]
        skew_ = self.skew
        dist_ = self.dist_coeffs.copy()

        if "fx" in self.param_flags:
            fc[0] = params[idx]
            idx += 1
        if "fy" in self.param_flags:
            fc[1] = params[idx]
            idx += 1
        if "cx" in self.param_flags:
            pp[0] = params[idx]
            idx += 1
        if "cy" in self.param_flags:
            pp[1] = params[idx]
            idx += 1
        if "skew" in self.param_flags:
            skew_ = params[idx]
            idx += 1
        for i in range(5):
            if f"dist_{i}" in self.param_flags:
                dist_[i] = params[idx]
                idx += 1

        # Now handle extrinsics if self.estimate_extrinsics:
        extrinsics = []
        if self.estimate_extrinsics:
            n_extr = 6 * self.num_active_images
            extrinsics_vals = params[idx : idx + n_extr]
            idx_e = 0
            for _ in range(self.num_active_images):
                rvec = extrinsics_vals[idx_e : idx_e + 3]
                tvec = extrinsics_vals[idx_e + 3 : idx_e + 6]
                extrinsics.append((rvec, tvec))
                idx_e += 6

        residuals = []
        # Iterate over active indices to correctly index calibration data
        for i, idx in enumerate(self.active_indices):
            off = i * 6
            rvec = extrinsics[i][0] if self.estimate_extrinsics else self.rvecs[idx]
            tvec = extrinsics[i][1] if self.estimate_extrinsics else self.tvecs[idx]

            obj_pts = self.object_points[idx]  # [N, 3]
            img_pts = self.image_points[idx]  # [N, 2]

            K = np.array([[fc[0], skew_, pp[0]], [0, fc[1], pp[1]], [0, 0, 1]])

            projected_points, _ = cv2.projectPoints(
                obj_pts,
                rvec,
                tvec,
                K,
                dist_,
            )
            residuals.append(projected_points.reshape(-1, 2) - img_pts)

        return np.concatenate(residuals).ravel()
