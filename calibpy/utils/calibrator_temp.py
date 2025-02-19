"""
Enhanced calibrator_temp.py

This code builds on the original structure and includes the following improvements:
1. Enhanced algorithm refinement: changed the least squares optimization method to "trf" with a robust "soft_l1" loss function and tightened termination criteria (ftol, xtol, gtol).
2. Analytical Jacobian computation: added project_point_and_jacobian() to compute the analytical Jacobian of the projection function with respect to intrinsic (including distortion) and extrinsic parameters (rotation and translation vectors) via the chain rule, replacing numerical differentiation.
3. Uncertainty estimation and error analysis: after calibration, compute the parameter covariance matrix from the global Jacobian and estimate the standard deviation for each parameter. Also, calculate the mean and standard deviation of the reprojection error per image to assess calibration quality.
4. Robustness improvements: use RANSAC for initial intrinsic and extrinsic estimation, and include a condition number check in the analytical Jacobian computation to warn of numerical instability.
5. Code clarity and documentation: main functions are documented in detail explaining the rationale and implementation.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import cv2
from scipy.optimize import least_squares


@dataclass
class InitParams:
    """
    Initialize camera intrinsic parameters.
    """

    IMG_WIDTH: int = 1280
    IMG_HEIGHT: int = 720
    INITIAL_FOV: int = 60  # degrees

    fx: float = None
    fy: float = None
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


@dataclass
class Intrinsics:
    """
    Camera intrinsic parameter data structure.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    skew: float
    dist_coeffs: np.ndarray


@dataclass
class Extrinsics:
    """
    Camera extrinsic parameter data structure.
    """

    rvec: np.ndarray
    tvec: np.ndarray


class CalibrationFlags:
    def __init__(self):
        self.estimate_focal = np.array([True, True])
        self.estimate_principal = True
        self.estimate_skew = True
        self.estimate_distort = np.array([True, True, True, True, True])
        self.estimate_extrinsics = True
        self.fix_aspect_ratio = False

        if np.array_equal(self.estimate_focal, np.array([False, False])):
            self.estimate_principal = False

    def to_dict(self):
        return {
            "focal": self.estimate_focal,
            "principal": self.estimate_principal,
            "skew": self.estimate_skew,
            "distortion": self.estimate_distort,
            "extrinsics": self.estimate_extrinsics,
        }


class Optimizer:
    """
    Optimizer class for minimizing the reprojection error.
    Improvements include:
      - Using the trust-region method ('trf') with robust loss ('soft_l1')
      - Analytical Jacobian computation for improved accuracy and efficiency
      - Parameter uncertainty estimation and condition number check of the Jacobian matrix
    """

    def __init__(self, init_params: InitParams, calib_data):
        self.IMG_WIDTH = init_params.IMG_WIDTH
        self.IMG_HEIGHT = init_params.IMG_HEIGHT
        self.fx = init_params.fx
        self.fy = init_params.fy
        self.cx = init_params.cx
        self.cy = init_params.cy
        self.skew = init_params.skew
        self.dist_coeffs = init_params.dist_coeffs

        self.object_points = calib_data.get("obj_points", {})
        self.image_points = calib_data.get("img_points", {})

        self.num_images = len(self.image_points)
        self.active_images = np.ones(self.num_images, dtype=bool)
        self.active_indices = []
        self.check_active_images()
        self.num_active_images = len(self.active_indices)

        self.flags = CalibrationFlags()
        self.optim_flags = self.flags.to_dict()

        self.init_pp = np.array([self.cx, self.cy])
        if self.flags.fix_aspect_ratio:
            self.init_fc = np.array(
                [(self.fx + self.fy) // 2, (self.fx + self.fy) // 2]
            )
        else:
            self.init_fc = np.array([self.fx, self.fy])
        self.initial_guess = True
        self.max_iter = 100
        self.alpha_smooth = 0.1

        if self.initial_guess:
            if self.fx is None or self.fy is None:
                fx_est, fy_est = estimate_intrinsics(
                    self.object_points,
                    self.image_points,
                    self.active_indices,
                    self.cx,
                    self.cy,
                    self.flags.fix_aspect_ratio,
                )
                if fx_est is not None and fy_est is not None:
                    self.fx = fx_est
                    self.fy = fy_est
                    self.init_fc = np.array([self.fx, self.fy])
            else:
                self.init_fc = np.array([self.fx, self.fy])

        self.dist_coeffs = self.dist_coeffs * self.flags.estimate_distort

        self.rvecs = {}
        self.tvecs = {}

        extrinsics_est = estimate_extrinsics(
            self.object_points,
            self.image_points,
            self.active_indices,
            Intrinsics(self.fx, self.fy, self.cx, self.cy, self.skew, self.dist_coeffs),
            self.dist_coeffs,
        )

        for idx in self.active_indices:
            if idx in extrinsics_est:
                self.rvecs[idx] = extrinsics_est[idx].rvec
                self.tvecs[idx] = extrinsics_est[idx].tvec
            else:
                self.rvecs[idx] = np.zeros(3)
                self.tvecs[idx] = np.zeros(3)
        print(
            f"Initial estimates: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, skew={self.skew}, dist_coeffs={self.dist_coeffs}"
        )

        self.intrinsics = Intrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            skew=self.skew,
            dist_coeffs=self.dist_coeffs.copy(),
        )
        self.extrinsics = {}
        for idx in self.active_indices:
            rvec = self.rvecs.get(idx, np.zeros(3))
            tvec = self.tvecs.get(idx, np.zeros(3))
            self.extrinsics[idx] = Extrinsics(
                rvec=np.array(rvec).flatten(), tvec=np.array(tvec).flatten()
            )

        self.params_history = self.pack_parameters()

        # Set optimization method and loss parameters for robustness
        self.optimization_method = (
            "trf"  # Trust-region reflective algorithm supporting robust loss
        )
        self.loss = "soft_l1"  # Robust loss to reduce influence of outliers
        self.ftol = 1e-12
        self.xtol = 1e-12
        self.gtol = 1e-12

    def pack_parameters(self):
        """
        Pack structured parameters into a 1D vector.
        """
        params = []
        if self.optim_flags["focal"][0]:
            params.append(float(self.intrinsics.fx))
        if self.optim_flags["focal"][1]:
            params.append(float(self.intrinsics.fy))
        if self.optim_flags["principal"]:
            params.append(float(self.intrinsics.cx))
            params.append(float(self.intrinsics.cy))
        if self.optim_flags["skew"]:
            params.append(float(self.intrinsics.skew))
        for i, flag in enumerate(self.optim_flags["distortion"]):
            if flag:
                params.append(float(self.intrinsics.dist_coeffs[i]))
        if self.optim_flags["extrinsics"]:
            for idx in self.active_indices:
                extr = self.extrinsics[idx]
                params.extend([float(x) for x in extr.rvec])
                params.extend([float(x) for x in extr.tvec])
        return np.array(params, dtype=np.float64)

    def unpack_parameters(self, params):
        """
        Unpack a 1D parameter vector into structured parameters.
        """
        params = np.asarray(params)
        idx = 0
        if self.optim_flags["focal"][0]:
            self.intrinsics.fx = float(params[idx])
            idx += 1
        if self.optim_flags["focal"][1]:
            self.intrinsics.fy = float(params[idx])
            idx += 1
        if self.optim_flags["principal"]:
            self.intrinsics.cx = float(params[idx])
            idx += 1
            self.intrinsics.cy = float(params[idx])
            idx += 1
        if self.optim_flags["skew"]:
            self.intrinsics.skew = float(params[idx])
            idx += 1
        for i, flag in enumerate(self.optim_flags["distortion"]):
            if flag:
                self.intrinsics.dist_coeffs[i] = float(params[idx])
                idx += 1
        if self.optim_flags["extrinsics"]:
            for key in self.active_indices:
                self.extrinsics[key].rvec = params[idx : idx + 3].astype(np.float64)
                idx += 3
                self.extrinsics[key].tvec = params[idx : idx + 3].astype(np.float64)
                idx += 3

    def compute_image_reprojection_error(self, idx):
        """
        Compute the reprojection error for a single image.
        """
        K = np.array(
            [
                [self.intrinsics.fx, self.intrinsics.skew, self.intrinsics.cx],
                [0, self.intrinsics.fy, self.intrinsics.cy],
                [0, 0, 1],
            ]
        )
        extr = self.extrinsics[idx]
        projected_points, _ = cv2.projectPoints(
            self.object_points[idx],
            extr.rvec,
            extr.tvec,
            K,
            self.intrinsics.dist_coeffs,
        )
        error = projected_points.reshape(-1, 2) - self.image_points[idx]
        return error.ravel()

    def project_point_and_jacobian(self, X, intrinsics, extrinsics):
        """
        Compute projection and analytical Jacobian for a single 3D point.
        """
        # Ensure proper shape for Rodrigues transform
        rvec = extrinsics.rvec.reshape(3, 1)
        R, dRodrigues = cv2.Rodrigues(rvec)

        # Transform point to camera coordinates
        X = X.reshape(3, 1)
        X_cam = R @ X + extrinsics.tvec.reshape(3, 1)
        Xc, Yc, Zc = X_cam.flatten()

        # Avoid division by zero
        Zc = np.maximum(Zc, 1e-8)
        x = Xc / Zc
        y = Yc / Zc
        r2 = x * x + y * y

        # Extract parameters
        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.cx
        cy = intrinsics.cy
        skew = intrinsics.skew
        k1, k2, p1, p2, k3 = intrinsics.dist_coeffs

        # Compute distortion
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        dx_tan = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy_tan = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

        x_d = x * radial + dx_tan
        y_d = y * radial + dy_tan

        # Project to image coordinates
        u = fx * x_d + skew * y_d + cx
        v = fy * y_d + cy
        proj = np.array([u, v])

        # Compute Jacobian for intrinsics
        jac_intr = np.zeros((2, 10))

        # Derivatives w.r.t. focal length and principal point
        jac_intr[0, 0] = x_d  # du/dfx
        jac_intr[1, 1] = y_d  # dv/dfy
        jac_intr[0, 2] = 1  # du/dcx
        jac_intr[1, 3] = 1  # dv/dcy
        jac_intr[0, 4] = y_d  # du/dskew

        # Derivatives w.r.t. distortion coefficients
        jac_intr[0, 5] = fx * x * r2  # du/dk1
        jac_intr[1, 5] = fy * y * r2  # dv/dk1
        jac_intr[0, 6] = fx * x * r2 * r2  # du/dk2
        jac_intr[1, 6] = fy * y * r2 * r2  # dv/dk2
        jac_intr[0, 7] = fx * (2 * x * y)  # du/dp1
        jac_intr[1, 7] = fy * (r2 + 2 * y * y)  # dv/dp1
        jac_intr[0, 8] = fx * (r2 + 2 * x * x)  # du/dp2
        jac_intr[1, 8] = fy * (2 * x * y)  # dv/dp2
        jac_intr[0, 9] = fx * x * r2 * r2 * r2  # du/dk3
        jac_intr[1, 9] = fy * y * r2 * r2 * r2  # dv/dk3

        # Compute Jacobian for extrinsics
        jac_ext = np.zeros((2, 6))

        # Derivative w.r.t. translation
        dproj_dX = np.array(
            [[fx / Zc, 0, -fx * Xc / (Zc * Zc)], [0, fy / Zc, -fy * Yc / (Zc * Zc)]]
        )
        jac_ext[:, 3:] = dproj_dX

        # Derivative w.r.t. rotation using Rodrigues formula Jacobian
        # dRodrigues is 3x9 matrix, each 3x3 block corresponds to derivatives
        # w.r.t. one component of the rotation vector
        dRX = np.zeros((3, 3))
        X_flat = X.flatten()

        for i in range(3):
            # Extract the 3x3 Jacobian block for the i-th rotation component
            dR = dRodrigues[:, 3 * i : 3 * (i + 1)]
            dRX[i] = (dR @ X_flat).flatten()

        jac_ext[:, :3] = dproj_dX @ dRX

        return proj, jac_intr, jac_ext

    def calc_reproj_jac_analytical(self, flat_params):
        """
        Calculate the analytical global Jacobian for all active images.
        Assemble the Jacobian for intrinsic and each image's extrinsics into a global matrix.
        """
        self.unpack_parameters(flat_params)
        # Construct global index mapping for intrinsics, order: [fx, fy, cx, cy, skew, k1, k2, p1, p2, k3]
        intrinsics_indices = []
        idx_counter = 0
        if self.optim_flags["focal"][0]:
            intrinsics_indices.append(0)  # fx
            idx_counter += 1
        if self.optim_flags["focal"][1]:
            intrinsics_indices.append(1)  # fy
            idx_counter += 1
        if self.optim_flags["principal"]:
            intrinsics_indices.extend([2, 3])  # cx, cy
            idx_counter += 2
        if self.optim_flags["skew"]:
            intrinsics_indices.append(4)  # skew
            idx_counter += 1
        distortion_indices = []
        for i, flag in enumerate(self.optim_flags["distortion"]):
            if flag:
                distortion_indices.append(5 + i)
                idx_counter += 1
        intrinsics_global_indices = []
        mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for j in intrinsics_indices + distortion_indices:
            intrinsics_global_indices.append(j)
        num_intrinsics = len(intrinsics_global_indices)
        num_extrinsics = 6 * self.num_active_images
        total_params = num_intrinsics + num_extrinsics

        # Calculate total residual dimension (2D for each point in all active images)
        total_points = 0
        for idx in self.active_indices:
            pts = self.object_points[idx]
            total_points += pts.shape[0]
        residual_dim = total_points * 2

        J_global = np.zeros((residual_dim, total_params), dtype=np.float64)

        row_counter = 0
        # Extrinsics start index in global parameter vector is num_intrinsics, each image has 6 parameters
        extrinsics_offset = num_intrinsics
        for img_counter, idx in enumerate(self.active_indices):
            extrinsics_params_idx = extrinsics_offset + img_counter * 6
            obj_pts = self.object_points[idx]
            img_pts = self.image_points[idx]
            for j, X in enumerate(obj_pts):
                proj, jac_intr_full, jac_ext = self.project_point_and_jacobian(
                    X, self.intrinsics, self.extrinsics[idx]
                )
                # Extract current optimized intrinsics part from jac_intr_full
                jac_intr_opt = jac_intr_full[:, intrinsics_global_indices]
                J_global[row_counter : row_counter + 2, 0:num_intrinsics] = jac_intr_opt
                J_global[
                    row_counter : row_counter + 2,
                    extrinsics_params_idx : extrinsics_params_idx + 6,
                ] = jac_ext
                row_counter += 2

        # Check Jacobian condition number, warn if too high indicating numerical instability
        cond_number = np.linalg.cond(J_global)
        if cond_number > 1e12:
            print(
                f"Warning: Jacobian condition number is high ({cond_number:.2e}), calibration may be unstable."
            )
        return J_global

    def estimate_uncertainty(self, J, residuals):
        """
        Estimate calibration parameter uncertainty:
          - Compute covariance matrix: cov = (J^T J)^-1 * σ², where σ² is the residual variance.
          - Compute standard deviation for each parameter.
        """
        dof = J.shape[0] - J.shape[1]
        if dof <= 0:
            print(
                "Warning: Insufficient degrees of freedom for uncertainty estimation."
            )
            return None
        sigma2 = np.sum(residuals**2) / dof
        try:
            JTJ_inv = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            print("Warning: Inversion of J^T J failed, uncertainty estimation aborted.")
            return None
        cov_matrix = JTJ_inv * sigma2
        param_std = np.sqrt(np.diag(cov_matrix))
        return cov_matrix, param_std

    def calibrate(self):
        """
        Perform camera calibration by minimizing the reprojection error.
        Returns calibration results including intrinsics, extrinsics, residual, and uncertainty estimation.
        """
        self.check_active_images()
        initial_params = self.pack_parameters().copy()
        res = least_squares(
            self.calc_reproj_error,
            initial_params,
            jac=self.calc_reproj_jac_analytical,
            method=self.optimization_method,
            loss=self.loss,
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            max_nfev=self.max_iter,
            verbose=1,
        )
        optimized_params = res.x
        self.unpack_parameters(optimized_params)

        # Recompute global Jacobian using analytical Jacobian for uncertainty estimation
        J_global = self.calc_reproj_jac_analytical(optimized_params)
        residuals = self.calc_reproj_error(optimized_params)
        uncertainty = self.estimate_uncertainty(J_global, residuals)
        if uncertainty is not None:
            cov_matrix, param_std = uncertainty
        else:
            cov_matrix, param_std = None, None

        # Compute reprojection error statistics for each image
        reproj_errors = []
        for idx in self.active_indices:
            err = self.compute_image_reprojection_error(idx)
            err_norm = np.linalg.norm(err) / (len(err) / 2)
            reproj_errors.append(err_norm)
        mean_error = np.mean(reproj_errors)
        std_error = np.std(reproj_errors)

        return {
            "fc": [self.intrinsics.fx, self.intrinsics.fy],
            "pp": [self.intrinsics.cx, self.intrinsics.cy],
            "skew": self.intrinsics.skew,
            "dist_coeffs": self.intrinsics.dist_coeffs,
            "extrinsics": [
                {"rvec": self.extrinsics[i].rvec, "tvec": self.extrinsics[i].tvec}
                for i in self.active_indices
            ],
            "residual": res.cost,
            "covariance": cov_matrix,
            "param_std": param_std,
            "mean_reproj_error": mean_error,
            "std_reproj_error": std_error,
        }

    def calc_reproj_error(self, flat_params):
        """
        Compute the concatenated reprojection error for all active images as a 1D vector.
        """
        self.unpack_parameters(flat_params)
        errors = []
        for idx in self.active_indices:
            err = self.compute_image_reprojection_error(idx)
            errors.append(err)
        return np.concatenate(errors)

    def check_active_images(self):
        """
        Check for active images and update the active indices list.
        """
        if not isinstance(self.num_images, int) or self.num_images < 0:
            raise ValueError("Number of images must be a non-negative integer")
        if self.num_images == 0:
            return False
        n_active = len(self.active_images)
        if n_active < self.num_images:
            self.active_images = np.concatenate(
                [self.active_images, np.ones(self.num_images - n_active, dtype=bool)]
            )
        elif n_active > self.num_images:
            self.active_images = self.active_images[: self.num_images]
        if not np.any(self.active_images):
            print("Error: No active images. Please add images.")
            return False
        self.active_indices = list(np.where(self.active_images)[0])
        return True


def estimate_intrinsics(
    object_points, image_points, active_indices, cx, cy, fix_aspect_ratio
):
    """
    Estimate camera intrinsics assuming a planar calibration target.
    Uses RANSAC to compute homographies and simple least squares to solve for intrinsic parameters.
    """
    valid_homographies = []
    pp_offset = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    for idx in active_indices:
        img_pts = image_points.get(idx, None)
        obj_pts = object_points.get(idx, None)
        if img_pts is None or obj_pts is None or np.isnan(img_pts).any():
            continue
        H, _ = cv2.findHomography(obj_pts[:, :2], img_pts, cv2.RANSAC)
        if H is not None and not np.isnan(H).any():
            valid_homographies.append(H)
    if len(valid_homographies) == 0:
        return None, None
    A_list = []
    b_list = []
    for H in valid_homographies:
        H_offset = pp_offset @ H
        if np.abs(H_offset[2, 2]) > 1e-8:
            H_offset = H_offset / H_offset[2, 2]
        h1 = H_offset[:, 0]
        h2 = H_offset[:, 1]
        A_list.append([h1[0] * h2[0], h1[1] * h2[1]])
        b_list.append(-h1[2] * h2[2])
    A = np.array(A_list)
    b = np.array(b_list)
    X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    inv_fx2 = np.abs(X[0])
    inv_fy2 = np.abs(X[1])
    fx = 1.0 / np.sqrt(inv_fx2) if inv_fx2 > 1e-8 else 0.0
    fy = 1.0 / np.sqrt(inv_fy2) if inv_fy2 > 1e-8 else 0.0
    if fix_aspect_ratio:
        f_scalar = (fx + fy) / 2.0
        fx = fy = f_scalar
    return fx, fy


def estimate_extrinsics(
    object_points, image_points, active_indices, intrinsics, dist_coeffs
):
    """
    Estimate extrinsic parameters for each image.
    For planar data, uses homography with RANSAC; for non-planar data, uses cv2.solvePnP with LM refinement.
    """
    extrinsics_dict = {}
    K = np.array(
        [
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1],
        ]
    )
    for idx in active_indices:
        img_pts = image_points.get(idx, None)
        obj_pts = object_points.get(idx, None)
        if img_pts is None or obj_pts is None:
            continue
        img_pts_reshaped = img_pts.reshape(-1, 1, 2).astype(np.float64)
        undistorted_pts = cv2.undistortPoints(
            img_pts_reshaped, K, np.array(dist_coeffs).flatten()
        )
        img_pts_norm = undistorted_pts.reshape(-1, 2)
        num_pts = img_pts_norm.shape[0]
        obj_pts_mean = np.mean(obj_pts, axis=0, keepdims=True)
        Y = obj_pts - obj_pts_mean
        _, S, _ = np.linalg.svd(Y.T @ Y)
        planar_ratio = S[2] / S[1] if S[1] != 0 else 0
        if planar_ratio < 1e-6 or num_pts < 5:
            H, _ = cv2.findHomography(img_pts_norm, obj_pts[:, :2], cv2.RANSAC)
            if H is None:
                continue
            h1 = H[:, 0]
            h2 = H[:, 1]
            norm_sum = np.linalg.norm(h1) + np.linalg.norm(h2)
            if norm_sum < 1e-6:
                continue
            lambda_ = 2.0 / norm_sum
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
                obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                raise RuntimeError("PnP estimation failed")
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts,
            img_pts,
            K,
            dist_coeffs,
            rvec,
            tvec,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
        )
        extrinsics_dict[idx] = Extrinsics(
            rvec=np.array(rvec).flatten(), tvec=np.array(tvec).flatten()
        )
    return extrinsics_dict
