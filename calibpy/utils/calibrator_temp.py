import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

np.set_printoptions(
    suppress=True, precision=2, formatter={"float_kind": "{:0.2f}".format}
)


@dataclass
class ParamsGuess:
    """
    Initialize camera intrinsic parameters.
    """

    def __init__(
        self, image_size, fx=None, fy=None, cx=None, cy=None, skew=None, dist_coeffs=None
    ):
        self.fx = fx
        self.fy = fy
        if cx is None:
            self.cx = image_size[0] / 2
        else:
            self.cx = cx
        if cy is None:
            self.cy = image_size[1] / 2
        else:
            self.cy = cy
        if skew is None:
            self.skew = 0.0
        else:
            self.skew = skew
        if dist_coeffs is None:
            self.dist_coeffs = np.zeros((5,), dtype=float)
        else:
            self.dist_coeffs = dist_coeffs

    def is_none(self):
        if self.fx is None or self.fy is None:
            return True


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

    def __init__(
        self,
        calib_data: dict,
        params_guess: ParamsGuess,
        max_iter: int = 100,
        step: float = 0.1,
        verbose: int = 1,
    ):
        self.fx = params_guess.fx
        self.fy = params_guess.fy
        self.cx = params_guess.cx
        self.cy = params_guess.cy
        self.skew = params_guess.skew
        self.dist_coeffs = params_guess.dist_coeffs

        self.object_points = calib_data.get("obj_points", {})
        self.image_points = calib_data.get("img_points", {})
        self.indices = sorted(self.image_points.keys())

        self.flags = CalibrationFlags()
        self.optim_flags = self.flags.to_dict()

        self.init_pp = np.array([self.cx, self.cy])
        if self.flags.fix_aspect_ratio:
            if self.fx != self.fy:
                self.init_fc = np.array(
                    [(self.fx + self.fy) // 2, (self.fx + self.fy) // 2]
                )
        else:
            self.init_fc = np.array([self.fx, self.fy])

        if params_guess.is_none():
            fx_est, fy_est = estimate_intrinsics(
                self.object_points,
                self.image_points,
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

        self.dist_coeffs = self.dist_coeffs * self.flags.estimate_distort.astype(int)

        print(
            f"Initial estimates: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, skew={self.skew}, dist_coeffs={self.dist_coeffs}"
        )

        self.rvecs = {}
        self.tvecs = {}
        extrinsics_est = estimate_extrinsics(
            self.object_points,
            self.image_points,
            Intrinsics(self.fx, self.fy, self.cx, self.cy, self.skew, self.dist_coeffs),
        )

        self.intrinsics = Intrinsics(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            skew=self.skew,
            dist_coeffs=self.dist_coeffs,
        )

        for ind in self.indices:
            if ind in extrinsics_est:
                self.rvecs[ind] = extrinsics_est[ind].rvec
                self.tvecs[ind] = extrinsics_est[ind].tvec
            else:
                self.rvecs[ind] = np.zeros(3)
                self.tvecs[ind] = np.zeros(3)

        self.extrinsics = {}
        for ind in self.indices:
            rvec = self.rvecs.get(ind, np.zeros(3))
            tvec = self.tvecs.get(ind, np.zeros(3))
            self.extrinsics[ind] = Extrinsics(
                rvec=np.array(rvec).flatten(), tvec=np.array(tvec).flatten()
            )

        self.params_history = self.pack_parameters()

        self.max_iter = max_iter
        self.alpha_smooth = step
        self.verbose = verbose

        # Set optimization method and loss parameters for robustness
        self.optimization_method = (
            "trf"  # Trust-region reflective algorithm supporting robust loss
        )
        self.loss = "soft_l1"  # Robust loss to reduce influence of outliers
        self.ftol = 1e-9
        self.xtol = 1e-9
        self.gtol = 1e-9

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
            for i in self.indices:
                extr = self.extrinsics[i]
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
            for i in self.indices:
                self.extrinsics[i].rvec = params[idx : idx + 3].astype(np.float64)
                idx += 3
                self.extrinsics[i].tvec = params[idx : idx + 3].astype(np.float64)
                idx += 3

    def calc_single_image_error(self, i):
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
        extr = self.extrinsics[i]
        projected_points, _ = cv2.projectPoints(
            self.object_points[i],
            extr.rvec,
            extr.tvec,
            K,
            self.intrinsics.dist_coeffs,
        )
        error = projected_points.reshape(-1, 2) - self.image_points[i]
        return error.ravel()

    def calc_reproj_error(self, flat_params):
        """
        Compute the concatenated reprojection error for all active images as a 1D vector.
        """
        self.unpack_parameters(flat_params)
        errors = []
        for ind in self.indices:
            err = self.calc_single_image_error(ind)
            errors.append(err)
        return np.concatenate(errors)

    def project_point_and_jacobian(self, X, intrinsics, extrinsics):
        """
        Compute projection and analytical Jacobian for a single 3D point.

        Parameters:
            X: 3D point in world coordinates.
            intrinsics: Intrinsic camera parameters.
            extrinsics: Extrinsic camera parameters (rotation and translation).
        Returns:
            proj: Projected 2D point in image.
            jac_intr: Jacobian of the projection with respect to intrinsic parameters.
            jac_extr: Jacobian of the projection with respect to extrinsic parameters.
        """

        # Rodrigues transformation: obtain rotation matrix and its derivative (dRodrigues is 9x3)
        rvec = extrinsics.rvec.reshape((3, 1))
        R, dRodrigues = cv2.Rodrigues(rvec)

        # Ensure the jacobian has shape (9,3) before reshaping.

        if dRodrigues.shape == (9, 3):
            dR_dr = dRodrigues.reshape(3, 3, 3)
        elif dRodrigues.shape == (3, 9):
            # Transpose to get shape (9,3)
            dR_dr = dRodrigues.T.reshape(3, 3, 3)
        else:
            raise ValueError(
                f"Unexpected shape of dRodrigues: expected (9,3) or (3,9), got {dRodrigues.shape}"
            )

        # Transform the 3D point into the camera coordinate system
        X = X.reshape(3, 1)
        tvec = extrinsics.tvec.reshape((3, 1))
        X_cam = R @ X + tvec
        Xc, Yc, Zc = X_cam.flatten()

        eps = 1e-8
        Zc = np.maximum(Zc, eps)

        # Compute normalized coordinates
        x = Xc / Zc
        y = Yc / Zc
        r2 = x * x + y * y

        # Extract intrinsic parameters
        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.cx
        cy = intrinsics.cy
        skew = intrinsics.skew
        k1, k2, p1, p2, k3 = intrinsics.dist_coeffs

        # Compute distortion components
        radial = 1 + k1 * r2 + k2 * (r2**2) + k3 * (r2**3)
        dx_tan = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy_tan = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x_d = x * radial + dx_tan
        y_d = y * radial + dy_tan

        # Project to image coordinates
        u = fx * x_d + skew * y_d + cx
        v = fy * y_d + cy
        proj = np.array([u, v])

        # Compute partial derivatives of distortion with respect to normalized coordinates
        # Derivatives of radial
        d_radial_dx = 2 * k1 * x + 4 * k2 * r2 * x + 6 * k3 * (r2**2) * x
        d_radial_dy = 2 * k1 * y + 4 * k2 * r2 * y + 6 * k3 * (r2**2) * y

        # Derivatives of dx_tan and dy_tan
        d_dx_tan_dx = 2 * p1 * y + 6 * p2 * x
        d_dx_tan_dy = 2 * p1 * x + 2 * p2 * y
        d_dy_tan_dx = 2 * p1 * x + 2 * p2 * y
        d_dy_tan_dy = 6 * p1 * y + 2 * p2 * x

        # Derivatives of x_d and y_d with respect to x and y
        dx_d_dx = radial + x * d_radial_dx + d_dx_tan_dx
        dx_d_dy = x * d_radial_dy + d_dx_tan_dy
        dy_d_dx = y * d_radial_dx + d_dy_tan_dx
        dy_d_dy = radial + y * d_radial_dy + d_dy_tan_dy

        # Derivatives of normalized coordinates with respect to camera coordinates
        # For x = Xc/Zc and y = Yc/Zc:
        dx_dxcam = np.array([1 / Zc, 0, -Xc / (Zc**2)])
        dy_dxcam = np.array([0, 1 / Zc, -Yc / (Zc**2)])

        # Chain rule: derivatives of x_d and y_d with respect to camera coordinates
        dx_dXc = dx_d_dx * dx_dxcam[0] + dx_d_dy * dy_dxcam[0]
        dx_dYc = dx_d_dx * dx_dxcam[1] + dx_d_dy * dy_dxcam[1]
        dx_dZc = dx_d_dx * dx_dxcam[2] + dx_d_dy * dy_dxcam[2]

        dy_dXc = dy_d_dx * dx_dxcam[0] + dy_d_dy * dy_dxcam[0]
        dy_dYc = dy_d_dx * dx_dxcam[1] + dy_d_dy * dy_dxcam[1]
        dy_dZc = dy_d_dx * dx_dxcam[2] + dy_d_dy * dy_dxcam[2]

        # Compute the Jacobian of the projection with respect to camera coordinates (X_cam)
        du_dXc = fx * dx_dXc + skew * dy_dXc
        du_dYc = fx * dx_dYc + skew * dy_dYc
        du_dZc = fx * dx_dZc + skew * dy_dZc

        dv_dXc = fy * dy_dXc
        dv_dYc = fy * dy_dYc
        dv_dZc = fy * dy_dZc

        J_proj = np.array([[du_dXc, du_dYc, du_dZc], [dv_dXc, dv_dYc, dv_dZc]])

        # Jacobian for intrinsic parameters
        # Parameters: [fx, fy, cx, cy, skew, k1, k2, p1, p2, k3]
        jac_intr = np.zeros((2, 10))
        jac_intr[0, 0] = x_d  # du/dfx
        jac_intr[1, 1] = y_d  # dv/dfy
        jac_intr[0, 2] = 1  # du/dcx
        jac_intr[1, 3] = 1  # dv/dcy
        jac_intr[0, 4] = y_d  # du/dskew

        # Derivatives with respect to distortion coefficients
        jac_intr[0, 5] = fx * (x * r2) + skew * (y * r2)  # du/dk1
        jac_intr[1, 5] = fy * (y * r2)  # dv/dk1

        jac_intr[0, 6] = fx * (x * (r2**2)) + skew * (y * (r2**2))  # du/dk2
        jac_intr[1, 6] = fy * (y * (r2**2))  # dv/dk2

        jac_intr[0, 7] = fx * (2 * x * y) + skew * (r2 + 2 * y * y)  # du/dp1
        jac_intr[1, 7] = fy * (r2 + 2 * y * y)  # dv/dp1

        jac_intr[0, 8] = fx * (r2 + 2 * x * x) + skew * (2 * x * y)  # du/dp2
        jac_intr[1, 8] = fy * (2 * x * y)  # dv/dp2

        jac_intr[0, 9] = fx * (x * (r2**3)) + skew * (y * (r2**3))  # du/dk3
        jac_intr[1, 9] = fy * (y * (r2**3))  # dv/dk3

        # Jacobian for extrinsic parameters
        # For rotation: use dRodrigues (reshape each column into a 3x3 matrix)
        jac_extr = np.zeros((2, 6))
        for i in range(3):
            # dR_dr[:,:,i] is the derivative of R with respect to rvec[i]
            dR_i = dR_dr[:, :, i]
            # Derivative of X_cam with respect to rvec[i] is dR_i @ X.
            dX_cam_dri = dR_i @ X
            # Chain rule: derivative of projection with respect to rvec[i].
            jac_extr[:, i] = (J_proj @ dX_cam_dri).flatten()

        # For translation: the derivative of X_cam with respect to tvec is identity
        jac_extr[:, 3:] = J_proj

        return proj, jac_intr, jac_extr

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
        for j in intrinsics_indices + distortion_indices:
            intrinsics_global_indices.append(j)

        num_intrinsics = len(intrinsics_global_indices)
        num_extrinsics = 6 * len(self.indices)
        total_params_count = num_intrinsics + num_extrinsics

        # Calculate total residual dimension (2D for each point in all active images)
        total_points = 0
        for idx in self.indices:
            pts = self.object_points[idx]
            total_points += pts.shape[0]
        residual_dim = total_points * 2

        J_global = np.zeros((residual_dim, total_params_count), dtype=np.float64)

        row_counter = 0
        # Extrinsics start index in global parameter vector is num_intrinsics, each image has 6 parameters
        extr_offset = num_intrinsics
        for count, i in enumerate(self.indices):
            extr_params_idx = extr_offset + count * 6
            obj_pts = self.object_points[i]
            for X in obj_pts:
                proj, jac_intr_full, jac_extr = self.project_point_and_jacobian(
                    X, self.intrinsics, self.extrinsics[i]
                )
                # Extract current optimized intrinsics part from jac_intr_full
                jac_intr_opt = jac_intr_full[:, intrinsics_global_indices]
                J_global[row_counter : row_counter + 2, 0:num_intrinsics] = jac_intr_opt
                J_global[
                    row_counter : row_counter + 2,
                    extr_params_idx : extr_params_idx + 6,
                ] = jac_extr
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
            verbose=self.verbose,
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
        for i in self.indices:
            err = self.calc_single_image_error(i)
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
                for i in self.indices
            ],
            "residual": res.cost,
            "covariance": cov_matrix,
            "param_std": param_std,
            "mean_reproj_error": mean_error,
            "std_reproj_error": std_error,
        }


def estimate_intrinsics(object_points, image_points, cx, cy, fix_aspect_ratio):
    """
    Estimate camera intrinsics assuming a planar calibration target.
    Uses RANSAC to compute homographies and simple least squares to solve for intrinsic parameters.
    """
    valid_homographies = []

    pp_offset = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])

    for i in sorted(image_points.keys()):
        img_pts = image_points[i]
        obj_pts = object_points[i]

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


def estimate_extrinsics(object_points, image_points, intrinsics):
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

    for i in sorted(image_points.keys()):
        img_pts = image_points[i]
        obj_pts = object_points[i]

        if img_pts is None or obj_pts is None:
            continue

        img_pts_reshaped = img_pts.reshape(-1, 1, 2).astype(np.float64)
        undistorted_pts = cv2.undistortPoints(
            img_pts_reshaped, K, intrinsics.dist_coeffs
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
                obj_pts,
                img_pts,
                K,
                intrinsics.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                raise RuntimeError("PnP estimation failed")

        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts,
            img_pts,
            K,
            intrinsics.dist_coeffs,
            rvec,
            tvec,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
        )

        extrinsics_dict[i] = Extrinsics(
            rvec=np.array(rvec).flatten(), tvec=np.array(tvec).flatten()
        )

    return extrinsics_dict
