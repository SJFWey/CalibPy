import cv2
import numpy as np
from scipy.optimize import least_squares

from .optimizer_params import (
    OptimizerParams,
    CalibrationFlags,
    OptimizationState,
    ParamsGuess,
    Intrinsics,
    Extrinsics,
)

np.set_printoptions(
    suppress=True, precision=2, formatter={"float_kind": "{:0.2f}".format}
)


def rodrigues_to_euler(rvec):
    """
    Convert Rodrigues rotation vector to Euler angles (roll, pitch, yaw) in degrees.
    """
    R, _ = cv2.Rodrigues(rvec)

    # Extract angles - using math to avoid potential gimbal lock issues
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))

    if np.abs(pitch) > np.pi / 2 - 1e-8:  # Close to ±90°
        # Special case near pitch = ±90°
        roll = 0  # Set roll to zero in this case
        if pitch > 0:
            yaw = np.arctan2(R[1, 2], R[1, 1])
        else:
            yaw = -np.arctan2(R[1, 2], R[1, 1])
    else:
        # Normal case
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    # Convert to degrees
    return np.array([np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)])


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
        optimizer_params: OptimizerParams = None,
        flags: CalibrationFlags = None,
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

        # Use provided flags or create default ones
        self.flags = flags if flags is not None else CalibrationFlags()
        self.optim_flags = self.flags.to_dict()

        # Use provided optimizer params or create default ones
        self.optim_params = (
            optimizer_params if optimizer_params is not None else OptimizerParams()
        )

        # Initialize optimization state
        self.optim_state = OptimizationState()

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
        if self.object_points[i].shape[0] == 0:
            return np.array([])
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
        if projected_points is None:
            return np.array([])
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
        if X.shape != (3, 1):
            raise ValueError(f"Invalid point shape: {X.shape}, expected (3, 1)")

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
        if np.any(np.isnan(J_global)) or np.any(np.isinf(J_global)):
            print("Warning: NaN or Inf values in Jacobian matrix")
            return J_global

        try:
            cond_number = np.linalg.cond(J_global)
            if cond_number > 1e12:
                print(
                    f"Warning: Jacobian condition number is high ({cond_number:.2e}), calibration may be unstable."
                )
        except np.linalg.LinAlgError:
            print("Warning: Could not compute condition number")

        return J_global

    def estimate_uncertainty(self, J, residuals):
        """
        Estimate calibration parameter uncertainty and correlations:
          - Compute covariance matrix: cov = (J^T J)^-1 * σ², where σ² is the residual variance
          - Extract standard deviations (diagonal elements)
          - Compute correlation matrix from covariance matrix

        Returns:
            Tuple of (covariance matrix, standard deviations, correlation matrix)
            or None if computation fails
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
            cov_matrix = JTJ_inv * sigma2

            # Extract standard deviations (sqrt of diagonal elements)
            param_std = np.sqrt(np.diag(cov_matrix))

            # Compute correlation matrix
            # corr_ij = cov_ij / (std_i * std_j)
            std_outer = np.outer(param_std, param_std)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.where(std_outer > 0, cov_matrix / std_outer, 0)

            # Set diagonal to exactly 1.0 to avoid numerical issues
            np.fill_diagonal(corr_matrix, 1.0)

            return cov_matrix, param_std, corr_matrix

        except np.linalg.LinAlgError:
            print("Warning: Inversion of J^T J failed, uncertainty estimation aborted.")
            return None

    def remove_outliers(self, threshold_sigma: float = 3.0):
        """
        Remove points whose reprojection error exceeds threshold_sigma * standard_deviation.
        Uses Median Absolute Deviation (MAD) for robust statistics.

        Args:
            threshold_sigma: Number of standard deviations for outlier threshold (default: 3.0)

        Returns:
            The number of removed points
        """
        if threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be positive")

        removed_count = 0
        new_object_points = {}
        new_image_points = {}

        # Collect all reprojection errors
        all_errors = []
        for i in self.indices:
            errors = self.calc_single_image_error(i)
            if len(errors) > 0:
                errors = errors.reshape(-1, 2)
                norm_errors = np.linalg.norm(errors, axis=1)
                all_errors.extend(norm_errors)

        if not all_errors:
            return 0

        # Calculate robust statistics using median and MAD
        all_errors = np.array(all_errors)
        median_error = np.median(all_errors)
        # MAD is more robust to outliers than standard deviation
        mad = np.median(np.abs(all_errors - median_error))
        # Convert MAD to standard deviation estimate (1.4826 factor assuming normal distribution)
        sigma = 1.4826 * mad

        # Calculate adaptive threshold
        adaptive_threshold = median_error + threshold_sigma * sigma

        for i in self.indices:
            errors = self.calc_single_image_error(i)
            if len(errors) == 0:
                # Skip empty images
                new_object_points[i] = self.object_points[i]
                new_image_points[i] = self.image_points[i]
                continue

            errors = errors.reshape(-1, 2)
            norm_errors = np.linalg.norm(errors, axis=1)
            mask = norm_errors < adaptive_threshold

            # Ensure we don't remove all points from an image
            if not np.any(mask):
                print(
                    f"Warning: All points would be removed from image {i}, keeping the best 3"
                )
                best_indices = np.argsort(norm_errors)[:3]
                mask[best_indices] = True

            new_object_points[i] = self.object_points[i][mask]
            new_image_points[i] = self.image_points[i][mask]
            removed_count += np.count_nonzero(~mask)

        self.object_points = new_object_points
        self.image_points = new_image_points

        return removed_count

    def calibrate(self):
        """
        Perform camera calibration by minimizing the reprojection error.
        """
        for iter in range(self.optim_params.max_outlier_iter):
            # Count total points before optimization
            total_points = sum(len(points) for points in self.image_points.values())

            res = least_squares(
                fun=self.calc_reproj_error,
                x0=self.pack_parameters(),
                jac=self.calc_reproj_jac_analytical,
                loss=self.optim_params.loss,
                method=self.optim_params.opt_method,
                f_scale=self.optim_params.loss_scale,
                ftol=self.optim_params.ftol,
                gtol=self.optim_params.gtol,
                xtol=self.optim_params.xtol,
                max_nfev=self.optim_params.max_nfev,
                verbose=self.optim_params.verbose,
            )

            # Update optimization state
            self.optim_state.update(res.x, res.cost)
            self.unpack_parameters(res.x)

            removed = self.remove_outliers(self.optim_params.outlier_threshold)
            if removed == 0:
                break  # no more outliers removed, stop
            else:
                remaining_points = sum(
                    len(points) for points in self.image_points.values()
                )
                print(
                    f"Iteration {iter + 1}: Removed {removed} points from {total_points} points ({remaining_points} points remaining)"
                )

        # Continue with final uncertainty estimation and summary:
        optimized_params = self.pack_parameters()

        # Recompute global Jacobian using analytical Jacobian for uncertainty estimation
        J_global = self.calc_reproj_jac_analytical(optimized_params)
        residuals = self.calc_reproj_error(optimized_params)

        # Update uncertainty estimation handling
        uncertainty = self.estimate_uncertainty(J_global, residuals)
        if uncertainty is not None:
            cov_matrix, param_std, corr_matrix = uncertainty
        else:
            cov_matrix, param_std, corr_matrix = None, None, None

        # Compute reprojection error statistics for each image
        reproj_errors = []
        for i in self.indices:
            err = self.calc_single_image_error(i)
            err_norm = np.linalg.norm(err) / (len(err) / 2)
            reproj_errors.append(err_norm)
        mean_error = np.mean(reproj_errors)
        std_error = np.std(reproj_errors)

        # Print calibration results
        # Convert rotation vectors to euler angles for each image
        extrinsics_with_angles = []
        for i in self.indices:
            euler_angles = rodrigues_to_euler(self.extrinsics[i].rvec)
            extrinsics_with_angles.append(
                {
                    "rvec": self.extrinsics[i].rvec,
                    "tvec": self.extrinsics[i].tvec,
                    "euler_angles": euler_angles,  # [roll, pitch, yaw] in degrees
                }
            )

        extrinsics_with_angles = []
        for i in self.indices:
            euler_angles = rodrigues_to_euler(self.extrinsics[i].rvec)
            extrinsics_with_angles.append(
                {
                    "rvec": self.extrinsics[i].rvec,
                    "tvec": self.extrinsics[i].tvec,
                    "euler_angles": euler_angles,  # [roll, pitch, yaw] in degrees
                }
            )

        return {
            "fc": [self.intrinsics.fx, self.intrinsics.fy],
            "pp": [self.intrinsics.cx, self.intrinsics.cy],
            "skew": self.intrinsics.skew,
            "dist_coeffs": self.intrinsics.dist_coeffs,
            "extrinsics": extrinsics_with_angles,
            "residual": res.cost,
            "covariance": cov_matrix,
            "param_std": param_std,
            "param_correlations": corr_matrix,
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

    fx = 1.0 / np.sqrt(inv_fx2) if inv_fx2 > 1e-8 else None
    fy = 1.0 / np.sqrt(inv_fy2) if inv_fy2 > 1e-8 else None

    if fx is None or fy is None:
        return None, None

    if fix_aspect_ratio:
        f_scalar = (fx + fy) / 2.0
        fx = fy = f_scalar

    return fx, fy


def estimate_extrinsics(object_points, image_points, intrinsics):
    """
    Estimate extrinsic parameters for each image.
    For planar data, uses homography with RANSAC; for non-planar data, uses cv2.solvePnP with LM refinement.

    Args:
        object_points: Dictionary of 3D object points per image
        image_points: Dictionary of 2D image points per image
        intrinsics: Camera intrinsic parameters

    Returns:
        Dictionary of extrinsic parameters per image

    Raises:
        ValueError: If input points are empty or invalid
        RuntimeError: If PnP estimation fails
    """
    if not object_points or not image_points:
        raise ValueError("Empty object_points or image_points")

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
            H, _ = cv2.findHomography(obj_pts[:, :2], img_pts_norm, cv2.RANSAC)

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
            # Try different PnP methods if initial one fails
            methods = [cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_DLS]
            for method in methods:
                try:
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, K, intrinsics.dist_coeffs, flags=method
                    )
                    if success:
                        break
                except cv2.error:
                    continue
            else:
                print(f"Warning: All PnP methods failed for image {i}")
                continue

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
