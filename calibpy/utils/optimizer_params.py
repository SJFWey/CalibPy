from dataclasses import dataclass, field
import numpy as np


@dataclass
class CalibrationFlags:
    """Flags controlling which parameters to optimize."""

    estimate_focal: np.ndarray = field(default_factory=lambda: np.array([True, True]))
    estimate_principal: bool = True
    estimate_skew: bool = True
    estimate_distort: np.ndarray = field(
        default_factory=lambda: np.array([True, True, True, True, True])
    )
    estimate_extrinsics: bool = True
    fix_aspect_ratio: bool = False

    def __post_init__(self):
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


@dataclass
class OptimizerParams:
    """Configuration parameters for the optimization process."""

    max_iter: int = 100
    step: float = 0.1
    verbose: int = 1
    outlier_threshold: float = 0.5
    max_outlier_iter: int = 3
    optimization_method: str = "trf"
    loss: str = "soft_l1"
    loss_scale: float = 1.0
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8

    def __post_init__(self):
        self.max_nfev = self.max_iter * 2


@dataclass
class OptimizationState:
    """Tracks the state of the optimization process."""

    params_history: np.ndarray = None
    current_iteration: int = 0
    current_error: float = float("inf")
    best_error: float = float("inf")
    best_params: np.ndarray = None
    converged: bool = False

    def update(self, params, error):
        if self.params_history is None:
            self.params_history = params.reshape(1, -1)
        else:
            self.params_history = np.vstack([self.params_history, params])

        self.current_iteration += 1
        self.current_error = error

        if error < self.best_error:
            self.best_error = error
            self.best_params = params.copy()


@dataclass
class ParamsGuess:
    """
    Initialize camera intrinsic parameters.
    """

    def __init__(
        self,
        image_size,
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        skew=None,
        dist_coeffs=None,
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
            self.dist_coeffs = np.zeros(5, dtype=np.float64)
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
