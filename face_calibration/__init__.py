"""face-calibration: Rig calibration via implicit differentiation for facial animation.

Core classes:
    TrackerFunction          -- Implicit differentiation through a live tracker.
    TrackerFunctionSeparate  -- Implicit differentiation with pre-computed controls.
    LBFGSTracker             -- L-BFGS tracker with implicit diff backward pass.
    LBFGSTrackerSeparate     -- Per-expression variant with condensation masks.
    LBFGSPerformanceTracker  -- Non-differentiable tracker for evaluation.
    RigCalibrationSolver     -- Simon-Says rig calibration.
    RigFineTuner             -- Multi-stage tracker fine-tuning.
"""

__version__ = "0.1.0"

from .implicit_diff import TrackerFunction, TrackerFunctionSeparate
from .tracker import LBFGSTracker, LBFGSTrackerSeparate, LBFGSPerformanceTracker
from .solver import RigCalibrationSolver
from .finetune import RigFineTuner
from .expressions import (
    create_expressions,
    create_flame_expressions,
    create_flame_combinations,
    FLAME_SIMON_SAYS_20,
    FLAME_COMBINATIONS,
)

__all__ = [
    "TrackerFunction",
    "TrackerFunctionSeparate",
    "LBFGSTracker",
    "LBFGSTrackerSeparate",
    "LBFGSPerformanceTracker",
    "RigCalibrationSolver",
    "RigFineTuner",
    "create_expressions",
    "create_flame_expressions",
    "create_flame_combinations",
    "FLAME_SIMON_SAYS_20",
    "FLAME_COMBINATIONS",
]
