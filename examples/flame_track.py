#!/usr/bin/env python3
"""Track a performance using a calibrated FLAME rig.

Given target geometry (e.g. from a face reconstruction method), this
script uses the L-BFGS tracker to find animation controls for each frame.

Usage:
    python examples/flame_track.py --config examples/configs/example_flame.yaml --targets targets.npy
"""

import argparse
import os

import numpy as np
import torch
import yaml

from face_calibration.tracker import LBFGSPerformanceTracker
from face_calibration.flame import SemanticFlameRigPytorch
from face_calibration.utils import get_path


def main():
    parser = argparse.ArgumentParser(description="Track performance with FLAME rig")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-t", "--targets", type=str, required=True,
                        help="Path to (num_frames, num_verts, 3) numpy array of target geometry")
    parser.add_argument("-o", "--output", type=str, default="tracking_result.npy")
    parser.add_argument("--calibrated-model", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cpu"
    rig = SemanticFlameRigPytorch(
        config, geom_path=args.calibrated_model, face_only=True, device=device
    )

    # Load targets
    targets = torch.from_numpy(np.load(args.targets)).float().to(device)
    if targets.dim() == 3:
        targets = targets.view(targets.shape[0], -1)  # flatten to (B, V*3)
    print(f"Tracking {targets.shape[0]} frames...")

    tracker = LBFGSPerformanceTracker(
        rig, targets,
        output_key=None,  # face_only=True returns tensor directly
        scale=1000.0,
        iterations=args.iterations,
        max_iter=args.max_iter,
        l1_reg=1e-4,
        log=True,
    )

    result = tracker.run(rig.rig_parameters)
    np.save(args.output, result.numpy())
    print(f"Saved tracking result to {args.output} (shape: {result.shape})")


if __name__ == "__main__":
    main()
