#!/usr/bin/env python3
"""Multi-stage tracker fine-tuning for a Semantic FLAME rig.

This script demonstrates how to fine-tune the tracker's internal rig
parameters using the 4-stage implicit differentiation pipeline.

Usage:
    python examples/flame_finetune.py --config examples/configs/example_flame.yaml

Requirements:
    - FLAME model downloaded from https://flame.is.tue.mpg.de
    - A calibrated rig (output of flame_calibrate.py)
    - Tracked geometry for Simon-Says expressions
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from face_calibration import RigFineTuner, create_flame_expressions
from face_calibration.flame import SemanticFlameRigPytorch
from face_calibration.utils import get_path

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-stage tracker fine-tuning for Semantic FLAME"
    )
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="output/finetune")
    parser.add_argument("--calibrated-model", type=str, default=None,
                        help="Path to calibrated FLAME model (from flame_calibrate.py)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output, exist_ok=True)
    device = "cpu"

    # Load rig (optionally with calibrated parameters)
    geom_path = args.calibrated_model
    rig = SemanticFlameRigPytorch(config, geom_path=geom_path, device=device)

    # Load control names
    with open(get_path(config["root_folder"], config["control_names"])) as f:
        control_names = f.read().splitlines()

    # Generate expressions
    expr_names, expr_controls, _ = create_flame_expressions(
        np.zeros(len(control_names)), control_names
    )
    true_controls = torch.from_numpy(np.asarray(expr_controls)).float().to(device)

    # Generate target geometry from the rig (in practice, use tracked geometry)
    print("Generating target geometry from rig...")
    with torch.no_grad():
        targets = rig(true_controls)["face"]

    print(f"Fine-tuning with {len(expr_names)} expressions, "
          f"{true_controls.shape[1]} controls, "
          f"{rig.rig_parameters.numel()} rig parameters")

    # Create fine-tuner
    finetuner = RigFineTuner(
        rig, targets, true_controls,
        output_key="face",
        scale=1000.0,  # FLAME geometry is in meters; scale up for numerical stability
    )

    # Run all 4 stages
    optimized_params = finetuner.run_all()

    # Save result
    np.save(
        os.path.join(args.output, "optimized_shapedirs.npy"),
        optimized_params.numpy(),
    )
    print(f"Saved optimized parameters to {args.output}/optimized_shapedirs.npy")

    # Evaluate: compare tracked controls before and after
    print("\n=== Evaluation ===")
    from face_calibration.tracker import LBFGSPerformanceTracker

    for label, params in [("Before", rig.rig_parameters), ("After", optimized_params)]:
        tracked = []
        for i in range(len(expr_names)):
            c = torch.zeros(rig.num_controls)
            c.requires_grad_(True)
            opt = torch.optim.LBFGS([c], max_iter=50)
            theta = params.detach()

            def closure():
                opt.zero_grad()
                v = rig(c, rig_parameters=theta)["face"]
                loss = ((v - targets[i]) ** 2).mean()
                loss.backward()
                return loss

            with torch.enable_grad():
                opt.step(closure)
            tracked.append(c.detach())

        tracked = torch.stack(tracked)
        error = (tracked - true_controls).abs().mean().item()
        print(f"  {label}: mean controls error = {error:.6f}")


if __name__ == "__main__":
    main()
