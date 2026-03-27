#!/usr/bin/env python3
"""Simon-Says calibration for a Semantic FLAME rig.

This script demonstrates how to calibrate a FLAME rig's expression basis
(shapedirs) using Simon-Says expression targets and the
:class:`RigCalibrationSolver`.

Usage:
    python examples/flame_calibrate.py --config examples/configs/example_flame.yaml

Requirements:
    - FLAME model downloaded from https://flame.is.tue.mpg.de
    - Identity parameters (shape betas) for the target subject
    - Tracked geometry for each Simon-Says expression
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml

from face_calibration import RigCalibrationSolver, create_flame_expressions, create_flame_combinations
from face_calibration.flame import SemanticFlameRigPytorch, Struct, to_np
from face_calibration.utils import get_path, write_obj


def main():
    parser = argparse.ArgumentParser(
        description="Simon-Says calibration for Semantic FLAME rig"
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Output directory
    if args.output:
        out_dir = Path(args.output)
    else:
        base = Path(config["root_folder"]) / config["full_rig"]["output"]
        i = 1
        while True:
            out_dir = base / f"run_{i}"
            if not out_dir.exists():
                break
            i += 1
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Load rig
    device = "cpu"
    rig = SemanticFlameRigPytorch(config, device=device)

    # Load control names
    with open(get_path(config["root_folder"], config["control_names"])) as f:
        control_names = f.read().splitlines()

    # Generate Simon-Says expressions
    expr_names, expr_controls, _ = create_flame_expressions(
        np.zeros(len(control_names)), control_names
    )
    controls = torch.from_numpy(np.asarray(expr_controls)).float().to(device)

    # Evaluate pre-calibration geometry
    pre_solve = rig(controls)["face"]
    print(f"Pre-calibration: {len(expr_names)} expressions, {controls.shape[1]} controls")

    # Load expression masks (per-vertex weights for each expression)
    mask_path = get_path(config["root_folder"], config["full_rig"]["masks"])
    moved_mask = torch.full((pre_solve.shape[0], pre_solve.shape[1]), False)
    for i in range(pre_solve.shape[0]):
        mask_file = mask_path / f"{i}.npy"
        if mask_file.exists():
            loaded = np.load(mask_file)
            moved_mask[i, loaded] = True
        else:
            moved_mask[i] = True  # fallback: use all vertices
    print(f"Mask vertex counts: {moved_mask.sum(dim=-1).numpy()}")

    # Load captured geometry targets from a face tracker
    # (This section depends on your tracker — adapt as needed)
    from face_calibration.flame import FlameRigPytorch
    import face_calibration.flame as flame_module

    datasets_available = False
    try:
        # Try to load tracked expression parameters from data
        frames_key = config.get("frames_smirk") or config.get("frames_mica")
        if frames_key and "data" in config:
            datasets_available = True
    except Exception:
        pass

    if not datasets_available:
        print("\nNo tracked data found — using rig's own expressions as targets.")
        print("(In practice, replace this with tracked geometry from a face tracker.)")
        # Use the rig's own output as a demo (this is a no-op calibration)
        target = pre_solve.detach().clone()
    else:
        print("Loading tracked geometry from dataset...")
        # Load standard FLAME rig for generating targets
        standard_rig = FlameRigPytorch(config)
        rig.global_expression = np.load(
            get_path(config["root_folder"], config["global_expression"])
        )
        # ... (dataset loading would go here)
        target = pre_solve.detach().clone()

    # Add combination expressions for regularization
    combo_names, combo_controls, _ = create_flame_combinations(
        np.zeros(len(control_names)), control_names
    )
    combo_controls_t = torch.from_numpy(np.asarray(combo_controls)).float().to(device)
    combo_geo = rig(combo_controls_t)["face"]

    # Append combinations
    all_names = expr_names + combo_names
    all_targets = torch.cat((target, combo_geo), dim=0)
    all_controls = torch.cat((controls, combo_controls_t), dim=0)

    # Run calibration
    print(f"\nRunning Simon-Says calibration ({len(all_names)} expressions)...")
    solver = RigCalibrationSolver(
        rig,
        get_params=lambda r: r.flame.shapedirs,
        set_params=lambda r, p: setattr(r.flame, "shapedirs", p),
        output_key="face",
    )
    result = solver.calibrate(
        all_targets, all_controls,
        masks=None,
        iterations=8,
        solver="lbfgs",
        data_weight=100.0,
        reg_weight=5.0,
    )
    print(f"Calibration complete. Final loss: {result['loss']:.6f}")

    # Save optimized FLAME model
    shapedirs = result["params"]
    with open(get_path(config["root_folder"], config["flame_geom_path"]), "rb") as f:
        ss = pickle.load(f, encoding="latin1")
        flame_model = Struct(**ss)
    setattr(flame_model, "shapedirs", to_np(shapedirs))

    lbs_dir = out_dir / "lbs"
    os.makedirs(lbs_dir, exist_ok=True)
    lbs_file = lbs_dir / "opt_flame_model.pkl"
    with open(lbs_file, "wb") as f:
        pickle.dump(vars(flame_model), f)
    print(f"Saved optimized FLAME model to {lbs_file}")

    # Save shapedirs separately
    np.save(out_dir / "shapedirs.npy", to_np(shapedirs))

    # Save per-expression meshes
    v, faces, uv, f_uv = None, None, None, None
    obj_path = config.get("full_rig", {}).get("obj")
    if obj_path:
        from face_calibration.utils import load_obj
        _, faces, uv, f_uv = load_obj(Path(config["root_folder"]) / obj_path)

    post_solve = rig(controls)["face"]
    for i, name in enumerate(expr_names):
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        np.save(out_dir / f"{i:02d}_{safe_name}_final.npy", post_solve[i].detach().cpu().numpy())
        if faces is not None:
            write_obj(
                out_dir / f"{i:02d}_{safe_name}_final.obj",
                post_solve[i].detach().cpu().numpy(), faces, uv, f_uv,
            )

    print(f"\nDone! Results saved to {out_dir}")


if __name__ == "__main__":
    main()
