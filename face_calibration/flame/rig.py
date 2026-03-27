"""FLAME rig wrappers with differentiable rig parameters.

These rigs satisfy the interface required by the face_calibration package:

    ``forward(controls, rig_parameters=None)`` -> geometry (tensor or dict)

The ``rig_parameters`` argument allows the implicit differentiation
pipeline to compute Jacobians with respect to the rig's internal
parameters (shapedirs).
"""

import yaml

import numpy as np
import torch
import torch.nn as nn

from face_calibration.utils import euler2mat, get_path
from .flame_model import FLAMERig, Struct, to_np


class FlameConfig:
    pass


def create_flame_config(config):
    """Build a FlameConfig from a YAML-loaded dict."""
    fc = FlameConfig()
    root = config["root_folder"]
    fc.flame_geom_path = get_path(root, config["flame_geom_path"])
    fc.num_shape_params = config["num_shape_params"]
    fc.num_exp_params = config["num_exp_params"]
    fc.flame_lmk_path = get_path(root, config["flame_lmk_path"])
    fc.flame_template_path = get_path(root, config.get("flame_template_path", ""))
    fc.tex_space_path = get_path(root, config.get("tex_space_path", ""))
    return fc


class FlameRigPytorch(nn.Module):
    """Standard FLAME rig driven by expression + jaw parameters.

    Controls layout: ``[expression_0, ..., expression_N, jaw_rot_x, jaw_rot_y, jaw_rot_z]``

    Args:
        config: path to YAML config or parsed dict.
        geom_path: optional override for the FLAME model path.
        face_only: if True, ``forward`` returns a plain tensor instead of dict.
    """

    def __init__(self, config, geom_path=None, face_only=False, device="cpu"):
        super().__init__()
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.safe_load(f)

        flame_config = create_flame_config(config)
        self.device = device
        if geom_path:
            flame_config.flame_geom_path = geom_path

        self.flame = FLAMERig(flame_config, device=device)
        self.betas = torch.from_numpy(
            np.load(get_path(config["root_folder"], config["identity"]))
        ).to(device)
        self.cameras = torch.eye(3).to(device)

        self.controls_min = [-1] * flame_config.num_exp_params + [-1, -1, -1]
        self.controls_max = [1] * flame_config.num_exp_params + [1, 1, 1]
        self.control_names = [
            str(x) for x in range(flame_config.num_exp_params)
        ] + ["jaw_rot_x", "jaw_rot_y", "jaw_rot_z"]
        self.num_controls = flame_config.num_exp_params + 3
        self.face_only = face_only

    @property
    def rig_parameters(self):
        """Flat view of the optimizable expression shapedirs."""
        return self.flame.shapedirs.reshape(-1)

    def forward(self, controls, rig_parameters=None, shape_parameters=None):
        """
        Args:
            controls: (B, num_controls) or (num_controls,).
            rig_parameters: optional flat tensor to replace shapedirs.
            shape_parameters: optional (B, num_shape_params).

        Returns:
            ``{'face': (B, V, 3)}`` or ``(B, V, 3)`` if ``face_only``.
        """
        batched = True
        if controls.dim() == 1:
            batched = False
            controls = controls.unsqueeze(0)

        B = controls.shape[0]
        expression = controls[:, :-3]
        jaw = euler2mat(controls[:, -3:])[:, :2, :].reshape(-1, 6)

        if shape_parameters is not None:
            shape_ext = shape_parameters
            if shape_ext.dim() == 1:
                shape_ext = shape_ext.unsqueeze(0).expand(B, -1)
        else:
            shape_ext = self.betas.unsqueeze(0).expand(B, -1)

        shapedirs_override = None
        if rig_parameters is not None:
            shapedirs_override = rig_parameters.view(self.flame.shapedirs.shape)

        verts = self.flame(
            shape_ext, self.cameras,
            expression_params=expression,
            jaw_pose_params=jaw,
            shapedirs_override=shapedirs_override,
        )
        out = {"face": verts}

        if not batched:
            out = {k: v[0] for k, v in out.items()}
        if self.face_only:
            return out["face"]
        return out

    def eval_controls(self, controls):
        """Numpy convenience: returns ``{name: (V, 3) ndarray}``."""
        if isinstance(controls, np.ndarray):
            controls = torch.from_numpy(controls).float()
        return {k: v.detach().cpu().numpy() for k, v in self.forward(controls).items()}


class SemanticFlameRigPytorch(nn.Module):
    """Semantic FLAME rig with artist-friendly control names.

    Maps semantic controls (e.g. ``browDownL``, ``mouthCornerPullR``)
    through a learned basis matrix to FLAME expression + jaw parameters.

    Args:
        config: path to YAML config or parsed dict.
        geom_path: optional override for the FLAME model path.
        face_only: if True, ``forward`` returns a plain tensor instead of dict.
    """

    def __init__(self, config, geom_path=None, face_only=False, device="cpu"):
        super().__init__()
        if isinstance(config, str):
            with open(config) as f:
                config = yaml.safe_load(f)

        flame_config = create_flame_config(config)
        self.device = device
        if geom_path:
            flame_config.flame_geom_path = geom_path

        self.flame = FLAMERig(flame_config, device=device)
        self.betas = torch.from_numpy(
            np.load(get_path(config["root_folder"], config["identity"]))
        ).to(device)
        self.cameras = torch.eye(3).to(device)

        with open(get_path(config["root_folder"], config["control_names"])) as f:
            self.control_names = f.read().splitlines()

        self.num_controls = len(self.control_names)
        self.controls_min = [0] * self.num_controls
        self.controls_max = [1] * self.num_controls

        self.mh_to_flame_basis = torch.from_numpy(
            np.load(get_path(config["root_folder"], config["mh_to_flame_basis"]))
        ).to(device)
        self.global_expression = torch.from_numpy(
            np.load(get_path(config["root_folder"], config["global_expression"]))
        ).to(device)

        self.face_only = face_only

    @property
    def rig_parameters(self):
        """Flat view of the optimizable expression shapedirs."""
        return self.flame.shapedirs.reshape(-1)

    def forward(self, controls, rig_parameters=None, shape_parameters=None, ml_deltas=None):
        """
        Args:
            controls: (B, num_controls) or (num_controls,).
            rig_parameters: optional flat tensor to replace shapedirs.
            shape_parameters: optional (B, num_shape_params).
            ml_deltas: optional additive correction to mapped controls.

        Returns:
            ``{'face': (B, V, 3)}`` or ``(B, V, 3)`` if ``face_only``.
        """
        batched = True
        if controls.dim() == 1:
            batched = False
            controls = controls.unsqueeze(0)

        B = controls.shape[0]

        # Map semantic controls to FLAME parameter space
        deltas = torch.matmul(
            self.mh_to_flame_basis.unsqueeze(0), controls.unsqueeze(-1)
        ).squeeze(-1)
        flame_controls = deltas + self.global_expression

        if ml_deltas is not None:
            flame_controls = flame_controls + ml_deltas

        expression = flame_controls[:, :-3]
        jaw = euler2mat(flame_controls[:, -3:])[:, :2, :].reshape(-1, 6)

        if shape_parameters is not None:
            shape_ext = shape_parameters
            if shape_ext.dim() == 1:
                shape_ext = shape_ext.unsqueeze(0).expand(B, -1)
        else:
            shape_ext = self.betas.unsqueeze(0).expand(B, -1)

        shapedirs_override = None
        if rig_parameters is not None:
            shapedirs_override = rig_parameters.view(self.flame.shapedirs.shape)

        verts = self.flame(
            shape_ext, self.cameras,
            expression_params=expression,
            jaw_pose_params=jaw,
            shapedirs_override=shapedirs_override,
        )
        out = {"face": verts}

        if not batched:
            out = {k: v[0] for k, v in out.items()}
        if self.face_only:
            return out["face"]
        return out

    def eval_controls(self, controls):
        """Numpy convenience: returns ``{name: (V, 3) ndarray}``."""
        if isinstance(controls, np.ndarray):
            controls = torch.from_numpy(controls).float()
        return {k: v.detach().cpu().numpy() for k, v in self.forward(controls).items()}
