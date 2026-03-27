# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright 2023 Max-Planck-Gesellschaft zur Foerderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

"""FLAME parametric head model (adapted for face-calibration)."""

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from face_calibration.utils import matrix_to_rotation_6d, rotation_6d_to_matrix
from .lbs import lbs, rot_mat_to_euler

logger = logging.getLogger(__name__)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FLAMERig(nn.Module):
    """Differentiable FLAME head model.

    Given shape, pose, and expression parameters, outputs deformed mesh
    vertices via linear blend skinning.

    Args:
        config: object with attributes ``flame_geom_path``,
            ``num_shape_params``, ``num_exp_params``, ``flame_lmk_path``.
        device: torch device.
    """

    def __init__(self, config, device="cpu"):
        super().__init__()
        with open(config.flame_geom_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.device = device
        self.dtype = torch.float32

        self.register_buffer(
            "faces",
            self._to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long),
        )
        self.register_buffer(
            "v_template", self._to_tensor(to_np(flame_model.v_template), dtype=self.dtype)
        )

        # Shape and expression basis
        shapedirs = self._to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [
                shapedirs[:, :, : config.num_shape_params],
                shapedirs[:, :, 300 : 300 + config.num_exp_params],
            ],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)

        # Pose basis
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer(
            "posedirs", self._to_tensor(to_np(posedirs), dtype=self.dtype)
        )

        self.register_buffer(
            "J_regressor",
            self._to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype),
        )

        parents = self._to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        self.register_buffer(
            "lbs_weights",
            self._to_tensor(to_np(flame_model.weights), dtype=self.dtype),
        )

        # Eyelid blendshapes (optional)
        blendshape_dir = Path(os.path.abspath(os.path.dirname(__file__))) / "blendshapes"
        for side in ("l_eyelid", "r_eyelid"):
            path = blendshape_dir / f"{side}.npy"
            if path.exists():
                data = torch.from_numpy(np.load(str(path))).to(self.device).to(self.dtype)[None]
            else:
                logger.warning("Eyelid blendshape %s not found; eyelid params disabled.", path)
                data = torch.zeros(1, self.v_template.shape[0], 3, dtype=self.dtype, device=self.device)
            self.register_buffer(side, data)

        # Default parameters
        self._register_default_params("neck_pose_params", 6)
        self._register_default_params("jaw_pose_params", 6)
        self._register_default_params("eye_pose_params", 12)
        self._register_default_params("shape_params", config.num_shape_params)
        self._register_default_params("expression_params", config.num_exp_params)

        # Landmark embeddings (optional — not needed for calibration)
        try:
            mp_emb = np.load(
                Path(__file__).parent / "mediapipe/mediapipe_landmark_embedding.npz",
                allow_pickle=True, encoding="latin1",
            )
            self.mediapipe_idx = mp_emb["landmark_indices"].astype(int)
        except FileNotFoundError:
            logger.warning("Mediapipe landmark embedding not found; landmarks disabled.")
            self.mediapipe_idx = None

        try:
            lmk_embeddings = np.load(config.flame_lmk_path, allow_pickle=True, encoding="latin1")
            lmk_embeddings = lmk_embeddings[()]
            self.register_buffer(
                "lmk_faces_idx",
                torch.from_numpy(lmk_embeddings["static_lmk_faces_idx"].astype(int)).to(torch.int64),
            )
            self.register_buffer(
                "lmk_bary_coords",
                torch.from_numpy(lmk_embeddings["static_lmk_bary_coords"]).to(self.dtype),
            )
        except (FileNotFoundError, KeyError):
            logger.warning("Landmark embedding not found; landmarks disabled.")

    def _to_tensor(self, array, dtype=torch.float32):
        if not isinstance(array, torch.Tensor):
            return torch.tensor(array, dtype=dtype).to(self.device)

    def _register_default_params(self, param_fname, dim):
        default_params = torch.zeros([1, dim], dtype=self.dtype, requires_grad=False)
        self.register_parameter(param_fname, nn.Parameter(default_params, requires_grad=False))

    def forward(
        self, shape_params, cameras,
        trans_params=None, rot_params=None,
        neck_pose_params=None, jaw_pose_params=None,
        eye_pose_params=None, expression_params=None,
        eyelid_params=None, shapedirs_override=None,
    ):
        """
        Args:
            shape_params: (B, num_shape_params)
            cameras: (3, 3) camera rotation.
            expression_params: (B, num_exp_params)
            jaw_pose_params: (B, 6) jaw pose in 6-D rotation repr.
            shapedirs_override: optional (V, 3, D) to replace self.shapedirs.

        Returns:
            vertices: (B, V, 3)
        """
        batch_size = shape_params.shape[0]
        I = matrix_to_rotation_6d(
            torch.cat([torch.eye(3)[None]] * batch_size, dim=0)
        ).to(self.device)

        if trans_params is None:
            trans_params = torch.zeros(batch_size, 3)
        if rot_params is None:
            rot_params = I.clone()
        if neck_pose_params is None:
            neck_pose_params = I.clone()
        if jaw_pose_params is None:
            jaw_pose_params = I.clone()
        if eye_pose_params is None:
            eye_pose_params = torch.cat([I.clone()] * 2, dim=1)
        if expression_params is None:
            expression_params = self.expression_params.expand(batch_size, -1)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [rot_params, neck_pose_params, jaw_pose_params, eye_pose_params], dim=1
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        shapedirs = shapedirs_override if shapedirs_override is not None else self.shapedirs

        vertices, _ = lbs(
            betas, full_pose, template_vertices,
            shapedirs, self.posedirs,
            self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype,
        )

        if eyelid_params is not None:
            vertices = vertices + self.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            vertices = vertices + self.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]

        return vertices
