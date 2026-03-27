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

"""Linear blend skinning for FLAME."""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn.functional as F

from face_calibration.utils import rotation_6d_to_matrix


def rot_mat_to_euler(rot_mats):
    sy = torch.sqrt(
        rot_mats[:, 0, 0] * rot_mats[:, 0, 0]
        + rot_mats[:, 1, 0] * rot_mats[:, 1, 0]
    )
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def vertices2joints(J_regressor, vertices):
    return torch.einsum("bik,ji->bjk", [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    return torch.einsum("bl,mkl->bmk", [betas, shape_disps])


def transform_mat(R, t):
    return torch.cat(
        [F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2
    )


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms


def lbs(
    betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
    lbs_weights, pose2rot=True, dtype=torch.float32,
):
    """Linear Blend Skinning.

    Args:
        betas: (B, num_betas) shape/expression coefficients.
        pose: (B, (J+1)*6) pose in 6-D rotation representation.
        v_template: (B, V, 3) template vertices.
        shapedirs: (V, 3, num_betas) shape displacements.
        posedirs: (P, V*3) pose PCA coefficients.
        J_regressor: (J, V) joint regressor.
        parents: (J,) kinematic tree.
        lbs_weights: (V, J+1) skinning weights.

    Returns:
        verts: (B, V, 3) deformed vertices.
        joints: (B, J, 3) joint locations.
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)

    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = rotation_6d_to_matrix(pose.view(-1, 6)).view(
            [batch_size, -1, 3, 3]
        )
        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)
        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), posedirs).view(
            batch_size, -1, 3
        )

    v_posed = pose_offsets + v_shaped
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(
        batch_size, -1, 4, 4
    )

    homogen_coord = torch.ones(
        [batch_size, v_posed.shape[1], 1], dtype=dtype, device=device
    )
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed
