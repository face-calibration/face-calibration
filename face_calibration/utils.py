"""Utility functions for face calibration: rotations, mesh I/O, path helpers."""

from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Batch helpers (adapted from roma)
# ---------------------------------------------------------------------------

def flatten_batch(tensor, end_dim):
    """Flatten multiple batch dimensions into one, or add a batch dim if none."""
    batch_shape = tensor.shape[:end_dim + 1]
    flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
    return flattened, batch_shape


def unflatten_batch(tensor, batch_shape):
    """Reverse :func:`flatten_batch`."""
    return tensor.reshape(batch_shape + tensor.shape[1:]) if len(batch_shape) > 0 else tensor.squeeze(0)


# ---------------------------------------------------------------------------
# 6-D rotation representation (replaces pytorch3d dependency)
# Zhou et al., "On the Continuity of Rotation Representations in Neural
# Networks", CVPR 2019.
# ---------------------------------------------------------------------------

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6-D rotation representation to 3x3 rotation matrix.

    Args:
        d6: (..., 6) tensor – the first two columns of the rotation matrix.

    Returns:
        (..., 3, 3) rotation matrix obtained by Gram-Schmidt orthogonalization.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to 6-D representation (first two columns).

    Args:
        matrix: (..., 3, 3) rotation matrix.

    Returns:
        (..., 6) tensor.
    """
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


# ---------------------------------------------------------------------------
# Quaternion / axis-angle / euler / matrix conversions (adapted from roma)
# Quaternion convention: XYZW (real part last)
# ---------------------------------------------------------------------------

def mat2quat(mat):
    """Rotation matrices (..., 3, 3) -> unit quaternions XYZW (..., 4)."""
    matrix, batch_shape = flatten_batch(mat, end_dim=-3)
    num_rotations, D1, D2 = matrix.shape
    assert (D1, D2) == (3, 3), "Input should be a Bx3x3 tensor."

    decision_matrix = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)
    decision_matrix[:, :3] = matrix.diagonal(dim1=1, dim2=2)
    decision_matrix[:, -1] = decision_matrix[:, :3].sum(axis=1)
    choices = decision_matrix.argmax(axis=1)

    quat = torch.empty((num_rotations, 4), dtype=matrix.dtype, device=matrix.device)

    ind = torch.nonzero(choices != 3, as_tuple=True)[0]
    i = choices[ind]
    j = (i + 1) % 3
    k = (j + 1) % 3
    quat[ind, i] = 1 - decision_matrix[ind, -1] + 2 * matrix[ind, i, i]
    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]

    ind = torch.nonzero(choices == 3, as_tuple=True)[0]
    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
    quat[ind, 3] = 1 + decision_matrix[ind, -1]

    quat = quat / torch.norm(quat, dim=1)[:, None]
    return unflatten_batch(quat, batch_shape)


def quat2mat(quat):
    """Unit quaternions XYZW (..., 4) -> rotation matrices (..., 3, 3)."""
    x, y, z, w = torch.unbind(quat, dim=-1)
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w

    matrix = torch.empty(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)
    matrix[..., 0, 0] = x2 - y2 - z2 + w2
    matrix[..., 1, 0] = 2 * (xy + zw)
    matrix[..., 2, 0] = 2 * (xz - yw)
    matrix[..., 0, 1] = 2 * (xy - zw)
    matrix[..., 1, 1] = -x2 + y2 - z2 + w2
    matrix[..., 2, 1] = 2 * (yz + xw)
    matrix[..., 0, 2] = 2 * (xz + yw)
    matrix[..., 1, 2] = 2 * (yz - xw)
    matrix[..., 2, 2] = -x2 - y2 + z2 + w2
    return matrix


def aa2quat(aa):
    """Axis-angle vectors (..., 3) -> unit quaternions XYZW (..., 4)."""
    aa, batch_shape = flatten_batch(aa, end_dim=-2)
    num_rotations, D = aa.shape
    assert D == 3, "Input should be a Bx3 tensor."

    norms = torch.norm(aa, dim=-1)
    small_angle = norms <= 1e-3
    large_angle = ~small_angle

    scale = torch.empty((num_rotations,), device=aa.device, dtype=aa.dtype)
    scale[small_angle] = 0.5 - norms[small_angle] ** 2 / 48 + norms[small_angle] ** 4 / 3840
    scale[large_angle] = torch.sin(norms[large_angle] / 2) / norms[large_angle]

    quat = torch.empty((num_rotations, 4), device=aa.device, dtype=aa.dtype)
    quat[:, :3] = scale[:, None] * aa
    quat[:, 3] = torch.cos(norms / 2)
    return unflatten_batch(quat, batch_shape)


def quat2aa(quat, shortest_arc=False):
    """Unit quaternions XYZW (..., 4) -> axis-angle vectors (..., 3)."""
    quat, batch_shape = flatten_batch(quat, end_dim=-2)
    quat = quat.clone()
    if shortest_arc:
        quat[quat[:, 3] < 0] *= -1
    half_angle = torch.atan2(torch.norm(quat[:, :3], dim=1), quat[:, 3])
    angle = 2 * half_angle
    small_angle = torch.abs(angle) <= 1e-3
    large_angle = ~small_angle

    num_rotations = len(quat)
    scale = torch.empty(num_rotations, dtype=quat.dtype, device=quat.device)
    scale[small_angle] = 2 + angle[small_angle] ** 2 / 12 + 7 * angle[small_angle] ** 4 / 2880
    scale[large_angle] = angle[large_angle] / torch.sin(half_angle[large_angle])

    rotvec = scale[:, None] * quat[:, :3]
    return unflatten_batch(rotvec, batch_shape)


def aa2mat(aa, eps=1e-6):
    """Axis-angle vectors (..., 3) -> rotation matrices (..., 3, 3) via Rodrigues."""
    aa, batch_shape = flatten_batch(aa, end_dim=-2)
    batch_size, D = aa.shape
    assert D == 3, "Input should be a Bx3 tensor."

    theta = torch.norm(aa, dim=-1)
    is_angle_small = theta < eps

    axis = aa / torch.clamp_min(theta[..., None], eps)
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    one_minus_cos_theta = 1 - cos_theta
    xs, ys, zs = kx * sin_theta, ky * sin_theta, kz * sin_theta
    xyc = kx * ky * one_minus_cos_theta
    xzc = kx * kz * one_minus_cos_theta
    yzc = ky * kz * one_minus_cos_theta
    xxc = kx ** 2 * one_minus_cos_theta
    yyc = ky ** 2 * one_minus_cos_theta
    zzc = kz ** 2 * one_minus_cos_theta
    R_rodrigues = torch.stack([
        1 - yyc - zzc, xyc - zs, xzc + ys,
        xyc + zs, 1 - xxc - zzc, -xs + yzc,
        xzc - ys, xs + yzc, 1 - xxc - yyc,
    ], dim=-1).reshape(-1, 3, 3)

    xs, ys, zs = aa[:, 0], aa[:, 1], aa[:, 2]
    one = torch.ones_like(xs)
    R_first_order = torch.stack([
        one, -zs, ys, zs, one, -xs, -ys, xs, one,
    ], dim=-1).reshape(-1, 3, 3)

    R = torch.where(is_angle_small[:, None, None], R_first_order, R_rodrigues)
    return unflatten_batch(R, batch_shape)


def mat2aa(mat):
    """Rotation matrices (..., 3, 3) -> axis-angle vectors (..., 3)."""
    return quat2aa(mat2quat(mat))


def mat2euler(mat, convention="XYZ"):
    """Rotation matrices (..., 3, 3) -> euler angles (..., 3)."""
    mat, batch_shape = flatten_batch(mat, end_dim=-3)
    if convention == "XYZ":
        y = torch.arcsin(torch.clamp(mat[..., 0, 2], -1, 1))
        has_z = torch.abs(mat[..., 0, 2]) < 0.9999999
        x_has_z = torch.atan2(-mat[..., 1, 2], mat[..., 2, 2])
        z_has_z = torch.atan2(-mat[..., 0, 1], mat[..., 0, 0])
        euler_has_z = torch.stack((x_has_z, y, z_has_z), dim=-1)
        x_no_z = torch.atan2(mat[..., 2, 1], mat[..., 1, 1])
        euler_no_z = torch.stack((x_no_z, y, torch.zeros_like(y)), dim=-1)
        euler = torch.where(has_z[:, None], euler_has_z, euler_no_z)
    elif convention == "ZYX":
        y = torch.arcsin(-torch.clamp(mat[..., 2, 0], -1, 1))
        has_x = torch.abs(mat[..., 2, 0]) < 0.9999999
        x_has_x = torch.atan2(mat[..., 2, 1], mat[..., 2, 2])
        z_has_x = torch.atan2(mat[..., 1, 0], mat[..., 0, 0])
        euler_has_x = torch.stack((x_has_x, y, z_has_x), dim=-1)
        z_no_x = torch.atan2(-mat[..., 0, 1], mat[..., 1, 1])
        euler_no_x = torch.stack((torch.zeros_like(y), y, z_no_x), dim=-1)
        euler = torch.where(has_x[:, None], euler_has_x, euler_no_x)
    else:
        raise ValueError(f"Unknown convention {convention}")
    return unflatten_batch(euler, batch_shape)


def euler2mat(euler, convention="XYZ"):
    """Euler angles (..., 3) -> rotation matrices (..., 3, 3)."""
    return quat2mat(euler2quat(euler, convention=convention))


def quat2euler(quat, convention="XYZ"):
    """Unit quaternions XYZW (..., 4) -> euler angles (..., 3)."""
    return mat2euler(quat2mat(quat), convention=convention)


def euler2quat(euler, convention="XYZ"):
    """Euler angles (..., 3) -> unit quaternions XYZW (..., 4)."""
    s1, s2, s3 = torch.unbind(torch.sin(euler / 2), dim=-1)
    c1, c2, c3 = torch.unbind(torch.cos(euler / 2), dim=-1)
    if convention == "XYZ":
        x = s1 * c2 * c3 + c1 * s2 * s3
        y = c1 * s2 * c3 - s1 * c2 * s3
        z = c1 * c2 * s3 + s1 * s2 * c3
        w = c1 * c2 * c3 - s1 * s2 * s3
    elif convention == "ZYX":
        x = s1 * c2 * c3 - c1 * s2 * s3
        y = c1 * s2 * c3 + s1 * c2 * s3
        z = c1 * c2 * s3 - s1 * s2 * c3
        w = c1 * c2 * c3 + s1 * s2 * s3
    else:
        raise ValueError(f"Unknown convention {convention}")
    return torch.stack((x, y, z, w), dim=-1)


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------

def load_obj(file_name, triangulate=True):
    """Load a Wavefront .obj file.

    Returns:
        (vertices, faces, uvs, face_uvs) as numpy arrays.
    """
    v, f, uv, f_uv = [], [], [], []
    with open(file_name) as fh:
        for line in fh:
            line = line.strip().split()
            if not line:
                continue
            if line[0] == "v":
                v.append([float(i) for i in line[1:]])
            elif line[0] == "vt":
                uv.append([float(i) for i in line[1:]])
            elif line[0] == "f":
                face = [int(i.split("/")[0]) - 1 for i in line[1:]]
                ft = None
                if "/" in line[1] and "//" not in line[1]:
                    ft = [int(i.split("/")[1]) - 1 for i in line[1:]]
                if len(face) == 4 and triangulate:
                    f.extend([[face[0], face[1], face[2]], [face[0], face[2], face[3]]])
                    if ft is not None:
                        f_uv.extend([[ft[0], ft[1], ft[2]], [ft[0], ft[2], ft[3]]])
                else:
                    f.append(face)
                    if ft is not None:
                        f_uv.append(ft)
    return np.asarray(v), np.asarray(f), np.asarray(uv), np.asarray(f_uv)


def write_obj(file_name, v, f=None, uv=None, f_uv=None):
    """Write a Wavefront .obj file."""
    with open(file_name, "w") as fh:
        for vert in v:
            fh.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        if uv is not None and len(uv) > 0:
            for tex in uv:
                fh.write(f"vt {tex[0]} {tex[1]}\n")
        if f is not None:
            f_out = np.asarray(f) + 1  # OBJ is 1-indexed
            if f_uv is not None and len(f_uv) > 0:
                f_uv_out = np.asarray(f_uv) + 1
                for face, ft in zip(f_out, f_uv_out):
                    indices = " ".join(f"{fi}/{ti}" for fi, ti in zip(face, ft))
                    fh.write(f"f {indices}\n")
            else:
                for face in f_out:
                    fh.write(f"f {' '.join(str(fi) for fi in face)}\n")


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def get_path(root, path):
    """Resolve *path* relative to *root* if it is not already absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(root) / path
