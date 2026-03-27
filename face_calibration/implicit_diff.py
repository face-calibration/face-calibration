"""Implicit differentiation through black-box trackers.

This module implements the core contribution of the paper:

    "Improving Facial Rig Semantics for Tracking and Retargeting"
    D. Omens, A. Thurman, J. Yu, R. Fedkiw

Given a rig R(c; theta) mapping controls c to geometry, and a tracker
T(I; theta) that inverts the rig to find controls from an image, the
derivative dT/dtheta is not directly available when the tracker is a
black box.  We compute it via implicit differentiation:

    dR/dc * dT/dtheta = -dR/dtheta          (Equation 7 in the paper)

solved in a least-squares sense.  This module packages that computation
as PyTorch autograd Functions so the derivative can be used in any
standard gradient-based optimization.

Two variants are provided:

* :class:`TrackerFunction` -- wraps a callable tracker; the forward pass
  actually runs the tracker.
* :class:`TrackerFunctionSeparate` -- takes pre-computed controls (the
  tracker has already been run); only the backward pass is non-trivial.

Both are rig-agnostic: they work with any differentiable PyTorch rig
whose ``forward(controls, rig_parameters=theta)`` returns geometry
(either a tensor or a dict of tensors).
"""

import logging
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


def _rig_output_to_flat(result):
    """Flatten rig output (tensor or dict of tensors) to a 1-D vector."""
    if isinstance(result, dict):
        return torch.cat([v.reshape(-1) for v in result.values()])
    return result.reshape(-1)


class TrackerFunction(torch.autograd.Function):
    """Implicit-differentiation autograd function wrapping a live tracker.

    **Forward**: runs ``tracker_func(rig_parameters)`` to obtain controls.

    **Backward**: computes ``dT/dtheta`` via the implicit equation
    ``dR/dc * dT/dtheta = -dR/dtheta`` (least-squares solve), then
    back-propagates gradients through ``rig_parameters``.

    The tracker itself need not be differentiable.
    """

    @staticmethod
    def forward(
        rig_parameters: torch.Tensor,
        tracker_func: Callable,
        rig: torch.nn.Module,
        dm_dtheta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            rig_parameters: (num_rig_parameters,) flat parameter vector.
            tracker_func: ``controls = f(rig_parameters)`` — runs the tracker.
            rig: a Module with ``forward(controls, rig_parameters=theta)``.
            dm_dtheta: optional Broyden correction term
                (B, num_rig_outputs, num_rig_parameters).

        Returns:
            controls: (B, num_controls) output of the tracker.
        """
        return tracker_func(rig_parameters)

    @staticmethod
    def setup_context(ctx, inputs, output):
        rig_parameters, tracker_func, rig, dm_dtheta = inputs
        controls = output
        ctx.save_for_backward(rig_parameters, controls, dm_dtheta)
        ctx.rig = rig

    @staticmethod
    def backward(ctx, grad_output):
        rig_parameters, controls, dm_dtheta = ctx.saved_tensors
        rig = ctx.rig

        def func(theta, c):
            return _rig_output_to_flat(rig(c, rig_parameters=theta))

        # Jacobian of rig output w.r.t. rig parameters: (B, M, P)
        dR_dtheta = torch.vmap(
            torch.func.jacfwd(func, argnums=0), in_dims=(None, 0)
        )(rig_parameters, controls)
        if dm_dtheta is not None:
            dR_dtheta = dR_dtheta - dm_dtheta

        # Jacobian of rig output w.r.t. controls: (B, M, C)
        dR_dc = torch.vmap(
            torch.func.jacfwd(func, argnums=1), in_dims=(None, 0)
        )(rig_parameters, controls)

        # Solve the implicit equation: dR/dc * dT/dtheta = -dR/dtheta
        # => dT/dtheta = lstsq(dR_dc, -dR_dtheta)   shape (B, C, P)
        dT_dtheta = torch.linalg.lstsq(dR_dc, -dR_dtheta, driver="gelsd").solution

        # Chain rule: grad_rig_parameters = grad_output @ dT/dtheta
        grad_rig_parameters = torch.matmul(
            grad_output.unsqueeze(1), dT_dtheta
        ).squeeze(1)

        return grad_rig_parameters, None, None, None


class TrackerFunctionSeparate(torch.autograd.Function):
    """Implicit-differentiation autograd function for pre-computed controls.

    Use this when the tracker has already been run and you have the
    resulting controls.  The forward pass is a simple passthrough; the
    backward pass computes the same implicit derivative as
    :class:`TrackerFunction`.

    This is the recommended variant for multi-stage optimization where
    the tracker is run once per iteration and gradients are computed
    separately.
    """

    @staticmethod
    def forward(
        rig_parameters: torch.Tensor,
        controls: torch.Tensor,
        rig: torch.nn.Module,
        dm_dtheta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            rig_parameters: (num_rig_parameters,) flat parameter vector.
            controls: (B, num_controls) pre-computed tracker output.
            rig: a Module with ``forward(controls, rig_parameters=theta)``.
            dm_dtheta: optional Broyden correction term.

        Returns:
            controls (unchanged — gradient flows through backward only).
        """
        return controls

    @staticmethod
    def setup_context(ctx, inputs, output):
        rig_parameters, controls, rig, dm_dtheta = inputs
        ctx.save_for_backward(rig_parameters, controls)
        ctx.rig = rig
        ctx.dm_dtheta = dm_dtheta

    @staticmethod
    def backward(ctx, grad_output):
        rig_parameters, controls = ctx.saved_tensors
        rig = ctx.rig
        dm_dtheta = ctx.dm_dtheta

        if rig_parameters.shape[0] == 0:
            return (
                torch.zeros((controls.shape[0], 0), dtype=rig_parameters.dtype),
                None, None, None,
            )

        def func(theta, c):
            return _rig_output_to_flat(rig(c, rig_parameters=theta))

        # dR/dtheta: (B, M, P)
        if hasattr(rig, "dR_dtheta_analytic"):
            dR_dtheta = rig.dR_dtheta_analytic(controls)
        else:
            dR_dtheta = torch.vmap(
                torch.func.jacfwd(func, argnums=0), in_dims=(None, 0)
            )(rig_parameters, controls)

        if dm_dtheta is not None:
            dR_dtheta = dR_dtheta - dm_dtheta

        # dR/dc: (B, M, C)
        dR_dc = torch.vmap(
            torch.func.jacfwd(func, argnums=1), in_dims=(None, 0)
        )(rig_parameters, controls)

        # Safety checks
        for name, tensor in [
            ("controls", controls), ("dR_dtheta", dR_dtheta), ("dR_dc", dR_dc),
        ]:
            if torch.any(torch.isnan(tensor)):
                raise ValueError(f"NaN detected in {name}")
            if torch.any(torch.isinf(tensor)):
                raise ValueError(f"Inf detected in {name}")
        if dm_dtheta is not None:
            if torch.any(torch.isnan(dm_dtheta)):
                raise ValueError("NaN detected in dm_dtheta")
            if torch.any(torch.isinf(dm_dtheta)):
                raise ValueError("Inf detected in dm_dtheta")

        # Implicit differentiation solve: (B, C, P)
        dT_dtheta = torch.linalg.lstsq(dR_dc, -dR_dtheta, driver="gelsd").solution

        # Chain rule
        grad_rig_parameters = torch.matmul(
            grad_output.unsqueeze(1), dT_dtheta
        ).squeeze(1)

        return grad_rig_parameters, None, None, None
