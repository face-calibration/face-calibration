"""Simon-Says rig calibration solver.

Given a rig, target geometries for a set of known expressions, and the
corresponding control values, this solver optimizes the rig's internal
parameters so that ``R(c_k; theta) ~= v_k`` for each expression k.

This corresponds to Equation 1 in the paper:

    min_theta  sum_k || R(c_k; theta) - v_k ||^2 + reg * || theta - theta_0 ||^2
"""

import logging

import torch

logger = logging.getLogger(__name__)


class RigCalibrationSolver:
    """Simon-Says rig calibration.

    Optimizes internal rig parameters (e.g. FLAME shapedirs) so that
    the rig produces geometry matching captured targets when driven by
    known control values.

    Args:
        rig: a differentiable rig module.
        get_params: callable that returns the current parameter tensor
            from the rig (e.g. ``lambda r: r.flame.shapedirs``).
        set_params: callable that sets the parameter tensor on the rig
            (e.g. ``lambda r, p: setattr(r.flame, 'shapedirs', p)``).
        output_key: key to extract geometry from the rig output dict.
    """

    def __init__(self, rig, get_params, set_params, output_key="face"):
        self.rig = rig
        self.get_params = get_params
        self.set_params = set_params
        self.output_key = output_key

    def calibrate(
        self,
        targets,
        controls,
        masks=None,
        iterations=100,
        solver="lbfgs",
        data_weight=100.0,
        reg_weight=5.0,
        lr=1e-2,
    ):
        """Run the calibration.

        Args:
            targets: (N, V, 3) target geometry for each expression.
            controls: (N, num_controls) known control values.
            masks: optional (N, V) per-vertex mask (1 = use, 0 = ignore).
            iterations: number of optimization steps.
            solver: ``'lbfgs'`` or ``'adam'``.
            data_weight: multiplier on the data (vertex) loss.
            reg_weight: multiplier on the regularization loss.
            lr: learning rate (for Adam only).

        Returns:
            Dictionary with keys:

            - ``'params'``: optimized parameter tensor.
            - ``'loss'``: final combined loss value.
        """
        if not isinstance(targets, torch.Tensor):
            targets = torch.from_numpy(targets).float()
        if not isinstance(controls, torch.Tensor):
            controls = torch.from_numpy(controls).float()

        # Clone current params as the starting point
        original_params = self.get_params(self.rig).detach().clone()
        optimized_params = original_params.clone()
        optimized_params.requires_grad = True

        if masks is None:
            masks = torch.ones(targets.shape[0], targets.shape[1], dtype=torch.float32)
        elif not isinstance(masks, torch.Tensor):
            masks = torch.tensor(masks, dtype=torch.float32)

        iteration = [0]

        def eval_loss():
            if solver == "lbfgs":
                opt.zero_grad()

            self.set_params(self.rig, optimized_params)

            vertices = self.rig(controls.detach())
            if isinstance(vertices, dict):
                vertices = vertices[self.output_key]

            # Geometry matching loss
            diff = (vertices - targets) * 100  # scale up for numerical stability
            verts_loss = (diff * diff * masks.unsqueeze(-1)).mean() * data_weight

            # Regularization: penalize drift from initial parameters
            reg_loss = (original_params - optimized_params).square().sum() * reg_weight

            loss = verts_loss + reg_loss

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Iter %d: verts_loss=%.4g  reg_loss=%.4g  total=%.4g",
                    iteration[0], verts_loss.item(), reg_loss.item(), loss.item(),
                )

            loss.backward()
            iteration[0] += 1
            return loss

        if solver == "lbfgs":
            opt = torch.optim.LBFGS(
                [optimized_params], tolerance_grad=1e-15, tolerance_change=1e-15
            )
            for _ in range(iterations):
                opt.step(eval_loss)
        elif solver == "adam":
            opt = torch.optim.Adam([optimized_params], lr=lr)
            for _ in range(iterations):
                opt.zero_grad()
                loss = eval_loss()
                opt.step()
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Apply final params
        self.set_params(self.rig, optimized_params)

        final_loss = eval_loss().item()
        return {
            "params": optimized_params.detach().clone(),
            "loss": final_loss,
        }
