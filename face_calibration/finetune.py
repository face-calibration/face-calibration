"""Multi-stage tracker fine-tuning via implicit differentiation.

This module implements the 4-stage optimization strategy from Section 6
of the paper for fine-tuning a tracker's internal rig parameters so that
the tracker produces more semantically meaningful animation controls.

The pipeline uses :class:`TrackerFunctionSeparate` to implicitly
differentiate through the tracker, enabling gradient-based optimization
of rig parameters even when the tracker itself is not differentiable.

Stage overview:

1. **Controls loss**: Optimize rig parameters so tracked controls match
   ground-truth controls (gamma_1 term).
2. **Geometry loss**: Optimize rig parameters so tracked controls produce
   geometry matching the ground truth (gamma_2 term).
3. **Heaviside-masked controls**: Minimize contributions of spurious
   (tweaker) controls by masking the controls loss.
4. **Per-control meta-iteration**: Iteratively find and optimize the
   worst-performing control dimension.

All stages regularize ``theta_T`` toward its initial value ``theta_0``
via a ``gamma_eps`` term.
"""

import logging
from typing import List, Optional

import torch
from tqdm import tqdm

from .implicit_diff import TrackerFunctionSeparate

logger = logging.getLogger(__name__)


class RigFineTuner:
    """Multi-stage rig fine-tuning for semantic tracking.

    Args:
        rig: the full differentiable rig.
        targets: (N, V, 3) target geometry for each expression.
        true_controls: (N, C) ground-truth control values.
        output_key: key to extract geometry from rig output dict.
        scale: geometry scale factor for tracker loss computation.
    """

    def __init__(
        self,
        rig,
        targets,
        true_controls,
        *,
        output_key="face",
        scale=1.0,
    ):
        self.rig = rig
        self.targets = targets.detach().clone()
        self.true_controls = true_controls.detach().clone()
        self.output_key = output_key
        self.scale = scale
        self.num_expressions = targets.shape[0]

        # Heaviside mask: True where a control is active in the ground truth
        self.H = self.true_controls != 0  # (N, C)

    def _extract_geometry(self, rig_output):
        if isinstance(rig_output, dict):
            return rig_output[self.output_key]
        return rig_output

    def _run_tracker_for_expression(self, rig, rig_parameters, target, num_controls,
                                    max_iter=50, l1_reg=1e-4):
        """Run a simple L-BFGS tracker for one expression."""
        controls = torch.zeros(num_controls, dtype=target.dtype)
        controls.requires_grad_(True)
        theta = rig_parameters.detach().clone()

        opt = torch.optim.LBFGS(
            [controls], tolerance_grad=1e-8, tolerance_change=1e-8, max_iter=max_iter
        )

        def eval_loss():
            opt.zero_grad()
            verts = self._extract_geometry(rig(controls, rig_parameters=theta))
            loss = ((verts * self.scale - target * self.scale) ** 2).mean()
            loss = loss + controls.abs().mean() * l1_reg
            loss.backward()
            return loss

        with torch.enable_grad():
            opt.step(eval_loss)

        return controls.detach().clone()

    def _run_tracker_all(self, rig_parameters, max_iter=50, l1_reg=1e-4):
        """Run the tracker on all expressions."""
        results = []
        for i in range(self.num_expressions):
            c = self._run_tracker_for_expression(
                self.rig, rig_parameters, self.targets[i],
                self.rig.num_controls, max_iter=max_iter, l1_reg=l1_reg,
            )
            results.append(c)
        return torch.stack(results)

    def stage1(self, rig_parameters, lr=5e-2, iterations=100, gamma_1=1e3, gamma_eps=1e0):
        """Stage 1: Controls-space loss.

        Optimize ``theta_T`` to minimize ``||T(I; theta_T) - c_k||^2``
        (the gamma_1 term) using implicit differentiation.

        Args:
            rig_parameters: initial flat parameter vector.
            lr: learning rate.
            iterations: number of SGD steps.
            gamma_1: weight on controls loss.
            gamma_eps: weight on regularization toward initial params.

        Returns:
            Optimized rig_parameters tensor.
        """
        return self._optimize_stage(
            rig_parameters, lr=lr, iterations=iterations,
            gamma_1=gamma_1, gamma_2=0, gamma_eps=gamma_eps,
            use_heaviside=False, stage_name="Stage 1",
        )

    def stage2(self, rig_parameters, lr=5e-2, iterations=100, gamma_2=1e-1, gamma_eps=1e0):
        """Stage 2: Geometry-space loss.

        Optimize ``theta_T`` to minimize ``||R(T(I; theta_T); theta_R) - v_k||^2``
        (the gamma_2 term).
        """
        return self._optimize_stage(
            rig_parameters, lr=lr, iterations=iterations,
            gamma_1=0, gamma_2=gamma_2, gamma_eps=gamma_eps,
            use_heaviside=False, stage_name="Stage 2",
        )

    def stage3(self, rig_parameters, lr=1e-2, iterations=500, gamma_1_H=1e3, gamma_eps=1e-4):
        """Stage 3: Heaviside-masked controls loss.

        Like Stage 1, but only penalizes controls that *should* be active
        according to the Heaviside mask (zero out spurious controls).
        """
        return self._optimize_stage(
            rig_parameters, lr=lr, iterations=iterations,
            gamma_1=gamma_1_H, gamma_2=0, gamma_eps=gamma_eps,
            use_heaviside=True, stage_name="Stage 3",
        )

    def stage4(self, rig_parameters, lr=5e-3, iterations=20,
               num_meta_iters=30, gamma_1_H=1e3, gamma_eps=1e-3,
               threshold=0.25):
        """Stage 4: Per-control meta-iteration.

        Iteratively identifies the worst-performing control dimension and
        optimizes rig parameters to improve it.

        Args:
            threshold: minimum control value to consider an expression
                "relevant" for a given control.
        """
        theta = rig_parameters.detach().clone()
        theta.requires_grad_(True)
        theta_0 = theta.detach().clone()
        modified_controls = set()
        best_theta = theta.detach().clone()
        best_total_error = float("inf")

        for meta_iter in range(num_meta_iters):
            # Track all expressions with current parameters
            tracked = self._run_tracker_all(theta.detach())

            # Find worst control (not already modified)
            errors = (tracked - self.true_controls).abs()
            mean_errors = errors.mean(dim=0)  # (C,)
            for idx in modified_controls:
                mean_errors[idx] = -1  # skip already-modified controls

            worst_ctrl = mean_errors.argmax().item()
            if mean_errors[worst_ctrl] <= 0:
                break
            modified_controls.add(worst_ctrl)

            logger.info(
                "Stage 4 meta-iter %d: worst control=%d, error=%.4g",
                meta_iter, worst_ctrl, mean_errors[worst_ctrl].item(),
            )

            # Find relevant expressions for this control
            relevant_mask = self.true_controls[:, worst_ctrl].abs() >= threshold
            if not relevant_mask.any():
                continue

            # Optimize for this control
            theta_opt = theta.detach().clone()
            theta_opt.requires_grad_(True)
            opt = torch.optim.SGD([theta_opt], lr=lr)

            for step in range(iterations):
                opt.zero_grad()
                loss = torch.tensor(0.0)

                for expr_idx in relevant_mask.nonzero(as_tuple=True)[0]:
                    c = self._run_tracker_for_expression(
                        self.rig, theta_opt.detach(), self.targets[expr_idx],
                        self.rig.num_controls,
                    )
                    c_with_grad = TrackerFunctionSeparate.apply(
                        theta_opt, c.unsqueeze(0), self.rig, None,
                    ).squeeze(0)

                    # Heaviside-masked controls loss
                    mask = self.H[expr_idx]
                    ctrl_diff = (c_with_grad - self.true_controls[expr_idx])
                    loss = loss + (ctrl_diff[mask] ** 2).mean() * gamma_1_H

                # Regularization
                loss = loss + ((theta_opt - theta_0) ** 2).mean() * gamma_eps
                loss.backward()
                opt.step()

            # Check if this improved overall performance
            test_tracked = self._run_tracker_all(theta_opt.detach())
            total_error = (test_tracked - self.true_controls).abs().mean().item()

            if total_error < best_total_error:
                best_total_error = total_error
                best_theta = theta_opt.detach().clone()
                theta = theta_opt

            logger.info(
                "Stage 4 meta-iter %d done: total_error=%.4g (best=%.4g)",
                meta_iter, total_error, best_total_error,
            )

        return best_theta.detach()

    def _optimize_stage(
        self, rig_parameters, *,
        lr, iterations, gamma_1, gamma_2, gamma_eps,
        use_heaviside, stage_name,
    ):
        """Generic optimization loop for stages 1-3."""
        theta = rig_parameters.detach().clone()
        theta.requires_grad_(True)
        theta_0 = theta.detach().clone()

        opt = torch.optim.SGD([theta], lr=lr)
        best_theta = theta.detach().clone()
        best_error = float("inf")

        for step in tqdm(range(iterations), desc=stage_name):
            opt.zero_grad()
            loss = torch.tensor(0.0)

            # Track all expressions
            tracked_controls = self._run_tracker_all(theta.detach())

            for expr_idx in range(self.num_expressions):
                c = tracked_controls[[expr_idx]]  # (1, C)
                c_with_grad = TrackerFunctionSeparate.apply(
                    theta, c, self.rig, None,
                )  # (1, C) with gradients for theta

                # Controls loss (gamma_1)
                if gamma_1 > 0:
                    true_c = self.true_controls[expr_idx].unsqueeze(0)
                    diff = c_with_grad - true_c
                    if use_heaviside:
                        mask = self.H[expr_idx].unsqueeze(0)
                        diff = diff * mask
                    loss = loss + (diff ** 2).mean() * gamma_1

                # Geometry loss (gamma_2)
                if gamma_2 > 0:
                    verts = self._extract_geometry(
                        self.rig(c_with_grad, rig_parameters=theta)
                    )
                    target = self.targets[expr_idx]
                    if target.dim() == 2:
                        target = target.unsqueeze(0)
                    geo_diff = verts * self.scale - target * self.scale
                    loss = loss + (geo_diff ** 2).mean() * gamma_2

            # Regularization
            loss = loss + ((theta - theta_0) ** 2).mean() * gamma_eps

            loss.backward()
            opt.step()

            # Track best via full controls error
            with torch.no_grad():
                ctrl_error = (tracked_controls - self.true_controls).abs().mean().item()
                if ctrl_error < best_error:
                    best_error = ctrl_error
                    best_theta = theta.detach().clone()

            if step % 20 == 0:
                logger.info(
                    "%s step %d: loss=%.4g  ctrl_error=%.4g  best=%.4g",
                    stage_name, step, loss.item(), ctrl_error, best_error,
                )

        return best_theta.detach()

    def run_all(self, rig_parameters=None):
        """Execute all 4 stages sequentially with default parameters.

        Args:
            rig_parameters: initial flat parameter vector.
                If None, uses ``self.rig.rig_parameters``.

        Returns:
            Final optimized rig_parameters tensor.
        """
        if rig_parameters is None:
            rig_parameters = self.rig.rig_parameters

        logger.info("Starting multi-stage fine-tuning")
        theta = self.stage1(rig_parameters)
        logger.info("Stage 1 complete")
        theta = self.stage2(theta)
        logger.info("Stage 2 complete")
        theta = self.stage3(theta)
        logger.info("Stage 3 complete")
        theta = self.stage4(theta)
        logger.info("Stage 4 complete — all stages done")
        return theta
