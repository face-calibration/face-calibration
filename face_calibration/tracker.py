"""Tracker implementations for rig inversion.

A tracker solves the inverse problem: given target geometry, find animation
controls that produce that geometry when evaluated in the rig.  These
trackers are used both for performance tracking and for the implicit
differentiation pipeline.

All trackers are rig-agnostic -- they work with any rig whose ``forward``
method satisfies ``forward(controls, rig_parameters=theta)``.
"""

import logging

import torch
from tqdm import tqdm

from .implicit_diff import TrackerFunction, TrackerFunctionSeparate

logger = logging.getLogger(__name__)


def _extract_geometry(rig_output, output_key=None):
    """Extract geometry tensor from a rig's output (tensor or dict)."""
    if output_key is not None:
        return rig_output[output_key]
    if isinstance(rig_output, dict):
        return next(iter(rig_output.values()))
    return rig_output


# ---------------------------------------------------------------------------
# L-BFGS Tracker (differentiable via implicit differentiation)
# ---------------------------------------------------------------------------

class LBFGSTracker(torch.nn.Module):
    """L-BFGS tracker that inverts a rig to find controls matching targets.

    Wraps :class:`TrackerFunction` so the backward pass computes
    ``dT/dtheta`` via implicit differentiation.

    Args:
        rig: differentiable rig module with ``forward(controls, rig_parameters=...)``.
        targets: (B, num_verts, 3) or (B, num_verts*3) target geometry.
        output_key: key to extract from dict rig output (e.g. ``'face'``).
            If ``None``, the rig output is used directly.
        scale: multiply geometry by this factor before computing loss
            (useful when geometry is in meters and differences are tiny).
        l1_reg: L1 regularization weight on controls.
        clamp: if ``True``, clamp final controls to ``[0, 1]``.
    """

    def __init__(
        self, rig, targets, *,
        initial_controls=None,
        output_key=None,
        scale=1.0,
        iterations=1,
        tolerance=1e-8,
        max_iter=30,
        l1_reg=1e-6,
        clamp=False,
        log=False,
    ):
        super().__init__()
        self.rig = rig
        self.iterations = iterations
        self.log = log
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.targets = targets.detach().clone()
        self.output_key = output_key
        self.scale = scale
        self.l1_reg = l1_reg
        self.clamp = clamp

        if initial_controls is None:
            self.initial_controls = torch.zeros(
                (targets.shape[0], rig.num_controls), dtype=targets.dtype
            )
        else:
            self.initial_controls = initial_controls.detach().clone()

        self.latest_loss = 0.0

    def forward(self, rig_parameters, dm_dtheta=None):
        return TrackerFunction.apply(rig_parameters, self.run, self.rig, dm_dtheta)

    def run(self, rig_parameters):
        """Run the tracker (not differentiable).

        Returns:
            controls: (B, num_controls)
        """
        controls = self.initial_controls.detach().clone()
        controls.requires_grad_(True)

        opt = torch.optim.LBFGS(
            [controls],
            tolerance_grad=self.tolerance,
            tolerance_change=self.tolerance,
            max_iter=self.max_iter,
        )
        theta = rig_parameters.detach().clone()

        def eval_loss():
            opt.zero_grad()
            verts = _extract_geometry(
                self.rig(controls, rig_parameters=theta), self.output_key
            )
            verts = verts * self.scale
            scaled_targets = self.targets * self.scale
            diff = verts - scaled_targets
            loss = diff.square().mean()
            loss = loss + controls.abs().mean() * self.l1_reg
            loss.backward()
            return loss

        with torch.enable_grad():
            self.latest_loss = eval_loss().item()
            if self.log:
                logger.info("tracker: initial loss: %.4g", self.latest_loss)
            for _ in range(self.iterations):
                opt.step(eval_loss)
                self.latest_loss = eval_loss().item()
                if self.log:
                    logger.info("tracker: loss: %.4g", self.latest_loss)

        if self.clamp:
            controls = torch.clamp(controls, 0, 1)
        return controls.detach().clone()


# ---------------------------------------------------------------------------
# L-BFGS Tracker Separate (per-expression, reduced parameter sets)
# ---------------------------------------------------------------------------

class LBFGSTrackerSeparate(LBFGSTracker):
    """Per-expression L-BFGS tracker with condensation masks.

    Runs the tracker once over all expressions, then computes implicit
    differentiation gradients separately for each expression using only
    the relevant subset of rig parameters (condensation masks).  This
    keeps memory manageable for rigs with many parameters.

    Args:
        rig: the full rig.
        targets: (B, ...) target geometry for each expression.
        sub_rigs: list of reduced rigs (one per expression) whose
            ``rig_parameters`` correspond to the condensed parameter subset.
        condensations: list of boolean masks ``(num_rig_params,)`` selecting
            the relevant parameters for each expression.
        H: optional Heaviside mask ``(B, num_controls)`` — if provided,
            controls outside the mask are zeroed after tracking.
    """

    def __init__(
        self, rig, targets, sub_rigs, condensations, *,
        H=None,
        initial_controls=None,
        output_key=None,
        scale=1.0,
        iterations=1,
        tolerance=1e-8,
        max_iter=30,
        l1_reg=1e-4,
        clamp=False,
        log=False,
    ):
        super().__init__(
            rig, targets,
            initial_controls=initial_controls,
            output_key=output_key,
            scale=scale,
            iterations=iterations,
            tolerance=tolerance,
            max_iter=max_iter,
            l1_reg=l1_reg,
            clamp=clamp,
            log=log,
        )
        self.sub_rigs = sub_rigs
        self.condensations = condensations
        self.H = H

    def run(self, rig_parameters):
        """Run per-expression optimization."""
        theta = rig_parameters.detach().clone()
        loss_all, reg_all = [], []
        result = []

        for expr in range(len(self.sub_rigs)):
            controls = self.initial_controls[expr].detach().clone()
            controls.requires_grad_(True)
            target = self.targets[expr]

            opt = torch.optim.LBFGS(
                [controls],
                tolerance_grad=self.tolerance,
                tolerance_change=self.tolerance,
                max_iter=self.max_iter,
            )

            def eval_loss():
                opt.zero_grad()
                verts = _extract_geometry(
                    self.rig(controls, rig_parameters=theta), self.output_key
                )
                verts = verts * self.scale
                scaled_target = target * self.scale
                diff = verts - scaled_target
                loss = diff.square().mean()
                loss = loss + controls.abs().mean() * self.l1_reg
                loss.backward()
                return loss

            with torch.enable_grad():
                for _ in range(self.iterations):
                    opt.step(eval_loss)
                final_loss = eval_loss().item()

            if self.log:
                logger.info("tracker expr %d: loss: %.4g", expr, final_loss)
            loss_all.append(final_loss)

            if self.clamp:
                controls = torch.clamp(controls, 0, 1)
            elif hasattr(self.rig, "controls_min"):
                controls = torch.clamp(
                    controls,
                    torch.tensor(self.rig.controls_min),
                    torch.tensor(self.rig.controls_max),
                )
            result.append(controls)

        result = torch.stack(result)
        if self.H is not None:
            result[~self.H] = 0

        self.latest_loss = {"loss": loss_all}
        return result.detach().clone()

    def forward(self, rig_parameters, dm_dthetas=None):
        """Run tracker then compute per-expression implicit derivatives."""
        controls = self.run(rig_parameters)
        c_i = []
        for expr in range(len(self.sub_rigs)):
            relevant_params = rig_parameters[self.condensations[expr]]
            relevant_rig = self.sub_rigs[expr]
            relevant_controls = controls[[expr]]
            dm = None if dm_dthetas is None else dm_dthetas[expr]
            c_i.append(
                TrackerFunctionSeparate.apply(
                    relevant_params, relevant_controls, relevant_rig, dm
                )
            )
        return torch.cat(c_i, dim=0)


# ---------------------------------------------------------------------------
# Performance Tracker (non-differentiable, for evaluation)
# ---------------------------------------------------------------------------

class LBFGSPerformanceTracker:
    """Non-differentiable L-BFGS tracker for per-frame performance tracking.

    Runs independent optimizations for each frame.  Not used in the
    implicit differentiation pipeline; intended for evaluation.

    Args:
        rig: rig module.
        targets: (num_frames, num_verts*3) target geometry per frame.
        output_key: key to extract from dict rig output.
        scale: geometry scale factor.
        l1_reg: L1 regularization on controls.
    """

    def __init__(
        self, rig, targets, *,
        initial_controls=None,
        output_key=None,
        scale=1.0,
        iterations=1,
        tolerance=1e-8,
        max_iter=30,
        l1_reg=1e-4,
        log=False,
    ):
        self.rig = rig
        self.iterations = iterations
        self.log = log
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.targets = targets.detach().clone()
        self.output_key = output_key
        self.scale = scale
        self.l1_reg = l1_reg

        if initial_controls is None:
            self.initial_controls = torch.zeros(
                (targets.shape[0], rig.num_controls), dtype=targets.dtype
            )
        else:
            ic = initial_controls.detach().clone()
            if ic.dim() == 1:
                ic = ic.unsqueeze(0)
            if ic.shape[0] == 1:
                ic = ic.expand(targets.shape[0], -1)
            self.initial_controls = ic

    def run(self, rig_parameters):
        """Track all frames.

        Returns:
            controls: (num_frames, num_controls)
        """
        result = []
        theta = rig_parameters.detach().clone()

        for frame in tqdm(range(self.targets.shape[0]), desc="Tracking"):
            controls = self.initial_controls[frame].detach().clone()
            controls.requires_grad_(True)

            opt = torch.optim.LBFGS(
                [controls],
                tolerance_grad=self.tolerance,
                tolerance_change=self.tolerance,
                max_iter=self.max_iter,
            )

            def eval_loss():
                opt.zero_grad()
                verts = _extract_geometry(
                    self.rig(controls, rig_parameters=theta), self.output_key
                )
                verts = (verts * self.scale).reshape(-1)
                target = (self.targets[frame] * self.scale).reshape(-1)
                loss = (verts - target).square().mean()
                loss = loss + controls.abs().mean() * self.l1_reg
                loss.backward()
                return loss

            with torch.enable_grad():
                for _ in range(self.iterations):
                    opt.step(eval_loss)

            result.append(controls.detach().clone())

        return torch.stack(result)
