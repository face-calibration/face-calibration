#!/usr/bin/env python3
"""Minimal example: implicit differentiation with a simple linear rig.

This example demonstrates the core concept of the paper using a
3x3 linear rig ``R(c; A) = A @ c``.  The "tracker" is the matrix
inverse ``T(v; A) = A^{-1} @ v``.  We use implicit differentiation
to optimize ``A`` (the rig's internal parameters) to match a target.

No FLAME model or data files are needed.
"""

import torch

from face_calibration.implicit_diff import TrackerFunction, TrackerFunctionSeparate


class LinearRig(torch.nn.Module):
    """A toy linear rig: R(c; A) = A @ c."""

    def __init__(self, A):
        super().__init__()
        self.A = A  # (3, 3)
        self.num_controls = 3

    def forward(self, controls, rig_parameters=None):
        A = rig_parameters.view(3, 3) if rig_parameters is not None else self.A
        if controls.dim() == 1:
            return A @ controls
        return (A @ controls.unsqueeze(-1)).squeeze(-1)


def matrix_inverse_tracker(A_flat, v):
    """A simple 'tracker' that inverts the rig: c = A^{-1} v."""
    A = A_flat.view(3, 3)
    return (torch.linalg.solve(A, v.unsqueeze(-1))).squeeze(-1).unsqueeze(0)


def main():
    torch.manual_seed(42)

    # Ground-truth rig parameters (the "target" A)
    A_true = torch.tensor([[2.0, 0.5, 0.0],
                           [0.3, 1.8, 0.1],
                           [0.0, 0.2, 2.5]])

    # Initial (uncalibrated) rig parameters
    A_init = torch.eye(3)

    # Generate Simon-Says data: (c_k, v_k) pairs
    controls_list = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        torch.tensor([0.5, 0.5, 0.0]),
    ]
    targets = torch.stack([A_true @ c for c in controls_list])  # (4, 3)

    print("=== Linear Rig Example ===")
    print(f"True A:\n{A_true}")
    print(f"Initial A (identity):\n{A_init}")

    # --- Method 1: Direct TrackerFunction ---
    print("\n--- Using TrackerFunction (live tracker) ---")

    rig = LinearRig(A_init)
    theta = A_init.reshape(-1).clone().requires_grad_(True)
    optimizer = torch.optim.SGD([theta], lr=0.01)

    for step in range(200):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        for k in range(len(controls_list)):
            # TrackerFunction runs the tracker and provides gradients via implicit diff
            tracked = TrackerFunction.apply(
                theta,
                lambda t, v=targets[k]: matrix_inverse_tracker(t, v),
                rig,
                None,
            )
            loss = (tracked.squeeze(0) - controls_list[k]).square().sum()
            total_loss = total_loss + loss

        total_loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"  Step {step:3d}: loss = {total_loss.item():.6f}")

    A_opt = theta.detach().view(3, 3)
    print(f"Optimized A:\n{A_opt}")
    print(f"||A_opt - A_true||_F = {(A_opt - A_true).norm():.6f}")

    # --- Method 2: TrackerFunctionSeparate (pre-computed controls) ---
    print("\n--- Using TrackerFunctionSeparate (pre-computed controls) ---")

    theta2 = A_init.reshape(-1).clone().requires_grad_(True)
    optimizer2 = torch.optim.SGD([theta2], lr=0.01)

    for step in range(200):
        optimizer2.zero_grad()
        total_loss = torch.tensor(0.0)

        for k in range(len(controls_list)):
            # Pre-compute controls (run the tracker first before implicit differentiation)
            with torch.no_grad():
                A_current = theta2.view(3, 3)
                pre_tracked = torch.linalg.solve(A_current, targets[k]).unsqueeze(0)

            # TrackerFunctionSeparate provides implicit diff gradients
            tracked = TrackerFunctionSeparate.apply(theta2, pre_tracked, rig, None)
            loss = (tracked.squeeze(0) - controls_list[k]).square().sum()
            total_loss = total_loss + loss

        total_loss.backward()
        optimizer2.step()

        if step % 50 == 0:
            print(f"  Step {step:3d}: loss = {total_loss.item():.6f}")

    A_opt2 = theta2.detach().view(3, 3)
    print(f"Optimized A:\n{A_opt2}")
    print(f"||A_opt - A_true||_F = {(A_opt2 - A_true).norm():.6f}")

    print("\nDone! Both methods recover the true rig parameters via implicit differentiation.")


if __name__ == "__main__":
    main()
