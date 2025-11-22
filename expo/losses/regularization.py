from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def sobolev_penalty(
    model: Optional[nn.Module],
    mean_pred: torch.Tensor,
    times: torch.Tensor,
    doses: torch.Tensor,
    lambda_first: float,
    lambda_second: float,
) -> torch.Tensor:
    """Sobolev-style smoothness penalty.

    For simplicity we treat `mean_pred` as already depending on (t, d) and
    approximate derivatives with respect to scalar inputs by autograd.
    """
    if lambda_first == 0.0 and lambda_second == 0.0:
        return mean_pred.new_tensor(0.0, requires_grad=True)

    times = times.clone().detach().requires_grad_(True)
    doses = doses.clone().detach().requires_grad_(True)
    f = mean_pred

    grads = torch.autograd.grad(
        outputs=f,
        inputs=[times, doses],
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    dt, dd = grads
    sob_first = mean_pred.new_tensor(0.0)
    sob_second = mean_pred.new_tensor(0.0)

    if lambda_first > 0.0:
        terms = []
        if dt is not None:
            terms.append(dt**2)
        if dd is not None:
            terms.append(dd**2)
        if terms:
            sob_first = torch.stack(terms).mean()

    if lambda_second > 0.0:
        d2_terms = []
        if dt is not None:
            d2t = torch.autograd.grad(
                outputs=dt,
                inputs=times,
                grad_outputs=torch.ones_like(dt),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if d2t is not None:
                d2_terms.append(d2t**2)
        if dd is not None:
            d2d = torch.autograd.grad(
                outputs=dd,
                inputs=doses,
                grad_outputs=torch.ones_like(dd),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if d2d is not None:
                d2_terms.append(d2d**2)
        if d2_terms:
            sob_second = torch.stack(d2_terms).mean()

    return lambda_first * sob_first + lambda_second * sob_second


def dose_monotonicity_penalty(
    mean_pred: torch.Tensor,
    doses: torch.Tensor,
    group_ids: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Encourage non-decreasing response magnitude with dose within groups."""
    device = mean_pred.device
    uniq_groups = group_ids.unique()
    penalties = []

    mag = mean_pred.abs().mean(dim=-1)

    for g in uniq_groups:
        mask = group_ids == g
        if mask.sum() < 2:
            continue
        g_doses = doses[mask]
        g_mag = mag[mask]

        sorted_idx = torch.argsort(g_doses)
        g_mag = g_mag[sorted_idx]

        diff = g_mag[1:] - g_mag[:-1]
        viol = (margin - diff).clamp_min(0.0)
        penalties.append(viol.mean())

    if not penalties:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(penalties).mean()
