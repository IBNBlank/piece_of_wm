"""Left-padded temporal windows and rollout history updates."""

from __future__ import annotations

import torch


def history_windows(
    sequence: torch.Tensor,
    valid_mask: torch.Tensor,
    history_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns a left-padded window ending at every sequence timestep.

    ``sequence`` is ``[B, T, ...]`` and the result is ``[B, T, H, ...]``.
    Padding is zero, and its validity is represented only by the returned mask.
    """
    _validate_sequence_mask(sequence, valid_mask)
    if history_len <= 0:
        raise ValueError("history_len must be positive.")
    time = sequence.shape[1]
    offsets = torch.arange(1 - history_len, 1, device=sequence.device)
    positions = torch.arange(time, device=sequence.device)[:, None] + offsets
    in_bounds = positions >= 0
    indices = positions.clamp_min(0)
    windows = sequence[:, indices]
    window_mask = valid_mask[:, indices] & in_bounds[None]
    padding = (~window_mask).reshape(
        *window_mask.shape, *([1] * (sequence.ndim - 2))
    )
    return windows.masked_fill(padding, 0), window_mask


def previous_history_windows(
    sequence: torch.Tensor,
    valid_mask: torch.Tensor,
    history_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns windows strictly before each timestep of ``sequence``.

    For an action sequence, output index ``t`` contains the actions preceding
    state ``t``. The output time dimension therefore equals the input length.
    """
    _validate_sequence_mask(sequence, valid_mask)
    if history_len <= 0:
        raise ValueError("history_len must be positive.")
    time = sequence.shape[1]
    offsets = torch.arange(-history_len, 0, device=sequence.device)
    positions = torch.arange(time, device=sequence.device)[:, None] + offsets
    in_bounds = positions >= 0
    indices = positions.clamp_min(0)
    windows = sequence[:, indices]
    window_mask = valid_mask[:, indices] & in_bounds[None]
    padding = (~window_mask).reshape(
        *window_mask.shape, *([1] * (sequence.ndim - 2))
    )
    return windows.masked_fill(padding, 0), window_mask


def append_history(
    history: torch.Tensor,
    valid_mask: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drops the oldest slot and appends one valid value without resizing."""
    _validate_sequence_mask(history, valid_mask)
    if value.shape != (history.shape[0], *history.shape[2:]):
        raise ValueError("value shape must match one history timestep.")
    updated = torch.cat((history[:, 1:], value[:, None]), dim=1)
    updated_mask = torch.cat(
        (valid_mask[:, 1:], torch.ones_like(valid_mask[:, :1])), dim=1
    )
    return updated, updated_mask


def _validate_sequence_mask(sequence: torch.Tensor, valid_mask: torch.Tensor) -> None:
    if sequence.ndim < 3:
        raise ValueError("sequence must have shape [batch, time, ...].")
    if valid_mask.shape != sequence.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape [batch, time].")
    if valid_mask.device != sequence.device:
        raise ValueError("sequence and valid_mask must be on the same device.")
