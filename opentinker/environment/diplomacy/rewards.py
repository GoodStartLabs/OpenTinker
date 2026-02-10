"""Reward functions for Diplomacy RL training."""

from __future__ import annotations

from typing import Any

from diplomacy import Game

from .utils import count_supply_centers, count_units

# Phase reward constants
CENTER_DELTA_REWARD = 1.0  # Per center gained/lost
RELATIVE_POSITION_REWARD = 0.2  # Per center above starting position

# Step reward constants
SEND_MESSAGE_REWARD = 0.02  # Per message during movement phase
SEND_MESSAGE_PENALTY = -0.05  # Per message during retreat/adjustment phases
MALFORMED_TOOL_PENALTY = -0.1  # Per malformed tool call


def compute_phase_reward(
    game: Game,
    power: str,
    prev_centers: dict[str, int],
    starting_centers: int = 3,
    phase_count: int = 0,
) -> tuple[float, dict[str, Any]]:
    """Compute the total reward for a completed phase."""
    current_centers = count_supply_centers(game, power)
    previous_centers = prev_centers.get(power, current_centers)
    delta = current_centers - previous_centers

    delta_reward = delta * CENTER_DELTA_REWARD
    centers_above_start = current_centers - starting_centers
    relative_reward = centers_above_start * RELATIVE_POSITION_REWARD
    total = delta_reward + relative_reward

    metrics = {
        "reward/phase_total": total,
        "reward/delta": delta_reward,
        "reward/relative": relative_reward,
        "centers": current_centers,
        "centers_delta": delta,
        "units": count_units(game, power),
        "phase": phase_count,
    }

    return total, metrics
