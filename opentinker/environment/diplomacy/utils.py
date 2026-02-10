"""Utility functions for Diplomacy environment."""

from __future__ import annotations

from diplomacy import Game


def get_observation(game: Game, power: str) -> str:
    """Generate a text observation of the game state for a power."""
    phase = game.get_current_phase()
    phase_type = game.phase_type

    lines = []
    lines.append(f"=== Game State for {power} ===")
    lines.append(f"Phase: {phase} ({game.phase})")
    lines.append("")

    # My units
    my_units = game.get_units(power)
    lines.append(f"Your units ({len(my_units)}):")
    for unit in sorted(my_units):
        lines.append(f"  - {unit}")
    lines.append("")

    # My supply centers
    my_centers = game.get_centers(power)
    lines.append(f"Your supply centers ({len(my_centers)}):")
    lines.append(f"  {', '.join(sorted(my_centers))}")
    lines.append("")

    # Current orders (if any)
    my_orders = game.get_orders(power)
    if my_orders:
        lines.append("Current orders:")
        for order in my_orders:
            lines.append(f"  - {order}")
        lines.append("")

    # Phase-specific info
    if phase_type == "R":  # Retreat phase
        try:
            power_obj = game.get_power(power)
            if power_obj.retreats:
                lines.append("Units requiring retreat:")
                for unit, options in power_obj.retreats.items():
                    lines.append(f"  - {unit} can retreat to: {', '.join(options)}")
                lines.append("")
        except Exception:
            pass
    elif phase_type == "A":  # Adjustment phase
        try:
            power_obj = game.get_power(power)
            num_units = len(my_units)
            num_centers = len(my_centers)
            diff = num_centers - num_units

            if diff > 0:
                lines.append(f"You may BUILD {diff} unit(s).")
                lines.append(f"Home centers: {', '.join(sorted(power_obj.homes))}")
            elif diff < 0:
                lines.append(f"You must DISBAND {-diff} unit(s).")
            else:
                lines.append("No adjustments needed.")
            lines.append("")
        except Exception:
            pass

    # Other powers summary
    lines.append("Other powers:")
    for p in sorted(game.powers.keys()):
        if p != power:
            p_units = game.get_units(p)
            p_centers = game.get_centers(p)
            lines.append(f"  {p}: {len(p_units)} units, {len(p_centers)} centers")
    lines.append("")

    # All units on the board
    lines.append("All units on board:")
    all_units = game.get_units()
    for p in sorted(all_units.keys()):
        units = all_units[p]
        if units:
            lines.append(f"  {p}: {', '.join(sorted(units))}")

    return "\n".join(lines)


def get_phase_prompt(game: Game, power: str) -> str:
    """Generate the user prompt for a new phase with phase-specific guidance."""
    from .prompts import get_user_message

    observation = get_observation(game, power)
    phase_guidance = get_user_message(game)

    return f"""{phase_guidance}

{observation}

Use the available tools to:
1. Check messages from other powers
2. Send diplomatic messages if desired
3. Get possible orders for your units
4. Submit your orders
5. Call finish_phase when done"""


def count_supply_centers(game: Game, power: str) -> int:
    """Count the number of supply centers for a power."""
    return len(game.get_centers(power))


def count_units(game: Game, power: str) -> int:
    """Count the number of units for a power."""
    return len(game.get_units(power))


def is_eliminated(game: Game, power: str) -> bool:
    """Check if a power has been eliminated (0 supply centers)."""
    return count_supply_centers(game, power) == 0


def get_winner(game: Game) -> str | None:
    """Get the winner if someone has 18+ supply centers."""
    for power in game.powers:
        if count_supply_centers(game, power) >= 18:
            return power
    return None


def is_game_over(game: Game, max_phases: int, phase_count: int) -> bool:
    """Check if the game is over."""
    if phase_count >= max_phases:
        return True
    if get_winner(game) is not None:
        return True
    return False
