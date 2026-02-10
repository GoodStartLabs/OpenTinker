"""Diplomacy environment module for OpenTinker.

This module provides a Diplomacy game environment for LLM RL training,
ported from rl_envs/tinker_diplomacy with all external dependencies
(tinker, tinker-cookbook, chz) replaced by OpenTinker's native pipeline.

Usage:
    from opentinker.environment.diplomacy import DiplomacyGame

    game = DiplomacyGame(power="FRANCE")
    obs = game.reset()
    result = game.step('<tool_call>{"name": "list_units", "args": {}}</tool_call>')
"""

from opentinker.environment.diplomacy.diplomacy_game import DiplomacyGame

__all__ = ["DiplomacyGame"]
