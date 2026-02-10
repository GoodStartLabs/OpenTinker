#!/usr/bin/env python3
"""Diplomacy Environment Server.

Starts a Diplomacy game server using the generic base_game_server.

Usage:
    python -m opentinker.environment.diplomacy.diplomacy_server
    python -m opentinker.environment.diplomacy.diplomacy_server --port 8093
"""

import os

# Diplomacy is a pure-logic game engine — no GPU needed.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Diplomacy Game Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8093, help="Server port")
    parser.add_argument(
        "--power",
        default="FRANCE",
        help="Default power for the agent (can be overridden per-reset)",
    )
    parser.add_argument(
        "--max_phases",
        type=int,
        default=30,
        help="Maximum phases per game",
    )
    parser.add_argument(
        "--max_turns_per_phase",
        type=int,
        default=40,
        help="Maximum LLM turns per phase before timeout",
    )
    parser.add_argument(
        "--opponent_model",
        default=os.getenv("DIPLOMACY_OPPONENT_MODEL", "x-ai/grok-4-fast"),
        help="OpenRouter model ID for opponent inference",
    )
    args = parser.parse_args()

    from opentinker.environment.base_game_server import run_game_server
    from opentinker.environment.diplomacy.diplomacy_game import DiplomacyGame

    print("\nDiplomacy Game Configuration:")
    print(f"  Power: {args.power}")
    print(f"  Max phases: {args.max_phases}")
    print(f"  Max turns/phase: {args.max_turns_per_phase}")
    print(f"  Opponent model: {args.opponent_model}")
    print(f"  OpenRouter API key: {'set' if os.getenv('OPENROUTER_API_KEY') else 'NOT SET'}")

    run_game_server(
        game_class=DiplomacyGame,
        host=args.host,
        port=args.port,
        stats_class=None,
        power=args.power,
        max_phases=args.max_phases,
        max_turns_per_phase=args.max_turns_per_phase,
        opponent_model=args.opponent_model,
    )


if __name__ == "__main__":
    main()
