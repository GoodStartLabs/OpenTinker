"""DiplomacyGame - AbstractGame implementation for Diplomacy RL training.

The agent plays as a single power against 6 LLM-controlled opponents.
Uses text-based tool calling (<tool_call> tags) for the agent and native
OpenAI function calling for opponents (via OpenRouter).

Each step() call processes one LLM response: tool calls are parsed from
the raw text, executed, and results returned as the next observation.
When finish_phase is called, opponents complete their turns, the game
adjudicates, and the next phase begins.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from diplomacy import Game

from opentinker.environment.base_game import AbstractGame, StepResult

from .prompts import get_system_prompt_for_power
from .rewards import (
    MALFORMED_TOOL_PENALTY,
    SEND_MESSAGE_PENALTY,
    SEND_MESSAGE_REWARD,
    compute_phase_reward,
)
from .tools import (
    ToolContext,
    execute_tool,
    get_openai_tools,
)
from .utils import (
    count_supply_centers,
    get_phase_prompt,
    is_game_over,
)

logger = logging.getLogger(__name__)

ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]


def _parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from <tool_call>...</tool_call> tags in LLM text output."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    results = []
    for match in matches:
        try:
            parsed = json.loads(match)
            name = parsed.get("name", "")
            args = parsed.get("args", {})
            if name:
                results.append({"name": name, "args": args})
        except json.JSONDecodeError:
            pass

    return results


class DiplomacyGame(AbstractGame):
    """Diplomacy game environment for OpenTinker RL training.

    The agent plays as a single power against 6 LLM-controlled opponents.
    Opponents are driven by an external model via OpenRouter with native
    function calling. The agent uses text-based tool calling.

    Attributes:
        power: The power the agent plays as (e.g. "FRANCE").
        max_phases: Maximum game phases before forced termination.
        max_turns_per_phase: Maximum LLM turns per phase before timeout.
        opponent_model: OpenRouter model ID for opponent inference.
    """

    def __init__(
        self,
        power: str = "FRANCE",
        max_phases: int = 30,
        max_turns_per_phase: int = 40,
        opponent_model: str = "x-ai/grok-4-fast",
    ):
        self.default_power = power
        self.power = power
        self.max_phases = max_phases
        self.max_turns_per_phase = max_turns_per_phase
        self.opponent_model = opponent_model

        # OpenRouter client for opponents (lazy init)
        self._openrouter_client = None
        self._openai_tools = get_openai_tools()

        # Game state (initialized on reset)
        self.game: Optional[Game] = None
        self.tool_ctx: Optional[ToolContext] = None
        self.phase_count = 0
        self.turn_in_phase = 0
        self.prev_centers: Dict[str, int] = {}
        self.starting_centers = 3

        # Opponent state
        self._opponent_histories: Dict[str, List[Dict]] = {}
        self._opponent_tool_ctxs: Dict[str, ToolContext] = {}
        self._finished_opponents: set = set()
        self._opponent_turns: Dict[str, int] = {}

        # Message queues for real-time exchange between powers
        self._message_queues: Dict[str, List] = {}

        # Other powers (set on reset)
        self._other_powers: List[str] = []

    # =========================================================================
    # Lazy client init
    # =========================================================================

    def _get_openrouter_client(self):
        """Lazy-initialize the OpenRouter client."""
        if self._openrouter_client is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                logger.warning(
                    "OPENROUTER_API_KEY not set - opponents will be disabled"
                )
                return None
            try:
                from openai import OpenAI

                self._openrouter_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
            except ImportError:
                logger.warning(
                    "openai package not installed - opponents will be disabled"
                )
                return None
        return self._openrouter_client

    # =========================================================================
    # AbstractGame: Required Methods
    # =========================================================================

    def reset(
        self,
        power: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Start a new Diplomacy game.

        Args:
            power: Which power the agent plays as. Defaults to constructor value.
            seed: Random seed for reproducibility.

        Returns:
            Initial board state observation.
        """
        if seed is not None:
            random.seed(seed)

        self.power = power or self.default_power

        # Create new game (starts at S1901M by default)
        self.game = Game()

        # Message queues
        self._message_queues = {p: [] for p in self.game.powers}

        # Agent tool context
        self.tool_ctx = ToolContext(
            game=self.game,
            power=self.power,
            message_queues=self._message_queues,
        )

        # Other powers & opponents
        self._other_powers = [p for p in self.game.powers if p != self.power]
        self._initialize_opponents()

        # Phase tracking
        self.phase_count = 0
        self.turn_in_phase = 0
        self.prev_centers = {
            p: count_supply_centers(self.game, p) for p in self.game.powers
        }
        self.starting_centers = count_supply_centers(self.game, self.power)

        logger.info(
            f"Game reset: playing as {self.power}, "
            f"phase {self.game.get_current_phase()}"
        )
        return get_phase_prompt(self.game, self.power)

    def step(self, action: str) -> StepResult:
        """Process one LLM response.

        Parses ``<tool_call>`` tags from the raw text, executes the
        corresponding tools, and returns the results. When ``finish_phase``
        is called, opponents complete their turns and the game advances.

        Args:
            action: Raw text output from the LLM.

        Returns:
            StepResult with observation, reward, done flag, and info.
        """
        self.turn_in_phase += 1

        # Parse tool calls from text
        tool_calls = _parse_tool_calls_from_text(action)

        if not tool_calls:
            return self._handle_no_tool_calls(action)

        # Execute tools
        results: List[Dict[str, Any]] = []
        step_reward = 0.0

        for tc in tool_calls:
            result = execute_tool(self.tool_ctx, tc["name"], tc["args"])
            results.append({"name": tc["name"], "result": result})

            # Track send_message rewards
            if tc["name"] == "send_message":
                if self.game.phase_type == "M":
                    step_reward += SEND_MESSAGE_REWARD
                else:
                    step_reward += SEND_MESSAGE_PENALTY

        # Check if finish_phase was called
        if self.tool_ctx.finished_phase:
            return self._handle_phase_completion(step_reward)

        # Step opponents once (interleaved execution for message exchange)
        self._step_all_opponents_once()

        # Check for phase timeout
        if self.turn_in_phase >= self.max_turns_per_phase:
            return self._handle_phase_timeout()

        # Format tool results + any pending messages
        observation = self._format_observation(results)

        return StepResult(
            observation=observation,
            reward=step_reward,
            done=False,
            info={
                "turn_in_phase": self.turn_in_phase,
                "phase": self.game.get_current_phase(),
            },
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt (text-based tool calling mode)."""
        return get_system_prompt_for_power(self.power)

    def get_initial_user_message(self) -> str:
        return (
            "The game of Diplomacy has begun. Analyze the board state and "
            "use your tools to negotiate, strategize, and submit orders."
        )

    # =========================================================================
    # AbstractGame: Optional / Data Generation Methods
    # =========================================================================

    def generate_initial_state(self) -> Dict[str, Any]:
        return {
            "power": random.choice(ALL_POWERS),
            "seed": random.randint(0, 1_000_000),
        }

    def get_user_message_with_state(
        self,
        power: Optional[str] = None,
        **kwargs,
    ) -> str:
        self.reset(power=power, **kwargs)
        return (
            f"{self.get_initial_user_message()}\n\n"
            f"{get_phase_prompt(self.game, self.power)}"
        )

    def get_interaction_name(self) -> str:
        return "diplomacy"

    def get_state(self) -> Dict[str, Any]:
        if self.game is None:
            return {}
        return {
            "phase": self.game.get_current_phase(),
            "power": self.power,
            "centers": count_supply_centers(self.game, self.power),
            "phase_count": self.phase_count,
            "turn_in_phase": self.turn_in_phase,
        }

    # =========================================================================
    # Internal: Step Handling
    # =========================================================================

    def _handle_no_tool_calls(self, action: str) -> StepResult:
        """Handle an LLM response that contains no valid tool calls."""
        content = action.lower()

        # Detect malformed tool call attempts
        tool_names = [
            "check_messages",
            "get_possible_orders",
            "submit_all_orders",
            "send_message",
            "finish_phase",
            "list_units",
            "get_game_state",
            "write_diary",
            "read_diary",
        ]
        attempted_tool = any(name in content for name in tool_names)
        has_malformed_tag = "tool_call" in content or "<tool" in content

        if attempted_tool or has_malformed_tag:
            observation = (
                "Tool format invalid. Use this exact format:\n"
                "<tool_call>\n"
                '{"name": "tool_name", "args": {"arg1": "value"}}\n'
                "</tool_call>\n\n"
                "Example to check messages:\n"
                "<tool_call>\n"
                '{"name": "check_messages", "args": {}}\n'
                "</tool_call>"
            )
            return StepResult(
                observation=observation,
                reward=MALFORMED_TOOL_PENALTY,
                done=False,
                info={"malformed_tool_call": 1},
            )

        # Check phase timeout
        if self.turn_in_phase >= self.max_turns_per_phase:
            return self._handle_phase_timeout()

        return StepResult(
            observation=(
                "You must use tools to play. Call check_messages, "
                "get_possible_orders, submit_all_orders, send_message, "
                "and finish_phase to complete your turn."
            ),
            reward=0.0,
            done=False,
            info={"no_tool_calls": 1},
        )

    def _handle_phase_completion(self, step_reward: float) -> StepResult:
        """Complete opponent turns, advance the game, compute reward."""
        phase = self.game.get_current_phase()
        logger.info(f"Phase {phase} completed by {self.power}")

        # Complete all remaining opponents
        self._complete_opponent_turns()

        # Store previous centers for reward calculation
        prev_centers = self.prev_centers.copy()

        # Advance game
        self.game.process()
        self.phase_count += 1
        new_phase = self.game.get_current_phase()

        # Update center counts
        self.prev_centers = {
            p: count_supply_centers(self.game, p) for p in self.game.powers
        }

        # Compute reward
        reward, metrics = compute_phase_reward(
            game=self.game,
            power=self.power,
            prev_centers=prev_centers,
            starting_centers=self.starting_centers,
            phase_count=self.phase_count,
        )
        reward += step_reward

        centers_now = self.prev_centers[self.power]
        centers_before = prev_centers.get(self.power, self.starting_centers)
        delta = centers_now - centers_before
        logger.info(
            f"Phase {phase} -> {new_phase}: reward={reward:.3f}, "
            f"centers={centers_now} ({'+' if delta >= 0 else ''}{delta}), "
            f"phase_count={self.phase_count}"
        )

        # Check if game is over
        done = is_game_over(self.game, self.max_phases, self.phase_count)

        if done:
            logger.info(
                f"Game over after {self.phase_count} phases. "
                f"Final centers: {centers_now}"
            )
            metrics["game/phases_completed"] = self.phase_count
            metrics["game/centers_final"] = centers_now
            return StepResult(
                observation=f"Game over. Final centers: {centers_now}",
                reward=reward,
                done=True,
                info=metrics,
            )

        # Setup next phase
        self._setup_next_phase()

        return StepResult(
            observation=get_phase_prompt(self.game, self.power),
            reward=reward,
            done=False,
            info=metrics,
        )

    def _handle_phase_timeout(self) -> StepResult:
        """Advance the game with whatever orders are set when turn limit is hit."""
        phase = self.game.get_current_phase()
        logger.warning(
            f"Phase {phase} timeout after {self.turn_in_phase} turns"
        )

        # Complete opponents
        self._complete_opponent_turns()

        prev_centers = self.prev_centers.copy()

        self.game.process()
        self.phase_count += 1
        new_phase = self.game.get_current_phase()

        self.prev_centers = {
            p: count_supply_centers(self.game, p) for p in self.game.powers
        }

        reward, metrics = compute_phase_reward(
            game=self.game,
            power=self.power,
            prev_centers=prev_centers,
            starting_centers=self.starting_centers,
            phase_count=self.phase_count,
        )
        metrics["phase_timeout"] = 1

        done = is_game_over(self.game, self.max_phases, self.phase_count)

        if done:
            metrics["game/phases_completed"] = self.phase_count
            metrics["game/centers_final"] = self.prev_centers[self.power]
            return StepResult(
                observation=(
                    f"Phase timeout. Game over. "
                    f"Final centers: {self.prev_centers[self.power]}"
                ),
                reward=reward,
                done=True,
                info=metrics,
            )

        self._setup_next_phase()

        return StepResult(
            observation=(
                f"Phase timeout. Advancing to next phase.\n\n"
                f"{get_phase_prompt(self.game, self.power)}"
            ),
            reward=reward,
            done=False,
            info=metrics,
        )

    # =========================================================================
    # Internal: Phase Transitions
    # =========================================================================

    def _setup_next_phase(self):
        """Reset per-phase state for the next phase."""
        self.turn_in_phase = 0
        self.tool_ctx.finished_phase = False
        self.tool_ctx.message_counts = {}
        self._finished_opponents = set()
        self._opponent_turns = {p: 0 for p in self._other_powers}

        # Reset message queues
        self._message_queues = {p: [] for p in self.game.powers}
        self.tool_ctx.message_queues = self._message_queues

        # Fresh opponent contexts for the new phase
        for p in self._other_powers:
            sys_prompt = get_system_prompt_for_power(p, native_fc=True)
            self._opponent_histories[p] = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": get_phase_prompt(self.game, p)},
            ]
            self._opponent_tool_ctxs[p].finished_phase = False
            self._opponent_tool_ctxs[p].message_counts = {}
            self._opponent_tool_ctxs[p].message_queues = self._message_queues

    # =========================================================================
    # Internal: Observation Formatting
    # =========================================================================

    def _format_observation(self, tool_results: List[Dict]) -> str:
        """Format tool results and pending messages as an observation string."""
        parts = []

        for tr in tool_results:
            result_str = json.dumps(tr["result"], default=str, indent=2)
            parts.append(f"[{tr['name']}] {result_str}")

        # Inject any pending messages sent by opponents
        pending = self._message_queues.get(self.power, [])
        if pending:
            pending.sort(key=lambda m: m["timestamp"])
            msg_lines = []
            for msg in pending:
                prefix = "[GLOBAL] " if msg.get("global") else ""
                msg_lines.append(f"{prefix}[{msg['sender']}]: {msg['message']}")
            parts.append(
                "\nNew messages received:\n" + "\n".join(msg_lines)
            )
            pending.clear()

        return "\n\n".join(parts)

    # =========================================================================
    # Internal: Opponent Management
    # =========================================================================

    def _initialize_opponents(self):
        """Initialize opponent conversation histories and tool contexts."""
        self._opponent_histories = {}
        self._opponent_tool_ctxs = {}
        self._finished_opponents = set()
        self._opponent_turns = {p: 0 for p in self._other_powers}

        for p in self._other_powers:
            sys_prompt = get_system_prompt_for_power(p, native_fc=True)
            self._opponent_histories[p] = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": get_phase_prompt(self.game, p)},
            ]
            self._opponent_tool_ctxs[p] = ToolContext(
                game=self.game,
                power=p,
                message_queues=self._message_queues,
            )

    def _inject_pending_messages(
        self, power: str, messages: List[Dict]
    ) -> None:
        """Inject pending messages into an opponent's conversation."""
        pending = self._message_queues.get(power, [])
        if not pending:
            return

        pending.sort(key=lambda m: m["timestamp"])
        lines = []
        for msg in pending:
            prefix = "[GLOBAL] " if msg.get("global") else ""
            lines.append(f"{prefix}[{msg['sender']}]: {msg['message']}")

        messages.append({
            "role": "user",
            "content": "New messages received:\n" + "\n".join(lines),
        })
        pending.clear()

    def _step_opponent_once(self, power: str) -> bool:
        """Run one inference step for an opponent.

        Returns True if the opponent called ``finish_phase``.
        """
        client = self._get_openrouter_client()
        if client is None:
            return True  # No client → skip opponents

        messages = self._opponent_histories[power]
        tool_ctx = self._opponent_tool_ctxs[power]

        # Inject pending messages before LLM call
        self._inject_pending_messages(power, messages)

        try:
            response = client.chat.completions.create(
                model=self.opponent_model,
                messages=messages,
                max_tokens=1024,
                tools=self._openai_tools,
            )

            msg = response.choices[0].message

            # Build assistant message for conversation history
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            # Execute tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = (
                            json.loads(tc.function.arguments)
                            if tc.function.arguments
                            else {}
                        )
                    except json.JSONDecodeError:
                        args = {}

                    result = execute_tool(tool_ctx, name, args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str),
                    })

                if tool_ctx.finished_phase:
                    return True
            else:
                # No tool calls → prompt to finish
                messages.append({
                    "role": "user",
                    "content": (
                        "You must call finish_phase to end your turn. "
                        "Please call the finish_phase tool."
                    ),
                })
                if tool_ctx.finished_phase:
                    return True

            return False

        except Exception as e:
            logger.warning(f"Opponent {power} error: {e}")
            return False

    def _step_all_opponents_once(self):
        """Step every unfinished opponent once, in parallel."""
        unfinished = [
            p for p in self._other_powers
            if p not in self._finished_opponents
        ]
        if not unfinished:
            return

        # Enforce per-opponent turn limits
        to_step = []
        for p in unfinished:
            if self._opponent_turns.get(p, 0) >= self.max_turns_per_phase:
                self._finished_opponents.add(p)
            else:
                to_step.append(p)

        if not to_step:
            return

        with ThreadPoolExecutor(max_workers=min(len(to_step), 6)) as pool:
            futures = {
                pool.submit(self._step_opponent_once, p): p for p in to_step
            }
            for future in as_completed(futures):
                power = futures[future]
                self._opponent_turns[power] = (
                    self._opponent_turns.get(power, 0) + 1
                )
                try:
                    finished = future.result()
                    if finished:
                        self._finished_opponents.add(power)
                except Exception as e:
                    logger.warning(f"Opponent {power} error: {e}")
                    self._finished_opponents.add(power)

    def _complete_opponent_turns(self):
        """Run all opponents until they finish or hit their turn limit."""
        unfinished = [
            p for p in self._other_powers
            if p not in self._finished_opponents
        ]
        if not unfinished:
            return

        logger.info(f"Completing {len(unfinished)} remaining opponents")

        def _run_until_done(power: str):
            for _ in range(self.max_turns_per_phase):
                if self._opponent_tool_ctxs[power].finished_phase:
                    return
                turns = self._opponent_turns.get(power, 0)
                if turns >= self.max_turns_per_phase:
                    logger.warning(f"Opponent {power} hit turn limit")
                    return
                finished = self._step_opponent_once(power)
                self._opponent_turns[power] = (
                    self._opponent_turns.get(power, 0) + 1
                )
                if finished:
                    return
            logger.warning(f"Opponent {power} force-finished after max turns")

        with ThreadPoolExecutor(max_workers=min(len(unfinished), 6)) as pool:
            futures = [pool.submit(_run_until_done, p) for p in unfinished]
            for future in futures:
                try:
                    future.result(timeout=300)  # 5 min timeout per opponent
                except Exception as e:
                    logger.warning(f"Opponent completion error: {e}")

        self._finished_opponents.update(unfinished)
