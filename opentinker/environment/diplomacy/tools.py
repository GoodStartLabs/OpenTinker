"""Tool implementations for Diplomacy environment.

Ported from rl_envs/tinker_diplomacy with tinker dependencies removed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

from diplomacy import Game, Message
import logging

logger = logging.getLogger(__name__)

MESSAGE_LIMIT = 5


@dataclass
class ToolContext:
    """Context for tool execution."""

    game: Game
    power: str
    finished_phase: bool = False
    log_dir: Optional[Path] = None
    message_counts: Dict[str, int] = field(
        default_factory=dict
    )  # recipient -> count sent this phase
    message_queues: Optional[Dict[str, List]] = None


# =============================================================================
# Tool Implementations
# =============================================================================


def list_units(ctx: ToolContext) -> dict[str, Any]:
    """Returns a list of all units on the board."""
    game = ctx.game
    all_units_by_power = game.get_units()

    result = []
    for power_name, units in all_units_by_power.items():
        for unit_str in units:
            clean_unit = unit_str.lstrip("*")
            unit_type = clean_unit[0]
            province = clean_unit[2:]
            result.append(
                {"province": province, "type": unit_type, "owner": power_name}
            )

    return {"units": result, "count": len(result)}


def get_possible_orders(
    ctx: ToolContext, unit_location: str | None = None
) -> dict[str, Any]:
    """Get all possible orders for units controlled by this agent."""
    game = ctx.game
    power_name = ctx.power

    all_possible = game.get_all_possible_orders()

    my_units = game.get_units(power_name)
    my_locations = {
        unit.lstrip("*")[2:] for unit in my_units
    }

    my_possible = {}
    for location in my_locations:
        if location in all_possible:
            my_possible[location] = all_possible[location]

    if unit_location:
        unit_location = unit_location.upper()
        if unit_location.startswith(("A ", "F ")) and len(unit_location) > 2:
            unit_location = unit_location[2:].strip()

        if unit_location in my_possible:
            return {"location": unit_location, "orders": my_possible[unit_location]}
        else:
            return {
                "location": unit_location,
                "orders": [],
                "error": f"No unit controlled by {power_name} at {unit_location}",
            }

    return {"power": power_name, "possible_orders": my_possible}


def get_game_state(ctx: ToolContext) -> dict[str, Any]:
    """Get comprehensive information about the current game state."""
    game = ctx.game
    power_name = ctx.power

    state = {
        "phase": game.get_current_phase(),
        "phase_long": game.phase,
        "phase_type": game.phase_type,
        "power": power_name,
        "my_units": game.get_units(power_name),
        "my_centers": game.get_centers(power_name),
        "my_current_orders": game.get_orders(power_name),
        "all_units": game.get_units(),
        "all_centers": {p: game.get_centers(p) for p in game.powers},
        "game_id": game.game_id,
        "map_name": game.map_name,
    }

    try:
        if hasattr(game, "get_power"):
            power = game.get_power(power_name)
            if game.phase_type == "R":
                state["retreats"] = power.retreats
                state["dislodged_units"] = list(power.retreats.keys())
            elif game.phase_type == "A":
                state["builds"] = power.adjust
                state["homes"] = power.homes
    except Exception:
        pass

    return state


def _find_unit_at_location(game: Game, location: str) -> str:
    """Find unit string at a given location."""
    location = location.upper()
    all_units = game.get_units()
    for power_units in all_units.values():
        for unit in power_units:
            if unit.lstrip("*")[2:] == location:
                return unit
    raise ValueError(f"No unit found at {location}")


def _construct_order_string(game: Game, power_name: str, order: dict) -> str:
    """Construct a Diplomacy order string from an order dict."""
    unit_location = order["unit_location"].upper()
    action = order["action"].upper()

    if unit_location.startswith(("A ", "F ")) and len(unit_location) > 2:
        unit_location = unit_location[2:].strip()

    if action not in ["BUILD", "WAIVE"]:
        my_units = game.get_units(power_name)
        unit_at_location = None
        for unit in my_units:
            location = unit.lstrip("*")[2:]
            if location == unit_location:
                unit_at_location = unit.lstrip("*")
                break

        if not unit_at_location:
            if " " in unit_location:
                raise ValueError(
                    f"No unit controlled by {power_name} found at {unit_location}. "
                    f"Did you accidentally include unit type? Use location only (e.g., 'ION' not 'F ION')"
                )
            raise ValueError(
                f"No unit controlled by {power_name} found at {unit_location}"
            )
    else:
        unit_at_location = None

    if action == "HOLD":
        return f"{unit_at_location} H"
    elif action == "MOVE":
        if "destination" not in order:
            raise ValueError("MOVE requires destination")
        return f"{unit_at_location} - {order['destination'].upper()}"
    elif action == "SUPPORT_HOLD":
        if "support_unit_location" not in order:
            raise ValueError("SUPPORT_HOLD requires support_unit_location")
        support_unit = _find_unit_at_location(game, order["support_unit_location"])
        return f"{unit_at_location} S {support_unit}"
    elif action == "SUPPORT_MOVE":
        if "support_unit_location" not in order or "support_destination" not in order:
            raise ValueError(
                "SUPPORT_MOVE requires support_unit_location and support_destination"
            )
        support_unit = _find_unit_at_location(game, order["support_unit_location"])
        return f"{unit_at_location} S {support_unit} - {order['support_destination'].upper()}"
    elif action == "CONVOY":
        if "convoy_unit_location" not in order or "convoy_destination" not in order:
            raise ValueError(
                "CONVOY requires convoy_unit_location and convoy_destination"
            )
        convoy_unit = _find_unit_at_location(game, order["convoy_unit_location"])
        return f"{unit_at_location} C {convoy_unit} - {order['convoy_destination'].upper()}"
    elif action == "MOVE_VIA_CONVOY":
        if "destination" not in order:
            raise ValueError("MOVE_VIA_CONVOY requires destination")
        return f"{unit_at_location} - {order['destination'].upper()} VIA"
    elif action == "RETREAT":
        if "destination" not in order:
            raise ValueError("RETREAT requires destination")
        return f"{unit_at_location} R {order['destination'].upper()}"
    elif action == "BUILD":
        unit_type = order.get("unit_type")
        if unit_type:
            return f"{unit_type} {unit_location} B"
        possible_orders = game.get_all_possible_orders()
        if unit_location in possible_orders:
            build_orders = [
                o for o in possible_orders[unit_location] if o.endswith(" B")
            ]
            if not build_orders:
                raise ValueError(f"No build orders available at {unit_location}")
            if len(build_orders) == 1:
                return build_orders[0]
            raise ValueError(
                f"Ambiguous build at {unit_location}. Specify unit_type='A' or 'F'. Options: {build_orders}"
            )
        raise ValueError(
            f"Cannot determine unit type for build at {unit_location}. Specify unit_type='A' or 'F'."
        )
    elif action == "DISBAND":
        return f"{unit_at_location} D"
    elif action == "WAIVE":
        return "WAIVE"
    else:
        raise ValueError(f"Unknown action: {action}")


def submit_all_orders(ctx: ToolContext, orders: list[dict]) -> dict[str, Any]:
    """Submit a complete list of orders for all units at once."""
    game = ctx.game
    power_name = ctx.power

    constructed_orders = []
    errors = []

    all_possible_orders = game.get_all_possible_orders()

    for order in orders:
        try:
            order_str = _construct_order_string(game, power_name, order)

            action = order.get("action", "").upper()
            if action == "WAIVE":
                waive_valid = any(
                    "WAIVE" in valid_orders
                    for valid_orders in all_possible_orders.values()
                )
                if not waive_valid:
                    errors.append({
                        "unit_location": order.get("unit_location"),
                        "error": "WAIVE is not valid.",
                        "attempted_order": "WAIVE",
                        "valid_orders": [],
                    })
                    continue
                constructed_orders.append(order_str)
                continue

            unit_location = order.get("unit_location", "").upper()
            if unit_location.startswith(("A ", "F ")) and len(unit_location) > 2:
                unit_location = unit_location[2:].strip()

            if unit_location in all_possible_orders:
                valid_orders = all_possible_orders[unit_location]
                if order_str not in valid_orders:
                    errors.append({
                        "unit_location": order.get("unit_location"),
                        "error": "Order is invalid.",
                        "attempted_order": order_str,
                        "valid_orders": valid_orders,
                    })
                    continue

            constructed_orders.append(order_str)
        except Exception as e:
            errors.append(
                {"unit_location": order.get("unit_location"), "error": str(e)}
            )

    if errors:
        return {
            "success": False,
            "errors": errors,
            "partial_orders": constructed_orders,
        }

    try:
        game.set_orders(power_name, constructed_orders)
        logger.info(f"Submitted {len(constructed_orders)} orders for {power_name}")
        return {
            "success": True,
            "power": power_name,
            "submitted_orders": constructed_orders,
            "count": len(constructed_orders),
        }
    except Exception as e:
        logger.error(f"Failed to submit orders: {e}")
        return {
            "success": False,
            "error": f"Failed to submit orders: {str(e)}",
            "attempted_orders": constructed_orders,
        }


def send_message(ctx: ToolContext, recipient: str, message: str) -> dict[str, Any]:
    """Send a message to a specific power or to global chat."""
    game = ctx.game
    power_name = ctx.power
    recipient = recipient.upper()

    if recipient != "GLOBAL" and recipient not in game.powers:
        return {
            "success": False,
            "error": f"Invalid recipient '{recipient}'. Must be a power name or 'GLOBAL'",
            "valid_powers": list(game.powers.keys()),
        }

    current_count = ctx.message_counts.get(recipient, 0)
    if current_count >= MESSAGE_LIMIT:
        return {
            "success": False,
            "error": f"Out of messages with {recipient}",
        }

    try:
        phase = game.get_current_phase()
        timestamp = int(time.time() * 1000000)

        msg = Message(
            sender=power_name,
            recipient=recipient,
            phase=phase,
            message=message,
            time_sent=timestamp,
        )

        game.add_message(msg)

        if ctx.message_queues is not None:
            if recipient != "GLOBAL":
                ctx.message_queues[recipient].append({
                    "sender": power_name,
                    "message": message,
                    "timestamp": timestamp,
                })
            else:
                for p in ctx.message_queues:
                    if p != power_name:
                        ctx.message_queues[p].append({
                            "sender": power_name,
                            "message": message,
                            "timestamp": timestamp,
                            "global": True,
                        })

        ctx.message_counts[recipient] = current_count + 1
        remaining = MESSAGE_LIMIT - (current_count + 1)

        return {
            "success": True,
            "messages_remaining_with_recipient": remaining,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to send message: {str(e)}",
        }


def check_messages(
    ctx: ToolContext,
    filter_by_sender: Optional[str] = None,
    limit: int = 30,
) -> dict[str, Any]:
    """Check messages received by this agent."""
    game = ctx.game
    power_name = ctx.power

    if filter_by_sender:
        filter_by_sender = filter_by_sender.upper()
        if filter_by_sender not in game.powers and filter_by_sender != "GLOBAL":
            return {
                "success": False,
                "error": f"Invalid sender '{filter_by_sender}'",
                "valid_senders": list(game.powers.keys()) + ["GLOBAL"],
            }

    try:
        result_messages = []
        all_msgs = list(game.messages.values())

        for message in all_msgs:
            if message.recipient != power_name and message.recipient != "GLOBAL":
                continue
            if message.sender == power_name:
                continue
            if filter_by_sender and message.sender != filter_by_sender:
                continue

            result_messages.append({
                "sender": message.sender,
                "recipient": message.recipient,
                "message": message.message,
                "timestamp": message.time_sent,
                "phase": message.phase,
            })

        result_messages.sort(key=lambda x: x["timestamp"], reverse=True)

        if limit and len(result_messages) > limit:
            result_messages = result_messages[:limit]

        return {
            "success": True,
            "messages": result_messages,
            "count": len(result_messages),
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to retrieve messages: {str(e)}",
        }


def finish_phase(ctx: ToolContext) -> dict[str, Any]:
    """Signal that the agent is done with this phase."""
    game = ctx.game
    power_name = ctx.power
    phase_type = game.phase_type
    current_orders = game.get_orders(power_name)

    missing_orders = []

    if phase_type == "M":
        my_units = game.get_units(power_name)
        ordered_units = set()
        for order in current_orders:
            parts = order.split()
            if len(parts) >= 2 and parts[0] in ["A", "F"]:
                ordered_units.add(f"{parts[0]} {parts[1]}")
        for unit in my_units:
            clean_unit = unit.lstrip("*")
            if clean_unit not in ordered_units:
                missing_orders.append(clean_unit)

    elif phase_type == "R":
        power = game.get_power(power_name)
        if hasattr(power, "retreats") and power.retreats:
            dislodged_units = power.retreats.keys()
            ordered_units = set()
            for order in current_orders:
                parts = order.split()
                if len(parts) >= 2 and parts[0] in ["A", "F"]:
                    ordered_units.add(f"{parts[0]} {parts[1]}")
            for unit in dislodged_units:
                if unit not in ordered_units:
                    missing_orders.append(unit)

    elif phase_type == "A":
        n_centers = len(game.get_centers(power_name))
        n_units = len(game.get_units(power_name))
        diff = n_centers - n_units

        if diff > 0:
            power = game.get_power(power_name)
            available_builds = power.adjust if hasattr(power, "adjust") else []
            if available_builds:
                submitted_count = len(current_orders)
                max_builds = len(available_builds)
                required_count = min(diff, max_builds)
                if submitted_count < required_count:
                    missing_orders.append(
                        f"{required_count - submitted_count} more build order(s)"
                    )
        elif diff < 0:
            submitted_count = len(current_orders)
            required_count = abs(diff)
            if submitted_count < required_count:
                missing_orders.append(
                    f"{required_count - submitted_count} more disband order(s)"
                )

    if missing_orders:
        return {
            "success": False,
            "error": f"Missing orders for: {', '.join(missing_orders)}. Submit orders before finishing.",
        }

    if phase_type == "M":
        other_powers = [p for p in game.powers if p != power_name]
        missing_recipients = [
            p for p in other_powers if ctx.message_counts.get(p, 0) == 0
        ]
        if missing_recipients:
            return {
                "success": False,
                "error": f"You haven't messaged: {', '.join(missing_recipients)}. Diplomacy requires negotiating with all powers! Use send_message to contact them before finishing.",
            }

    ctx.finished_phase = True
    logger.info(f"{power_name} finished phase {game.get_current_phase()}")
    return {
        "success": True,
        "message": "Orders submitted. Your turn is complete.",
    }


def write_diary(ctx: ToolContext, message: str) -> Dict[str, Any]:
    """Write a message to the diary for the current game phase."""
    game = ctx.game
    power = ctx.power
    current_phase = game.get_current_phase()

    diary_path = _get_diary_path(ctx.log_dir, power)

    try:
        diary = _load_diary(diary_path)
        if current_phase not in diary:
            diary[current_phase] = []
        diary[current_phase].append(message)
        _save_diary(diary_path, diary)

        return {
            "success": True,
            "phase": current_phase,
            "message": message,
            "total_entries_in_phase": len(diary[current_phase]),
        }
    except Exception as e:
        logger.error(f"Failed to write diary for {power} phase {current_phase}: {e}")
        return {"success": False, "error": str(e), "phase": current_phase}


def read_diary(
    ctx: ToolContext,
    phase: Optional[str] = None,
    last_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Read entries from the diary."""
    power = ctx.power
    diary_path = _get_diary_path(ctx.log_dir, power)

    try:
        diary = _load_diary(diary_path)

        if not diary:
            return {"success": True, "entries": {}, "total_entries": 0, "message": "Diary is empty"}

        if phase:
            entries = diary.get(phase, [])
            return {"success": True, "phase": phase, "entries": entries, "total_entries": len(entries)}

        if last_n:
            last_messages = _get_last_n_messages(diary, last_n)
            return {"success": True, "last_n": last_n, "entries": last_messages, "total_entries": len(last_messages)}

        total = sum(len(messages) for messages in diary.values())
        return {"success": True, "entries": diary, "total_entries": total}

    except Exception as e:
        logger.error(f"Error reading diary for {power}: {e}")
        return {"success": False, "error": str(e)}


def _get_diary_path(log_dir: Optional[Path], power: str) -> Path:
    if log_dir is None:
        log_dir = Path("./logs")
    return log_dir / f"{power.lower()}_diary.json"


def _load_diary(diary_path: Path) -> Dict[str, List[str]]:
    if not diary_path.exists():
        return {}
    try:
        with open(diary_path, "r") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
    except (json.JSONDecodeError, Exception):
        return {}


def _save_diary(diary_path: Path, data: Dict[str, List[str]]) -> None:
    diary_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = diary_path.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
    temp_file.replace(diary_path)


def _get_last_n_messages(diary: Dict[str, List[str]], n: int) -> List[Dict[str, str]]:
    all_messages = []
    for phase, messages in diary.items():
        for message in messages:
            all_messages.append({"phase": phase, "message": message})
    return all_messages[-n:] if len(all_messages) > n else all_messages


# =============================================================================
# Tool Dispatch
# =============================================================================

TOOL_DISPATCH = {
    "list_units": list_units,
    "get_possible_orders": get_possible_orders,
    "get_game_state": get_game_state,
    "submit_all_orders": submit_all_orders,
    "send_message": send_message,
    "check_messages": check_messages,
    "finish_phase": finish_phase,
    "write_diary": write_diary,
    "read_diary": read_diary,
}


def execute_tool(
    ctx: ToolContext, tool_name: str, args: dict[str, Any]
) -> dict[str, Any] | str:
    """Execute a tool by name with the given arguments."""
    if tool_name not in TOOL_DISPATCH:
        available_tools = list(TOOL_DISPATCH.keys())
        return {
            "error": f"Unknown tool: '{tool_name}'",
            "available_tools": available_tools,
        }

    func = TOOL_DISPATCH[tool_name]
    try:
        result = func(ctx, **args)
        return result
    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {e.__class__.__name__}: {e}", exc_info=True)
        return f"Tool {tool_name} failed: {e.__class__.__name__}: {e}"


# =============================================================================
# Tool Schemas (for system prompt)
# =============================================================================

TOOL_SCHEMAS = [
    {
        "name": "list_units",
        "description": "Returns a list of all units on the board, who owns them, and what type they are (Army 'A' or Fleet 'F').",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_possible_orders",
        "description": "Get all possible orders for units controlled by this agent.\n\nExamples:\n  get_possible_orders()  # All units\n  get_possible_orders(unit_location=\"PAR\")  # Just Paris unit",
        "parameters": {
            "type": "object",
            "properties": {
                "unit_location": {
                    "type": "string",
                    "description": "Optional: specific unit location (e.g., 'PAR')",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_game_state",
        "description": "Get comprehensive information about the current game state.\n\nReturns phase, units, supply centers, current orders, and phase-specific info.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "submit_all_orders",
        "description": "Submit a complete list of orders for all units at once.\n\nIMPORTANT: unit_location should always be only a province name.\n  Correct: {\"unit_location\": \"PAR\", \"action\": \"HOLD\"}\n  Invalid: {\"unit_location\": \"A PAR\", \"action\": \"HOLD\"}\n\nExample:\n  submit_all_orders(orders=[{\"unit_location\": \"PAR\", \"action\": \"HOLD\"}, {\"unit_location\": \"MAR\", \"action\": \"MOVE\", \"destination\": \"BUR\"}])",
        "parameters": {
            "type": "object",
            "properties": {
                "orders": {
                    "type": "array",
                    "description": "List of orders to submit",
                    "items": {
                        "type": "object",
                        "properties": {
                            "unit_location": {"type": "string", "description": "Province name only (e.g., 'PAR')"},
                            "action": {"type": "string", "enum": ["HOLD", "MOVE", "SUPPORT_HOLD", "SUPPORT_MOVE", "CONVOY", "MOVE_VIA_CONVOY", "RETREAT", "BUILD", "DISBAND", "WAIVE"]},
                            "destination": {"type": "string", "description": "Destination for MOVE/RETREAT actions"},
                            "support_unit_location": {"type": "string", "description": "Location of unit being supported"},
                            "support_destination": {"type": "string", "description": "Destination of supported move"},
                            "convoy_unit_location": {"type": "string", "description": "Location of army being convoyed"},
                            "convoy_destination": {"type": "string", "description": "Destination of convoyed army"},
                            "unit_type": {"type": "string", "enum": ["A", "F"], "description": "Unit type for BUILD"},
                        },
                        "required": ["unit_location", "action"],
                    },
                },
            },
            "required": ["orders"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a specific power or to global chat.\n\nExamples:\n  send_message(recipient=\"GERMANY\", message=\"Would you like to ally?\")\n  send_message(recipient=\"GLOBAL\", message=\"Good luck everyone!\")",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Power name (e.g., 'GERMANY') or 'GLOBAL'"},
                "message": {"type": "string", "description": "The message content"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "check_messages",
        "description": "Check messages received by this agent.\n\nExamples:\n  check_messages()  # All messages\n  check_messages(filter_by_sender=\"GERMANY\")",
        "parameters": {
            "type": "object",
            "properties": {
                "filter_by_sender": {"type": "string", "description": "Optional: filter to messages from a specific power"},
                "limit": {"type": "integer", "description": "Maximum number of messages to return (default: 30)"},
            },
            "required": [],
        },
    },
    {
        "name": "finish_phase",
        "description": "When you have submitted all orders and finished sending messages, call this to end your turn.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "write_diary",
        "description": "Write a message to your diary for the current game phase.\n\nExamples:\n  write_diary(message=\"Germany seems trustworthy.\")\n  write_diary(message=\"CRITICAL: Italy violated our agreement.\")",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The diary entry to write"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "read_diary",
        "description": "Read entries from your diary.\n\nExamples:\n  read_diary()  # Read all entries\n  read_diary(phase=\"S1901M\")  # Entries from specific phase\n  read_diary(last_n=5)  # Last 5 messages",
        "parameters": {
            "type": "object",
            "properties": {
                "phase": {"type": "string", "description": "Optional: specific phase to read"},
                "last_n": {"type": "integer", "description": "Optional: get last N messages"},
            },
            "required": [],
        },
    },
]


def _simplify_property(prop: dict) -> str | dict:
    """Simplify a JSON Schema property to a readable format."""
    if prop.get("type") == "array" and "items" in prop:
        items = prop["items"]
        if items.get("type") == "object" and "properties" in items:
            fields = []
            for name, item_prop in items["properties"].items():
                if "enum" in item_prop:
                    fields.append(f"{name}: one of {item_prop['enum']}")
                else:
                    desc = item_prop.get("description", item_prop.get("type", "string"))
                    fields.append(f"{name}: {desc}")
            return f"ARRAY of objects, each with: {', '.join(fields)}"
        return prop.get("description", "array")
    elif "enum" in prop:
        return f"one of: {prop['enum']}"
    else:
        return prop.get("description", prop.get("type", ""))


def get_openai_tools() -> list[dict]:
    """Convert TOOL_SCHEMAS to OpenAI function calling format for opponents."""
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        }
        for schema in TOOL_SCHEMAS
    ]


def get_tool_schemas_str() -> str:
    """Return tool schemas as formatted string for system prompt."""
    simplified = []
    for schema in TOOL_SCHEMAS:
        tool = {"name": schema["name"], "description": schema["description"]}
        params = schema.get("parameters", {})
        props = params.get("properties", {})
        if props:
            tool["args"] = {
                name: _simplify_property(prop) for name, prop in props.items()
            }
        else:
            tool["args"] = {}
        simplified.append(tool)
    return json.dumps(simplified, indent=2)
