#!/usr/bin/env python3
"""ALFWorld Game Implementation (Text Engine Version).

This module provides the ALFWorldGame class that implements AbstractGame interface.
It wraps the ALFWorld text-based environment for LLM training.

Example:
    from alfworld_game import ALFWorldGame

    game = ALFWorldGame()
    obs = game.reset()
    result = game.step("go to desk 1")
"""

import glob
import os
import random
import re
import threading
from typing import Any, Dict, List, Optional
import logging

from opentinker.environment.base_game import AbstractGame, StepResult

# ALFWorld imports - install with: pip install alfworld
try:
    import alfworld.agents.environment as alfworld_env
    import textworld
    from textworld.envs.wrappers import Filter
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
    import yaml

    ALFWORLD_AVAILABLE = True
except ImportError:
    ALFWORLD_AVAILABLE = False
    logging.warning("alfworld not installed. Install with: pip install alfworld")


class ALFWorldGame(AbstractGame):
    """ALFWorld text-based environment game implementation.

    This implementation wraps ALFWorld's TextWorld environment for LLM RL training.
    The agent interacts through text commands to complete household tasks.

    Attributes:
        max_steps: Maximum steps per episode (default: 50)
        task_types: List of task types to sample from (default: all)
    """

    # Reward constants
    REWARD_SUCCESS = 10.0
    REWARD_FAILURE = -1.0
    REWARD_STEP = -0.01  # Small penalty per step to encourage efficiency
    REWARD_INVALID_ACTION = -0.1

    # Dense reward constants (intermediate progress signals)
    REWARD_NAVIGATE = 0.05      # Successfully moved to a new location
    REWARD_TAKE_OBJECT = 0.2    # Picked up an object
    REWARD_PUT_OBJECT = 0.15    # Put an object somewhere
    REWARD_OPEN_CLOSE = 0.05    # Opened or closed a container
    REWARD_USE_APPLIANCE = 0.3  # Used heat/cool/clean (key task actions)
    REWARD_EXAMINE = 0.1        # Examined an object (for examine tasks)

    # Limits
    DEFAULT_MAX_STEPS = 30

    # Task types in ALFWorld
    ALL_TASK_TYPES = [
        "pick_and_place_simple",
        "look_at_obj_in_light",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
    ]

    # Process-level cache for ALFWorld environments (safe within same process)
    # Key: (config_path, split, num_games) -> initialized environment
    # Note: Each Ray worker process gets its own cache, avoiding cross-process issues
    # IMPORTANT: Set to False to disable sharing when running parallel instances
    _shared_envs: dict = {}
    _use_shared_env: bool = False  # Disabled by default to prevent state conflicts

    # ==========================================
    # OPTIMIZATION: Shared game path cache
    # Only the first instance scans the disk, others reuse the cached paths
    # ==========================================
    _cached_game_paths: Dict[
        str, List[str]
    ] = {}  # key: cache_key -> list of game paths
    _cache_lock = threading.Lock()

    # NOTE: TextWorld's tatsu parser is NOT thread-safe.
    # Parallelism is achieved via sharding (--shards N launches N independent server processes).

    def __init__(
        self,
        config_path: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        task_types: Optional[List[str]] = None,
        split: str = "train",
        num_games: int = -1,
        use_shared_env: bool = False,  # Default to per-instance env for safety
    ):
        """Initialize ALFWorld game.

        Args:
            config_path: Path to ALFWorld config file (base_config.yaml)
            max_steps: Maximum steps per episode
            task_types: Task types to sample from (None = all types)
            split: Dataset split ("train", "eval_in_distribution", "eval_out_of_distribution")
            num_games: Number of games to load (-1 = all games, e.g. 64 for faster loading)
            use_shared_env: If True, share env across instances (fast but causes state conflicts with parallel runs).
                           If False, each instance gets its own env (slower init but correct).
        """
        if not ALFWORLD_AVAILABLE:
            raise ImportError(
                "alfworld package not installed. " "Install with: pip install alfworld"
            )

        self.config_path = config_path
        self.max_steps = max_steps
        self.task_types = task_types or self.ALL_TASK_TYPES
        self.split = split
        self.num_games = num_games
        self._use_shared_env = use_shared_env

        # Load ALFWorld config
        self._load_alfworld_config()

        # Game state (instance-specific, NOT shared)
        self._env = None  # Will be created in reset()
        self._own_env = None  # Per-instance env when not using shared env
        self._tw_env = None  # TextWorld environment instance (for optimized mode)
        self._current_obs = None
        self._current_info = None
        self._step_count = 0
        self._task_desc = ""
        self._admissible_commands = []
        self._done = False
        self._initialized = False  # Track if engine is ready

    def _load_alfworld_config(self):
        """Load ALFWorld configuration."""

        if self.config_path:
            with open(self.config_path, "r") as f:
                self._config = yaml.safe_load(f)
        else:
            # ALFWorld data directory (need to run 'alfworld-download' first)
            alfworld_data = os.path.expandvars(
                os.environ.get("ALFWORLD_DATA", "$HOME/.cache/alfworld")
            )

            # Use default config with all required fields
            self._config = {
                "general": {
                    "training_method": "dqn",  # or 'dagger'
                },
                "env": {
                    "type": "AlfredTWEnv",
                    "regen_game_files": False,
                    "domain_randomization": False,
                    "goal_desc_human_anns_prob": 0,  # Required by AlfredTWEnv
                    "task_types": [1, 2, 3, 4, 5, 6],  # All 6 task types
                    "expert_type": "handcoded",  # or 'planner'
                    "registration": {
                        "batch_size": 1,
                    },
                    "split": self.split,
                },
                "dataset": {
                    "data_path": f"{alfworld_data}/json_2.1.1/train",
                    "eval_id_data_path": f"{alfworld_data}/json_2.1.1/valid_seen",
                    "eval_ood_data_path": f"{alfworld_data}/json_2.1.1/valid_unseen",
                    "num_train_games": self.num_games,  # -1 means use all games
                    "num_eval_games": self.num_games,
                },
                "logic": {
                    "domain": f"{alfworld_data}/logic/alfred.pddl",
                    "grammar": f"{alfworld_data}/logic/alfred.twl2",
                },
                "rl": {
                    "training": {
                        "max_nb_steps_per_episode": self.max_steps,
                    },
                },
            }

    def _get_cached_game_paths(self) -> List[str]:
        """Get cached game paths. Only the first instance scans the disk.

        OPTIMIZATION: This avoids 32 instances each scanning 8810 game directories.
        Instead, only the first instance scans, and others reuse the cached result.

        ALFWorld directory structure:
            json_2.1.1/train/pick_and_place_simple-xxx/trial_xxx/game.tw-pddl

        Returns:
            List of trial directory paths (containing game.tw-pddl files)
        """
        # Create a unique cache key based on split and task types
        cache_key = f"{self.split}:{','.join(sorted(self.task_types))}:{self.num_games}"

        with ALFWorldGame._cache_lock:
            if cache_key not in ALFWorldGame._cached_game_paths:
                # First instance: scan the disk
                alfworld_data = os.path.expandvars(
                    os.environ.get("ALFWORLD_DATA", "$HOME/.cache/alfworld")
                )

                # Determine base path based on split
                if self.split == "train":
                    base_path = f"{alfworld_data}/json_2.1.1/train"
                elif self.split == "eval_in_distribution":
                    base_path = f"{alfworld_data}/json_2.1.1/valid_seen"
                elif self.split == "eval_out_of_distribution":
                    base_path = f"{alfworld_data}/json_2.1.1/valid_unseen"
                else:
                    base_path = f"{alfworld_data}/json_2.1.1/{self.split}"

                print(
                    f"[ALFWorldGame] First-time scanning: {base_path} (PID: {os.getpid()})..."
                )

                # Scan for all matching task types
                # ALFWorld structure: task_type-xxx/trial_xxx/game.tw-pddl
                # NOTE: Some trial directories are incomplete (missing game files), so we filter them
                all_paths = []
                for task_type in self.task_types:
                    # Pattern to find trial directories
                    pattern = os.path.join(base_path, f"{task_type}-*", "trial_*")
                    matching = glob.glob(pattern)

                    # Only keep directories that actually contain a game file
                    for trial_path in matching:
                        game_file = os.path.join(trial_path, "game.tw-pddl")
                        if os.path.exists(game_file):
                            all_paths.append(trial_path)
                        else:
                            # Check for alternative game file formats
                            alt_files = glob.glob(os.path.join(trial_path, "game.*"))
                            if alt_files:
                                all_paths.append(trial_path)

                # Sort for consistency
                all_paths.sort()
                print(
                    f"[ALFWorldGame] Found {len(all_paths)} valid games (with game files)."
                )

                # Apply num_games limit if specified
                if self.num_games > 0 and len(all_paths) > self.num_games:
                    # Use a fixed seed so all instances get the same subset
                    rng = random.Random(42)
                    all_paths = rng.sample(all_paths, self.num_games)
                    all_paths.sort()

                ALFWorldGame._cached_game_paths[cache_key] = all_paths
                print(
                    f"[ALFWorldGame] Scan complete. Found {len(all_paths)} games. Cached for reuse."
                )
            else:
                # Subsequent instances: reuse cached paths (no disk scan!)
                pass

        return ALFWorldGame._cached_game_paths[cache_key]

    def _init_env(self):
        """Initialize ALFWorld environment.

        OPTIMIZED: Uses direct TextWorld loading instead of ALFWorld's heavy init_env.
        This avoids redundant disk scanning across multiple instances.
        """
        if self._initialized:
            return

        # Mark as initialized (engine ready, will load specific game on reset)
        self._initialized = True
        print(f"[ALFWorldGame] Engine ready (PID: {os.getpid()}, id: {id(self)})")

    def reset(
        self, task_type: Optional[str] = None, seed: Optional[int] = None, **kwargs
    ) -> str:
        """Reset the game to a new episode.

        OPTIMIZED: Uses cached game paths and direct TextWorld loading.
        This avoids redundant disk scanning - only the first instance scans.

        Args:
            task_type: Specific task type to use (None = random from task_types)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments (ignored)

        Returns:
            Initial observation string
        """
        # Initialize engine if needed
        self._init_env()

        # Set seed if provided
        if seed is not None:
            random.seed(seed)

        # Get cached game paths (only first instance scans disk)
        # Paths are now trial directories: .../task_type-xxx/trial_xxx/
        game_paths = self._get_cached_game_paths()

        if not game_paths:
            raise RuntimeError(
                f"No games found for split='{self.split}', task_types={self.task_types}"
            )

        # Select a random game from the cached paths (trial directory)
        selected_path = random.choice(game_paths)

        # Game file is directly in the trial directory
        game_file = os.path.join(selected_path, "game.tw-pddl")

        # Fallback to other possible game file names
        if not os.path.exists(game_file):
            game_file = os.path.join(selected_path, "game.z8")
        if not os.path.exists(game_file):
            # Try to find any game file
            possible_files = glob.glob(os.path.join(selected_path, "game.*"))
            if possible_files:
                game_file = possible_files[0]
            else:
                raise RuntimeError(f"No game file found in {selected_path}")

        # Close previous TextWorld environment if exists
        if self._tw_env is not None:
            try:
                self._tw_env.close()
            except Exception as e:
                logging.warning("Failed to close previous TextWorld environment: %s", e)

        # Create TextWorld environment with direct file loading (no disk scan!)
        infos = textworld.EnvInfos(
            feedback=True,
            inventory=True,
            description=True,
            admissible_commands=True,
            score=True,
            max_score=True,
            won=True,
            lost=True,
            extras=["walkthrough", "expert_plan"],
        )

        # Direct loading: bypass ALFWorld's init_env which scans all 8810 games
        # Thread safety: parallelism comes from sharding (multiple server processes)
        self._tw_env = textworld.start(
            game_file, infos, wrappers=[Filter, AlfredDemangler()]
        )
        self._env = self._tw_env  # For compatibility with step()

        # Reset the environment
        game_state, info = self._tw_env.reset()

        # Extract observation
        if hasattr(game_state, "feedback"):
            self._current_obs = game_state.feedback
        elif isinstance(game_state, str):
            self._current_obs = game_state
        else:
            self._current_obs = str(game_state)

        self._current_info = info

        # Parse task description from observation
        self._task_desc = self._extract_task_description(self._current_obs)

        # Get admissible commands
        if hasattr(game_state, "admissible_commands"):
            self._admissible_commands = game_state.admissible_commands or []
        else:
            self._admissible_commands = info.get("admissible_commands", [])

        # Reset state
        self._step_count = 0
        self._done = False

        return self._format_observation(self._current_obs)

    def _extract_task_description(self, obs: str) -> str:
        """Extract task description from observation."""
        # ALFWorld observations typically start with the task
        lines = obs.strip().split("\n")
        for line in lines:
            if line.startswith("Your task is to"):
                return line
        return "Complete the household task."

    def _format_observation(self, obs: str) -> str:
        """Format observation for LLM."""
        formatted = f"=== Current State ===\n{obs}\n"

        if self._admissible_commands:
            formatted += "\n=== Available Actions ===\n"
            formatted += "\n".join(f"- {cmd}" for cmd in self._admissible_commands)

        return formatted

    def step(self, action: str) -> StepResult:
        """Execute an action in the environment.

        Args:
            action: Text command to execute (e.g., "go to desk 1", "take book 1")

        Returns:
            StepResult with observation, reward, done flag, and info
        """
        if self._done:
            return StepResult(
                observation="Episode already finished.",
                reward=0.0,
                done=True,
                info={"error": "episode_finished"},
            )

        self._step_count += 1

        # Parse action from LLM output
        parsed_action = self._parse_action(action)

        # Execute action in TextWorld environment
        # TextWorld returns: game_state, reward, done, info
        # Thread safety: parallelism comes from sharding (multiple server processes)
        game_state, reward, done, info = self._tw_env.step(parsed_action)

        # Extract observation from game_state
        if hasattr(game_state, "feedback"):
            obs = game_state.feedback
        elif isinstance(game_state, str):
            obs = game_state
        else:
            obs = str(game_state)

        # Update state
        self._current_obs = obs

        # Get admissible commands
        # NOTE: Depending on wrappers / TextWorld versions, admissible commands may be
        # stored either on game_state or inside info dict. If we drop this, the agent
        # becomes "blind" and often collapses to repetitive actions (e.g., always 'look').
        if hasattr(game_state, "admissible_commands"):
            self._admissible_commands = game_state.admissible_commands or []
        else:
            self._admissible_commands = info.get("admissible_commands", [])

        # Check for timeout
        if self._step_count >= self.max_steps and not done:
            done = True
            reward = self.REWARD_FAILURE
            obs = f"TIMEOUT: Maximum steps ({self.max_steps}) reached.\n\n{obs}"

        # Adjust rewards with dense reward shaping
        if done and reward > 0:
            # Task completed successfully
            final_reward = self.REWARD_SUCCESS
            obs = f"SUCCESS! Task completed!\n\n{obs}"
        elif done:
            # Task failed
            final_reward = self.REWARD_FAILURE
        else:
            # Calculate dense rewards based on action success
            final_reward = self._calculate_dense_reward(parsed_action, obs)

        self._done = done

        return StepResult(
            observation=self._format_observation(obs),
            reward=final_reward,
            done=done,
            info={
                # Note: Don't include "step" here as gym_environment_interaction.py
                # already passes it explicitly to observation_template.format()
                "raw_reward": float(reward),
                "action_taken": parsed_action,
                "task": self._task_desc,
            },
        )

    def _parse_action(self, raw_action: str) -> str:
        """Parse action from LLM output.

        Supports formats:
        - <action>go to desk 1</action>
        - Direct command: go to desk 1
        """
        # Try to extract from <action> tags
        match = re.search(
            r"<action>\s*(.*?)\s*</action>", raw_action, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()

        # Otherwise, use the last line as action
        lines = raw_action.strip().split("\n")
        return lines[-1].strip()

    def _calculate_dense_reward(self, action: str, obs: str) -> float:
        """Calculate dense reward based on action success.

        Provides intermediate rewards for progress toward task completion.
        This helps the agent learn faster by providing more frequent feedback.

        Args:
            action: The action that was taken
            obs: The observation/feedback from the environment

        Returns:
            Dense reward value
        """
        action_lower = action.lower()
        obs_lower = obs.lower()

        # Check for invalid actions first
        if "nothing happens" in obs_lower or "invalid" in obs_lower:
            return self.REWARD_INVALID_ACTION

        # Navigation reward: successfully moved to a location
        if action_lower.startswith("go to"):
            # Successful navigation shows what's at the location
            if "you see" in obs_lower or "on the" in obs_lower or "is closed" in obs_lower:
                return self.REWARD_NAVIGATE + self.REWARD_STEP
            return self.REWARD_STEP

        # Take object reward: successfully picked up an object
        if action_lower.startswith("take"):
            if "you pick up" in obs_lower:
                return self.REWARD_TAKE_OBJECT + self.REWARD_STEP
            return self.REWARD_STEP

        # Put object reward: successfully placed an object
        if action_lower.startswith("put"):
            if "you put" in obs_lower:
                return self.REWARD_PUT_OBJECT + self.REWARD_STEP
            return self.REWARD_STEP

        # Open/close reward: successfully opened or closed a container
        if action_lower.startswith("open") or action_lower.startswith("close"):
            if "you open" in obs_lower or "you close" in obs_lower:
                return self.REWARD_OPEN_CLOSE + self.REWARD_STEP
            return self.REWARD_STEP

        # Use appliance rewards: heat, cool, clean (key task actions)
        if action_lower.startswith("heat"):
            if "you heat" in obs_lower:
                return self.REWARD_USE_APPLIANCE + self.REWARD_STEP
            return self.REWARD_STEP

        if action_lower.startswith("cool"):
            if "you cool" in obs_lower:
                return self.REWARD_USE_APPLIANCE + self.REWARD_STEP
            return self.REWARD_STEP

        if action_lower.startswith("clean"):
            if "you clean" in obs_lower:
                return self.REWARD_USE_APPLIANCE + self.REWARD_STEP
            return self.REWARD_STEP

        # Use (for examine tasks with desklamp)
        if action_lower.startswith("use"):
            if "you turn on" in obs_lower:
                return self.REWARD_EXAMINE + self.REWARD_STEP
            return self.REWARD_STEP

        # Examine reward
        if action_lower.startswith("examine"):
            return self.REWARD_EXAMINE + self.REWARD_STEP

        # Default: small step penalty
        return self.REWARD_STEP

    def get_system_prompt(self) -> str:
        """Return the system prompt for ALFWorld."""
        return (
            "You are an AI assistant playing ALFWorld, a text-based household environment.\n"
            "Your goal is to complete household tasks by interacting with objects.\n\n"
            "IMPORTANT: You MUST respond in the following format:\n"
            "1. First, briefly think about your next step in <thinking></thinking> tags\n"
            "2. Then, output EXACTLY ONE action in <action></action> tags\n\n"
            "Available actions:\n"
            "- go to [receptacle N]: Move to a location (e.g., 'go to desk 1', 'go to fridge 1')\n"
            "- take [object N] from [receptacle N]: Pick up an object (e.g., 'take apple 1 from countertop 1')\n"
            "- put [object N] in/on [receptacle N]: Place an object (e.g., 'put apple 1 in/on fridge 1')\n"
            "- open [receptacle N]: Open a container (e.g., 'open fridge 1')\n"
            "- close [receptacle N]: Close a container\n"
            "- use [appliance N]: Use a device like lamp, microwave (e.g., 'use desklamp 1')\n"
            "- heat [object N] with [appliance N]: Heat object (e.g., 'heat apple 1 with microwave 1')\n"
            "- cool [object N] with [appliance N]: Cool object (e.g., 'cool apple 1 with fridge 1')\n"
            "- clean [object N] with [receptacle N]: Clean object (e.g., 'clean apple 1 with sinkbasin 1')\n"
            "- examine [object N]: Look at an object closely\n"
            "- inventory: Check what you're holding\n"
            "- look: Look around the current location\n\n"
            "CRITICAL RULES:\n"
            "- Always include the number after object/receptacle names (e.g., 'desk 1' not 'desk')\n"
            "- You must 'go to' a receptacle before you can interact with objects there\n"
            "- You must 'take' an object before you can 'put' it somewhere\n"
            "- For heating: go to microwave, put object in microwave, heat, take object out\n"
            "- For cooling: go to fridge, open fridge, put object in fridge, cool, take object out\n"
            "- For cleaning: go to sinkbasin, clean object with sinkbasin\n\n"
            "=== EXAMPLE 1: Pick and Place ===\n"
            "Task: put a pencil in desk.\n\n"
            "<thinking>I need to find a pencil. Let me check the shelf first.</thinking>\n"
            "<action>go to shelf 1</action>\n"
            "> On the shelf 1, you see a pencil 1, a book 2.\n"
            "<thinking>Found pencil 1. I'll pick it up.</thinking>\n"
            "<action>take pencil 1 from shelf 1</action>\n"
            "> You pick up the pencil 1 from the shelf 1.\n"
            "<thinking>Now I need to put it in the desk.</thinking>\n"
            "<action>go to desk 1</action>\n"
            "> On the desk 1, you see a lamp 1.\n"
            "<thinking>I'm at the desk. Now I'll put the pencil here.</thinking>\n"
            "<action>put pencil 1 in/on desk 1</action>\n"
            "> You put the pencil 1 in/on the desk 1.\n"
            "SUCCESS!\n\n"
            "=== EXAMPLE 2: Heat then Place ===\n"
            "Task: heat some egg and put it in fridge.\n\n"
            "<thinking>I need to find an egg first.</thinking>\n"
            "<action>go to countertop 1</action>\n"
            "> On the countertop 1, you see an egg 1.\n"
            "<thinking>Found egg 1. I'll take it.</thinking>\n"
            "<action>take egg 1 from countertop 1</action>\n"
            "> You pick up the egg 1.\n"
            "<thinking>Now I need to heat it in the microwave.</thinking>\n"
            "<action>go to microwave 1</action>\n"
            "> The microwave 1 is closed.\n"
            "<thinking>I need to open it first.</thinking>\n"
            "<action>open microwave 1</action>\n"
            "> You open the microwave 1.\n"
            "<thinking>Now put the egg in and heat it.</thinking>\n"
            "<action>put egg 1 in/on microwave 1</action>\n"
            "> You put the egg 1 in/on the microwave 1.\n"
            "<action>heat egg 1 with microwave 1</action>\n"
            "> You heat the egg 1 using the microwave 1.\n"
            "<thinking>Egg is heated. Take it out and put in fridge.</thinking>\n"
            "<action>take egg 1 from microwave 1</action>\n"
            "> You pick up the egg 1.\n"
            "<action>go to fridge 1</action>\n"
            "> The fridge 1 is closed.\n"
            "<action>open fridge 1</action>\n"
            "> You open the fridge 1.\n"
            "<action>put egg 1 in/on fridge 1</action>\n"
            "> You put the egg 1 in/on the fridge 1.\n"
            "SUCCESS!\n\n"
            "=== EXAMPLE 3: Examine in Light ===\n"
            "Task: examine the book with the desklamp.\n\n"
            "<thinking>I need to find a book and bring it to a desklamp.</thinking>\n"
            "<action>go to shelf 1</action>\n"
            "> On the shelf 1, you see a book 1.\n"
            "<action>take book 1 from shelf 1</action>\n"
            "> You pick up the book 1.\n"
            "<thinking>Now find a desklamp.</thinking>\n"
            "<action>go to desk 1</action>\n"
            "> On the desk 1, you see a desklamp 1.\n"
            "<thinking>Use the desklamp to examine the book.</thinking>\n"
            "<action>use desklamp 1</action>\n"
            "> You turn on the desklamp 1.\n"
            "SUCCESS!\n\n"
            "Now complete the given task. Keep your thinking brief and take action!"
        )

    def get_initial_user_message(self) -> str:
        """Return the initial user message for ALFWorld."""
        return (
            f"Task: {self._task_desc}\n\n"
            "Explore the environment and complete the task. "
            "What would you like to do first?"
        )

    def get_state(self) -> Dict[str, Any]:
        """Return current game state."""
        return {
            "observation": self._current_obs,
            "task": self._task_desc,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "done": self._done,
            "admissible_commands": self._admissible_commands[:10],
        }

    # =========================================================================
    # Data Generation Methods (for training)
    # =========================================================================

    def generate_initial_state(self) -> Dict[str, Any]:
        """Generate random initial state for training data.

        Returns a dict with task configuration that reset() will use.
        """
        # Randomly select a task type
        task_type = random.choice(self.task_types)

        return {
            "task_type": task_type,
            "seed": random.randint(0, 1000000),
        }

    def get_user_message_with_state(
        self, task_type: Optional[str] = None, **kwargs
    ) -> str:
        """Generate user message with rendered initial state for prompt."""
        # Reset to get actual observatio
        self.reset(task_type=task_type, **kwargs)

        return (
            f"Task: {self._task_desc}\n\n"
            f"{self._format_observation(self._current_obs)}\n\n"
            "What would you like to do?"
        )

    def get_interaction_name(self) -> str:
        """Return interaction name for ALFWorld."""
        return "alfworld"
