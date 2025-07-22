import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure parent directory in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ------------------------- Task Loading -------------------------
_tasks_cache: Optional[Dict[str, Dict]] = None


def load_tasks() -> Dict[str, Dict]:
    """Load system prompt task presets from presets/system_prompt_tasks.json."""
    global _tasks_cache
    if _tasks_cache is not None:
        return _tasks_cache

    project_root = Path(current_dir).parent
    tasks_path = project_root / "presets" / "system_prompt_tasks.json"
    if not tasks_path.exists():
        raise FileNotFoundError(f"system_prompt_tasks.json not found at {tasks_path}")

    with open(tasks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # map by task name
        _tasks_cache = {entry["task"]: entry for entry in data if isinstance(entry, dict) and entry.get("task")}
        return _tasks_cache

# ------------------------- Node Definition -------------------------

class SystemPromptTaskGenerator:
    """ComfyUI node to generate a system prompt based on predefined context tasks."""

    _task_names: list[str] = []

    @classmethod
    def INPUT_TYPES(cls):
        try:
            tasks = load_tasks()
            cls._task_names = list(tasks.keys())
        except FileNotFoundError as e:
            print(f"Error loading tasks for node: {e}")
            cls._task_names = ["Error: system_prompt_tasks.json not found"]

        return {
            "required": {
                "task": (cls._task_names, {"default": cls._task_names[0] if cls._task_names else ""}),
            },
            "optional": {
                "context": ("*", {}),
            },
        }

    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("context", "system_prompt")
    FUNCTION = "generate_prompt"
    CATEGORY = "llm_toolkit/prompt"

    def generate_prompt(self, task: str, context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        # Prepare context copy / init
        if context is None:
            output_context: Dict[str, Any] = {}
        elif isinstance(context, dict):
            output_context = context.copy()
        else:
            output_context = {"passthrough_data": context}

        tasks = load_tasks()
        task_data = tasks.get(task)
        if not task_data:
            err = f"Error: Task '{task}' not found."
            print(err)
            # ensure provider_config exists
            if "provider_config" not in output_context:
                output_context["provider_config"] = {}
            output_context["provider_config"]["system_message"] = err
            return (output_context, err)

        system_prompt: str = task_data.get("prompt", "")

        provider_config = output_context.get("provider_config", {})
        if not isinstance(provider_config, dict):
            provider_config = {}
        provider_config["system_message"] = system_prompt
        # Optionally add meta like num_instructions
        if task_data.get("num_instructions"):
            provider_config["num_instructions"] = task_data["num_instructions"]
        output_context["provider_config"] = provider_config

        print(f"SystemPromptTaskGenerator: Generated prompt for task '{task}'.")
        return (output_context, system_prompt)

# --- Node Mappings ---
NODE_CLASS_MAPPINGS = {
    "SystemPromptTaskGenerator": SystemPromptTaskGenerator,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SystemPromptTaskGenerator": "Kontext System Prompt text (LLMToolkit)",
} 