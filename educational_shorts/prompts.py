from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_prompt(name: str) -> str:
    """
    Load a prompt by filename.

    Example:
        load_prompt("topic_generation")
    """

    path = PROMPTS_DIR / f"{name}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")

    return path.read_text(encoding="utf-8").strip()