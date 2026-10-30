"""Smoke test: send one message to the configured OpenAI model (makes a paid API call).

Usage: python -m scripts.smoke_test_llm
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage

from agent.config import ConfigError, load_llm_config
from agent.llm import create_llm


def main() -> int:
    try:
        config = load_llm_config()
        llm = create_llm(config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print(f"Model: {config.model}")
    try:
        response = llm.invoke([HumanMessage(content="Reply with one short sentence confirming you are working.")])
    except Exception as exc:  # network/auth/rate-limit errors; never print secrets
        print(f"LLM call failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Response: {response.content}")
    usage = getattr(response, "usage_metadata", None)
    print(f"Token usage: {usage if usage else 'not available'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
