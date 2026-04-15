"""Local token counting for chat completions.

Amplify does not return token usage in its responses, so this module
provides translator-estimated counts using a characters-per-token
heuristic.  All numbers are estimates, not authoritative provider
billing figures.

The counting is performed against the fully rendered Amplify request
(post-transformation) so that system-prompt injection, tool-schema
insertion, and any wrapper text are included in the prompt count.
"""
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Characters per token -- conservative heuristic.
# English text averages ~4 chars/token for GPT-family tokenizers.
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from a string using chars/4 heuristic.

    Returns 0 for empty or None input.
    """
    if not text:
        return 0
    return len(text) // _CHARS_PER_TOKEN


def count_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Sum estimated token counts across a list of Amplify-format messages.

    Each message is expected to have a 'content' string field.
    Per-message overhead (role, delimiters) is approximated as 4 tokens.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        # ~4 tokens overhead per message for role + delimiters
        total += 4
    return total


def count_prompt_tokens(amplify_request: Dict[str, Any]) -> int:
    """Count estimated prompt tokens from a fully rendered Amplify request.

    This operates on the post-transformation request so that system-prompt
    injection, tool-schema text, and wrapper formatting are captured.
    """
    data = amplify_request.get("data", {})
    messages = data.get("messages", [])
    return count_message_tokens(messages)


def count_completion_tokens(content: str) -> int:
    """Count estimated completion tokens from assistant response text.

    Should be called after mixed-content and tool-call parsing has
    stabilized, so the count reflects what the OpenAI-compatible
    response actually represents.
    """
    return estimate_tokens(content)
