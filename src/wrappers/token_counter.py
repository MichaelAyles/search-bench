"""Token estimation utilities for cross-tool comparison."""

from dataclasses import dataclass

# Approximate pricing per 1M tokens (input/output) as of 2025
PRICING = {
    "claude": {"input": 3.0, "output": 15.0, "model": "claude-sonnet-4"},
    "codex": {"input": 2.5, "output": 10.0, "model": "codex-1"},
    "gemini": {"input": 0.075, "output": 0.30, "model": "gemini-2.5-flash"},
    "copilot": {"input": 1.0, "output": 5.0, "model": "claude-haiku-4.5"},
}


@dataclass
class TokenCost:
    tool_name: str
    tokens_input: int
    tokens_output: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output


def estimate_cost(tool_name: str, tokens_input: int, tokens_output: int) -> TokenCost:
    """Estimate USD cost for a tool invocation."""
    pricing = PRICING.get(tool_name, {"input": 3.0, "output": 15.0})
    cost = (tokens_input * pricing["input"] + tokens_output * pricing["output"]) / 1_000_000
    return TokenCost(
        tool_name=tool_name,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        cost_usd=cost,
    )


def estimate_tokens_from_text(text: str) -> int:
    """Rough token estimate from text length (chars / 4 approximation).
    Use tiktoken for accurate counts when available."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        # Rough approximation: ~4 chars per token for English/code
        return len(text) // 4
