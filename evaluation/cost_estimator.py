# Rough per-1K-token costs in USD
# (input + output blended for simplicity)

MODEL_COSTS = {
    "gpt-4o": 0.005,
    "gpt-3.5-turbo": 0.0015,
}


def estimate_tokens(text: str) -> int:
    """
    Very rough token estimation:
    ~4 characters ≈ 1 token
    """
    return max(1, int(len(text) / 4))


def estimate_cost(context: str, answer: str, model: str):
    input_tokens = estimate_tokens(context)
    output_tokens = estimate_tokens(answer)

    total_tokens = input_tokens + output_tokens
    cost_per_1k = MODEL_COSTS.get(model, 0.0)

    cost = (total_tokens / 1000) * cost_per_1k

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost, 6)
    }
