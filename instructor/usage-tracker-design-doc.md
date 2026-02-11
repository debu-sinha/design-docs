# Instructor Usage Tracker: Built-in Usage Aggregation and Cost Estimation

**Author:** Debu Sinha ([@debu-sinha](https://github.com/debu-sinha))
**Status:** Draft
**Related Issues:** [#2080](https://github.com/567-labs/instructor/issues/2080), [#1814](https://github.com/567-labs/instructor/issues/1814) (closed, partial)
**Target:** `instructor` library

---

## Part I: Design Sketch

### 1. Motivation

Instructor has a well-designed hooks system (`completion:kwargs`, `completion:response`, `completion:error`, `parse:error`, `completion:last_attempt`) that fires at every critical point in the completion lifecycle. It also tracks token usage per-response and aggregates across retries within a single call.

What's missing is a **built-in way to aggregate usage across multiple calls in a session** and **estimate costs across providers**. Today, every user who wants usage visibility has to write their own handler from scratch (as demonstrated in `examples/hooks/run.py`). This is boilerplate that the library should provide.

The existing `instructor usage` CLI command only works with OpenAI's `/v1/usage` API endpoint, meaning users of Anthropic, Gemini, Bedrock, LiteLLM, and other providers have no built-in usage tracking at all.

### 2. Goals

1. **Session-level usage aggregation** -- track tokens, calls, retries, and errors across the lifetime of an Instructor client
2. **Provider-agnostic normalization** -- unified usage schema regardless of whether the response comes from OpenAI, Anthropic, Bedrock, or Gemini
3. **Cost estimation** -- built-in pricing data for common models across providers, with user-extensible pricing
4. **Zero-config activation** -- one line to enable, plugs into existing hooks
5. **Per-call and aggregate views** -- access both individual call records and running totals
6. **Thread-safe** -- safe for concurrent usage in async code and multi-threaded applications
7. **Export-friendly** -- `.to_dict()`, `.to_json()`, and `.summary()` for logging, dashboards, and debugging

### 3. Non-Goals

- Real-time streaming cost calculation (token counts aren't available until stream completes)
- Persistent storage or database integration (users can export and store themselves)
- OpenTelemetry span integration (separate concern, can be layered on top via hooks)
- Replacing the existing `update_total_usage` retry-level aggregation in `utils/core.py`

### 4. Requirements

| Requirement | Priority | Notes |
|-------------|----------|-------|
| Track prompt_tokens, completion_tokens, total_tokens per call | P0 | Normalized across providers |
| Track cached/reasoning token breakdowns when available | P1 | OpenAI details, Anthropic cache tokens |
| Aggregate across all calls on a client | P0 | Session-level totals |
| Per-call history with timestamps and model info | P0 | For debugging and auditing |
| Cost estimation with extensible pricing | P1 | Built-in for common models |
| Track retry counts and validation errors | P0 | Already available via hooks |
| Thread-safe for async usage | P0 | Lock-based or atomic operations |
| No new dependencies | P0 | Use only stdlib + existing deps |
| Works with all providers (OpenAI, Anthropic, Bedrock, Gemini, LiteLLM) | P0 | Provider detection from response type |

### 5. Proposal Overview

Add a `UsageTracker` class to `instructor/usage_tracker.py` that:

1. Registers handlers on the existing `completion:kwargs`, `completion:response`, and `completion:error` hooks
2. Normalizes provider-specific usage objects into a unified `UsageRecord` dataclass
3. Maintains a thread-safe list of per-call records and running totals
4. Provides `.summary()`, `.to_dict()`, `.to_json()`, and `.reset()` methods
5. Supports user-extensible cost estimation via `CostCalculator`

**One-line activation:**

```python
import instructor
from instructor import UsageTracker

client = instructor.from_openai(openai.OpenAI())
tracker = UsageTracker()
client.on("completion:response", tracker.on_response)
client.on("completion:kwargs", tracker.on_request)
client.on("completion:error", tracker.on_error)
```

**Or even simpler with a convenience method:**

```python
tracker = UsageTracker.attach(client)
```

**Access usage data:**

```python
# After making calls...
print(tracker.total_tokens)          # 4521
print(tracker.total_cost)            # 0.0234
print(tracker.total_calls)           # 12
print(tracker.total_retries)         # 3
print(tracker.summary())             # Pretty-printed summary
print(tracker.to_dict())             # Full data as dict
```

### 6. Alternatives Considered

#### Alternative A: Add new hook types (e.g., `usage:accumulated`)

**Pros:**
- Clean separation of concerns in hook naming
- Could carry normalized data

**Cons:**
- Breaks the existing hook contract (5 well-defined hooks)
- Requires changes to `retry.py` to emit new events
- Hook data is already available in `completion:response`

**Rejected because:** The existing hooks already provide all necessary data. Adding new hooks increases API surface without adding capability.

#### Alternative B: Extend `update_total_usage` in `utils/core.py`

**Pros:**
- Modifies existing code, no new modules

**Cons:**
- `update_total_usage` is scoped to a single call's retries, not cross-call
- Would conflate per-call retry tracking with session-level aggregation
- Hard to expose to users without changing the return type

**Rejected because:** Different scopes. Per-call retry aggregation and session-level tracking serve different purposes and should remain separate.

#### Alternative C: Make it a separate package (e.g., `instructor-usage`)

**Pros:**
- No changes to core library
- Independent release cycle

**Cons:**
- Discovery problem (users won't find it)
- Misses the opportunity for "batteries included"
- The code is small enough to live in-tree

**Rejected because:** This is a natural extension of the hooks system and should ship with the library.

---

## Part II: Detailed Design

### 7. Data Model

```python
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageRecord:
    """Normalized usage record for a single completion call."""

    timestamp: float
    model: str | None
    provider: str  # "openai", "anthropic", "bedrock", "gemini", "litellm", "unknown"

    # Core token counts (normalized across providers)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Extended breakdowns (when available)
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    audio_tokens: int = 0

    # Call metadata
    retries: int = 0
    had_error: bool = False
    error_type: str | None = None

    # Cost (calculated if pricing available)
    estimated_cost: float | None = None
```

### 8. Provider Normalization

The key design challenge is mapping provider-specific usage objects to a unified schema:

| Field | OpenAI | Anthropic | Bedrock | Gemini |
|-------|--------|-----------|---------|--------|
| `prompt_tokens` | `usage.prompt_tokens` | `usage.input_tokens` | `usage.inputTokens` | `usage_metadata.prompt_token_count` |
| `completion_tokens` | `usage.completion_tokens` | `usage.output_tokens` | `usage.outputTokens` | `usage_metadata.candidates_token_count` |
| `total_tokens` | `usage.total_tokens` | input + output | input + output | `usage_metadata.total_token_count` |
| `cached_tokens` | `prompt_tokens_details.cached_tokens` | `cache_read_input_tokens` | N/A | `usage_metadata.cached_content_token_count` |
| `reasoning_tokens` | `completion_tokens_details.reasoning_tokens` | N/A | N/A | N/A |

The normalizer detects the provider from the response object type:

```python
def _normalize_usage(self, response: Any, model: str | None) -> UsageRecord:
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageRecord(timestamp=time.time(), model=model, provider="unknown")

    # OpenAI (CompletionUsage)
    if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
        return self._normalize_openai(usage, model)

    # Anthropic (anthropic.types.Usage)
    if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
        return self._normalize_anthropic(usage, model)

    # Bedrock (dict with inputTokens/outputTokens)
    if isinstance(usage, dict) and "inputTokens" in usage:
        return self._normalize_bedrock(usage, model)

    return UsageRecord(timestamp=time.time(), model=model, provider="unknown")
```

### 9. Cost Estimation

Built-in pricing for common models, extensible by users:

```python
# Default pricing (per token, not per 1K)
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"prompt": 2.50 / 1_000_000, "completion": 10.00 / 1_000_000},
    "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
    "gpt-4-turbo": {"prompt": 10.00 / 1_000_000, "completion": 30.00 / 1_000_000},
    "o1": {"prompt": 15.00 / 1_000_000, "completion": 60.00 / 1_000_000},
    "o1-mini": {"prompt": 1.10 / 1_000_000, "completion": 4.40 / 1_000_000},
    "o3-mini": {"prompt": 1.10 / 1_000_000, "completion": 4.40 / 1_000_000},
    # Anthropic
    "claude-sonnet-4-5-20250929": {"prompt": 3.00 / 1_000_000, "completion": 15.00 / 1_000_000},
    "claude-3-5-haiku-20241022": {"prompt": 0.80 / 1_000_000, "completion": 4.00 / 1_000_000},
    "claude-opus-4-6": {"prompt": 15.00 / 1_000_000, "completion": 75.00 / 1_000_000},
    # Gemini
    "gemini-2.0-flash": {"prompt": 0.10 / 1_000_000, "completion": 0.40 / 1_000_000},
    "gemini-2.5-pro": {"prompt": 1.25 / 1_000_000, "completion": 10.00 / 1_000_000},
}


class CostCalculator:
    """Extensible cost calculator for LLM token usage."""

    def __init__(self, custom_pricing: dict[str, dict[str, float]] | None = None):
        self._pricing = {**_DEFAULT_PRICING}
        if custom_pricing:
            self._pricing.update(custom_pricing)

    def add_model(self, model: str, prompt_cost: float, completion_cost: float) -> None:
        """Add or override pricing for a model (costs per token)."""
        self._pricing[model] = {"prompt": prompt_cost, "completion": completion_cost}

    def estimate(self, record: UsageRecord) -> float | None:
        """Estimate cost for a usage record. Returns None if model pricing unknown."""
        if record.model is None:
            return None
        # Try exact match, then prefix match (for versioned model names)
        pricing = self._pricing.get(record.model)
        if pricing is None:
            for prefix, p in self._pricing.items():
                if record.model.startswith(prefix):
                    pricing = p
                    break
        if pricing is None:
            return None
        return (
            record.prompt_tokens * pricing["prompt"]
            + record.completion_tokens * pricing["completion"]
        )
```

### 10. UsageTracker Core

```python
class UsageTracker:
    """
    Built-in usage aggregation for instructor clients.

    Plugs into the existing hooks system to track token usage, costs,
    and call statistics across the lifetime of an instructor client.

    Example:
        >>> client = instructor.from_openai(openai.OpenAI())
        >>> tracker = UsageTracker.attach(client)
        >>> # ... make calls ...
        >>> print(tracker.summary())
    """

    def __init__(self, cost_calculator: CostCalculator | None = None):
        self._records: list[UsageRecord] = []
        self._lock = threading.Lock()
        self._cost_calculator = cost_calculator or CostCalculator()
        self._current_model: str | None = None
        self._current_attempt: int = 0

    @classmethod
    def attach(cls, client: Any, **kwargs: Any) -> UsageTracker:
        """Attach a UsageTracker to an instructor client."""
        tracker = cls(**kwargs)
        client.on("completion:kwargs", tracker.on_request)
        client.on("completion:response", tracker.on_response)
        client.on("completion:error", tracker.on_error)
        return tracker

    def detach(self, client: Any) -> None:
        """Remove this tracker's handlers from an instructor client."""
        client.off("completion:kwargs", self.on_request)
        client.off("completion:response", self.on_response)
        client.off("completion:error", self.on_error)

    # --- Hook handlers ---

    def on_request(self, *args: Any, **kwargs: Any) -> None:
        """Handler for completion:kwargs hook. Captures model name."""
        self._current_model = kwargs.get("model") or kwargs.get("modelId")
        self._current_attempt += 1

    def on_response(self, response: Any) -> None:
        """Handler for completion:response hook. Records usage."""
        record = self._normalize_usage(response, self._current_model)
        record.retries = max(0, self._current_attempt - 1)
        record.estimated_cost = self._cost_calculator.estimate(record)
        with self._lock:
            self._records.append(record)
        self._current_attempt = 0

    def on_error(self, error: Exception) -> None:
        """Handler for completion:error hook. Records failed call."""
        record = UsageRecord(
            timestamp=time.time(),
            model=self._current_model,
            provider="unknown",
            had_error=True,
            error_type=type(error).__name__,
        )
        with self._lock:
            self._records.append(record)
        self._current_attempt = 0

    # --- Aggregated properties ---

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return sum(r.total_tokens for r in self._records)

    @property
    def total_prompt_tokens(self) -> int:
        with self._lock:
            return sum(r.prompt_tokens for r in self._records)

    @property
    def total_completion_tokens(self) -> int:
        with self._lock:
            return sum(r.completion_tokens for r in self._records)

    @property
    def total_cost(self) -> float:
        with self._lock:
            return sum(r.estimated_cost or 0.0 for r in self._records)

    @property
    def total_calls(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def total_errors(self) -> int:
        with self._lock:
            return sum(1 for r in self._records if r.had_error)

    @property
    def records(self) -> list[UsageRecord]:
        with self._lock:
            return list(self._records)

    # --- Export ---

    def summary(self) -> str:
        """Human-readable usage summary."""
        with self._lock:
            models = {}
            for r in self._records:
                key = r.model or "unknown"
                if key not in models:
                    models[key] = {"calls": 0, "tokens": 0, "cost": 0.0}
                models[key]["calls"] += 1
                models[key]["tokens"] += r.total_tokens
                models[key]["cost"] += r.estimated_cost or 0.0

        lines = [
            f"Usage Summary ({self.total_calls} calls, {self.total_tokens} tokens, ${self.total_cost:.4f})",
            "-" * 60,
        ]
        for model, stats in sorted(models.items()):
            lines.append(
                f"  {model}: {stats['calls']} calls, {stats['tokens']} tokens, ${stats['cost']:.4f}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export all usage data as a dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost": self.total_cost,
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "records": [
                {
                    "timestamp": r.timestamp,
                    "model": r.model,
                    "provider": r.provider,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "total_tokens": r.total_tokens,
                    "cached_tokens": r.cached_tokens,
                    "reasoning_tokens": r.reasoning_tokens,
                    "estimated_cost": r.estimated_cost,
                    "retries": r.retries,
                    "had_error": r.had_error,
                    "error_type": r.error_type,
                }
                for r in self.records
            ],
        }

    def reset(self) -> None:
        """Clear all recorded usage data."""
        with self._lock:
            self._records.clear()
            self._current_attempt = 0
```

### 11. Module Layout

```
instructor/
  usage_tracker.py           # UsageTracker, UsageRecord, CostCalculator
  __init__.py                # Export UsageTracker
tests/
  test_usage_tracker.py      # Unit tests
examples/
  hooks/
    usage_tracking.py        # Example with real client
```

Changes to existing files:
- `instructor/__init__.py`: Add `UsageTracker` to public exports
- No changes to `hooks.py`, `retry.py`, or `client.py`

### 12. Testing Plan

**Unit tests (no API keys needed):**

```python
def test_normalize_openai_usage():
    """OpenAI CompletionUsage is normalized correctly."""
    tracker = UsageTracker()
    mock_response = Mock()
    mock_response.usage = CompletionUsage(
        prompt_tokens=100, completion_tokens=50, total_tokens=150
    )
    tracker.on_response(mock_response)
    assert tracker.total_tokens == 150
    assert tracker.total_prompt_tokens == 100
    assert tracker.total_completion_tokens == 50


def test_normalize_anthropic_usage():
    """Anthropic Usage is normalized correctly."""
    tracker = UsageTracker()
    mock_response = Mock()
    mock_response.usage = Mock(input_tokens=80, output_tokens=40,
                                cache_read_input_tokens=10,
                                cache_creation_input_tokens=5)
    tracker.on_response(mock_response)
    assert tracker.total_tokens == 120
    assert tracker.total_prompt_tokens == 80


def test_cost_estimation():
    """Cost is calculated from model pricing."""
    tracker = UsageTracker()
    tracker._current_model = "gpt-4o"
    mock_response = Mock()
    mock_response.usage = CompletionUsage(
        prompt_tokens=1000, completion_tokens=500, total_tokens=1500
    )
    tracker.on_response(mock_response)
    assert tracker.total_cost > 0


def test_custom_pricing():
    """Users can provide custom model pricing."""
    calc = CostCalculator(custom_pricing={
        "my-model": {"prompt": 0.001, "completion": 0.002}
    })
    tracker = UsageTracker(cost_calculator=calc)
    # ...


def test_thread_safety():
    """Concurrent calls don't corrupt state."""
    tracker = UsageTracker()
    # Spawn threads that call on_response concurrently
    # Assert total_tokens matches expected sum


def test_attach_detach():
    """attach() registers hooks, detach() removes them."""
    mock_client = Mock()
    tracker = UsageTracker.attach(mock_client)
    assert mock_client.on.call_count == 3
    tracker.detach(mock_client)
    assert mock_client.off.call_count == 3


def test_reset():
    """reset() clears all data."""
    tracker = UsageTracker()
    # Add some records
    tracker.reset()
    assert tracker.total_calls == 0
    assert tracker.total_tokens == 0


def test_to_dict_roundtrip():
    """to_dict() produces complete, serializable output."""
    # ...


def test_summary_format():
    """summary() produces human-readable output."""
    # ...


@pytest.mark.parametrize("provider", ["openai", "anthropic", "bedrock"])
def test_multi_provider_aggregation(provider):
    """Usage from different providers aggregates correctly."""
    # ...
```

### 13. Rollout Plan

1. **Phase 1: Core implementation** -- `UsageTracker`, `UsageRecord`, `CostCalculator`, tests
2. **Phase 2: Documentation** -- Update hooks docs, add usage tracking guide
3. **Phase 3: CLI integration** -- Extend `instructor usage` to support local tracking (not just OpenAI API)

### 14. Open Questions

1. **Should `attach()` be a method on `Instructor` itself?** E.g., `client.track_usage()` returning a tracker. This would be more discoverable but requires modifying `client.py`.

2. **Should we track latency per call?** The hooks fire before and after the API call, so wall-clock time per call is available. Adding `duration_ms` to `UsageRecord` would be useful but slightly complicates the handler (need to store start time in `on_request`).

3. **Should pricing be fetched dynamically?** Services like LiteLLM maintain pricing databases. We could optionally pull from there, but this adds a network dependency. Keeping static pricing with user override seems safer.

4. **Context manager support?** E.g., `with UsageTracker.track(client) as tracker:` for scoped tracking. Clean but adds API surface.

---

## Appendix

### A. Existing Hooks Architecture

The hooks system in `instructor/core/hooks.py` provides 5 event types fired in this order during a completion call:

```
completion:kwargs  -->  [API Call]  -->  completion:response (success)
                                   -->  completion:error (API error)
                                   -->  parse:error (validation error)
                                        --> [retry if attempts remain]
                                        --> completion:last_attempt (final failure)
```

The `UsageTracker` only needs 3 of these hooks:
- `completion:kwargs` -- to capture the model name and count attempts
- `completion:response` -- to extract usage from the response
- `completion:error` -- to record failed calls

### B. Related Issues

- **#2080**: Feature proposal issue for this design (opened by author)
- **#1814**: "Add usage field when receiving response from LLM" -- closed, but shows demand for usage access in structured responses

### C. Relationship to Existing `update_total_usage`

The `update_total_usage` function in `utils/core.py` aggregates tokens **within a single call's retry loop**. It modifies the response's `.usage` field so the final response contains cumulative usage from all retry attempts.

`UsageTracker` operates at a different level -- it aggregates **across multiple calls** on a client. These two mechanisms are complementary and do not overlap.

### D. References

- [Instructor Hooks Documentation](https://python.useinstructor.com/concepts/hooks/)
- [OpenAI Usage API](https://platform.openai.com/docs/api-reference/chat/object#chat-create-usage)
- [Anthropic Usage](https://docs.anthropic.com/en/docs/build-with-claude/token-counting)
- [LiteLLM Cost Tracking](https://docs.litellm.ai/docs/completion/token_usage)
