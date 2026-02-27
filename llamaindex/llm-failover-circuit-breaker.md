# LLM Failover and Circuit Breaker Framework for LlamaIndex

|                    |                                        |
| ------------------ | -------------------------------------- |
| **Author(s)**      | Debu Sinha ([@debu-sinha](https://github.com/debu-sinha)) |
| **Organization**   | LlamaIndex Community                   |
| **Status**         | DRAFT                                  |
| **GitHub Issue**   | [#19631](https://github.com/run-llama/llama_index/issues/19631) |
| **Pull Request**   | TBD                                    |
| **Target Reviewers** | [@AstraBert](https://github.com/AstraBert) (Clelia Bertelli, LlamaIndex Maintainer), [@logan-markewich](https://github.com/logan-markewich) (Logan Markewich, LlamaIndex Lead Maintainer) |

**Change Log:**

- 2026-02-27: v1.0 -- Initial design proposal

---

## Executive Summary

LlamaIndex has no built-in mechanism to handle LLM provider failures at the application level. When a provider goes down, hits rate limits, or returns server errors, every call fails even though the same request could succeed on a different provider. LangChain solved this with `RunnableWithFallbacks`, Semantic Kernel ships retry + circuit breaker + resource governor, and Haystack provides pipeline recovery snapshots.

This proposal adds two composable primitives to `llama-index-core`:

1. **`FallbackLLM`** -- wraps a primary LLM and an ordered list of fallbacks. When the primary fails with a retriable error, the request is transparently retried on the next provider. Follows the `StructuredLLM` composition pattern already established in the codebase.

2. **`CircuitBreaker`** -- tracks consecutive failures per provider and short-circuits requests to unhealthy providers for a configurable cooldown window, preventing wasted latency on providers that are known to be down.

Together with the existing `TokenBucketRateLimiter` (PR [#20712](https://github.com/run-llama/llama_index/pull/20712)) and `Retry-After` header support (PR [#20813](https://github.com/run-llama/llama_index/pull/20813)), this forms a complete production resilience stack:

- **Proactive throttling**: `TokenBucketRateLimiter` prevents hitting rate limits
- **Reactive backoff**: `Retry-After` header parsing waits the right amount
- **Provider-level recovery**: `FallbackLLM` + `CircuitBreaker` routes around failures

---

## Part I: Design Overview

### 1.1 Motivation

Production LLM applications commonly use multiple providers. A chatbot might use GPT-4o as primary and Claude 3.5 Sonnet as backup. A RAG pipeline might use a fast model for retrieval augmentation and fall back to a slower one during outages. Today, implementing this in LlamaIndex requires wrapping every LLM call in try/except logic at the application layer:

```python
# What users have to write today
try:
    response = openai_llm.chat(messages)
except Exception:
    try:
        response = anthropic_llm.chat(messages)
    except Exception:
        response = local_llm.chat(messages)
```

This is error-prone, duplicated across every call site, and doesn't compose with `Settings.llm`, pipelines, or agents. LangChain recognized this and added `with_fallbacks()` as a first-class API. LlamaIndex should match this capability.

### 1.2 User Stories

**Story 1: Simple failover.** "I want to set `Settings.llm` to an LLM that tries OpenAI first, falls back to Anthropic on failure, without changing any downstream code."

```python
from llama_index.core.llms import FallbackLLM
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic

Settings.llm = FallbackLLM(
    llm=OpenAI(model="gpt-4o"),
    fallbacks=[Anthropic(model="claude-sonnet-4-20250514")],
)

# Everything downstream works unchanged
index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine().query("What is LlamaIndex?")
```

**Story 2: Rate-limit-aware failover.** "When OpenAI returns 429, I want to fail over to Anthropic immediately instead of waiting for the retry-after delay, because the user is waiting."

```python
Settings.llm = FallbackLLM(
    llm=OpenAI(model="gpt-4o"),
    fallbacks=[Anthropic(model="claude-sonnet-4-20250514")],
    # Only fail over on rate limits and server errors, not auth errors
    failover_on=(RateLimitError, ServerError, ConnectionError, TimeoutError),
)
```

**Story 3: Circuit breaker.** "If OpenAI has failed 5 times in a row, stop trying it for 60 seconds and go straight to the fallback. Don't waste latency on a provider that's clearly down."

```python
Settings.llm = FallbackLLM(
    llm=OpenAI(model="gpt-4o"),
    fallbacks=[Anthropic(model="claude-sonnet-4-20250514")],
    circuit_breaker=CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60.0,
    ),
)
```

### 1.3 Design Principles

1. **Transparent composition.** `FallbackLLM` is a subclass of `LLM`. It can be assigned to `Settings.llm`, passed to any component that takes an LLM, and serialized/deserialized like any other LLM. Users don't change downstream code.

2. **Failover happens after provider-level retries.** Each provider LLM (OpenAI, Anthropic, etc.) already has its own retry logic (tenacity, SDK retries). `FallbackLLM` only kicks in when those retries are exhausted. This prevents double-retrying.

3. **Configurable exception filter.** Not all errors should trigger failover. Authentication errors (`401`) mean the API key is bad on all providers. Only transient errors (rate limits, timeouts, server errors, connection errors) should trigger failover. Users can customize this.

4. **Circuit breaker is optional.** Simple failover works without it. The circuit breaker adds value for high-throughput applications where a downed provider wastes latency.

5. **Follows existing patterns.** The implementation mirrors `StructuredLLM` (wraps an inner LLM, delegates `metadata`), uses the same Pydantic model pattern as `TokenBucketRateLimiter`, and integrates with the existing `@llm_chat_callback()`/`@llm_completion_callback()` decorator stack.

### 1.4 Scope

**In scope:**
- `FallbackLLM` class in `llama-index-core`
- `CircuitBreaker` class in `llama-index-core`
- All 8 `BaseLLM` abstract methods (chat, complete, stream_chat, stream_complete + async variants)
- Unit tests and integration tests
- Documentation

**Out of scope (future work):**
- `FallbackEmbedding` (same pattern, separate PR)
- Load balancing / round-robin routing (different use case)
- Cost-aware routing (pick cheapest available provider)
- Distributed circuit breaker state (Redis-backed, for multi-process deployments)

---

## Part II: Detailed Design

### 2.1 Exception Classification

Since LlamaIndex has no common exception base across providers, we define a default set of exception types that should trigger failover. These map to the transient error categories that all major providers share.

```python
# llama_index/core/llms/failover.py

# Default exceptions that trigger failover.
# Users can override this per FallbackLLM instance.
DEFAULT_FAILOVER_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
)
```

Provider-specific exceptions (like `openai.RateLimitError` or `anthropic.InternalServerError`) are SDK-specific and can't be imported in core. Instead, we match on HTTP semantics: the callback layer already catches `BaseException` and re-raises. The `FallbackLLM` catches exceptions in the `DEFAULT_FAILOVER_EXCEPTIONS` tuple, plus any additional types the user configures. Since `openai.APIConnectionError` inherits from Python's `ConnectionError` and `openai.APITimeoutError` inherits from `TimeoutError`, the defaults cover the most common cases without importing provider SDKs.

For provider-specific exceptions that don't inherit from standard types (like `openai.RateLimitError` or `openai.InternalServerError`), users pass them explicitly:

```python
import openai

Settings.llm = FallbackLLM(
    llm=OpenAI(model="gpt-4o"),
    fallbacks=[Anthropic(model="claude-sonnet-4-20250514")],
    failover_on=(
        *DEFAULT_FAILOVER_EXCEPTIONS,
        openai.RateLimitError,
        openai.InternalServerError,
    ),
)
```

### 2.2 CircuitBreaker

The circuit breaker tracks consecutive failures per LLM instance and prevents requests to providers that are known to be unhealthy.

```python
# llama_index/core/llms/circuit_breaker.py

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _ProviderState:
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False


class CircuitBreaker:
    """
    Tracks consecutive failures per provider and short-circuits
    requests to unhealthy providers.

    States:
      CLOSED   -- provider is healthy, requests flow normally
      OPEN     -- provider has failed >= failure_threshold times,
                  requests are rejected immediately
      HALF-OPEN -- recovery_timeout has elapsed since the circuit
                   opened, one probe request is allowed through

    Args:
        failure_threshold: Number of consecutive failures before
            the circuit opens. Default 5.
        recovery_timeout: Seconds to wait before allowing a probe
            request to a failed provider. Default 60.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._states: dict[int, _ProviderState] = {}
        self._lock = threading.Lock()

    def _get_state(self, provider_id: int) -> _ProviderState:
        if provider_id not in self._states:
            self._states[provider_id] = _ProviderState()
        return self._states[provider_id]

    def is_available(self, provider_id: int) -> bool:
        """Check if a provider is available for requests."""
        with self._lock:
            state = self._get_state(provider_id)
            if not state.is_open:
                return True
            # Check if recovery timeout has elapsed (half-open)
            if time.monotonic() - state.last_failure_time >= self.recovery_timeout:
                return True
            return False

    def record_success(self, provider_id: int) -> None:
        """Record a successful request, resetting the failure count."""
        with self._lock:
            state = self._get_state(provider_id)
            state.consecutive_failures = 0
            state.is_open = False

    def record_failure(self, provider_id: int) -> None:
        """Record a failed request, potentially opening the circuit."""
        with self._lock:
            state = self._get_state(provider_id)
            state.consecutive_failures += 1
            state.last_failure_time = time.monotonic()
            if state.consecutive_failures >= self.failure_threshold:
                state.is_open = True
```

Design decisions:

- **`id(llm)` as provider key.** Each LLM instance gets a unique provider ID via Python's `id()`. This naturally handles the case where the same provider class is used with different API keys or endpoints.
- **Thread-safe via `threading.Lock`.** Same pattern as `TokenBucketRateLimiter`.
- **Half-open state.** After `recovery_timeout`, one request is allowed through. If it succeeds, the circuit closes. If it fails, the circuit stays open for another `recovery_timeout` period.
- **Not a Pydantic model.** Unlike `TokenBucketRateLimiter`, the circuit breaker holds mutable state that doesn't need serialization. A plain class is simpler.

### 2.3 FallbackLLM

The core class. Follows the `StructuredLLM` composition pattern.

```python
# llama_index/core/llms/fallback_llm.py

import logging
from typing import Any, Optional, Sequence

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, SerializeAsAny
from llama_index.core.llms.llm import LLM
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)

logger = logging.getLogger(__name__)

DEFAULT_FAILOVER_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
)


class FallbackLLM(LLM):
    """
    An LLM wrapper that tries a primary LLM and falls back to
    alternatives on transient failures.

    When the primary LLM raises an exception matching ``failover_on``,
    the request is retried on each fallback in order. If all providers
    fail, the exception from the primary LLM is raised.

    Works transparently with ``Settings.llm``, query engines, agents,
    and any component that accepts an LLM.

    Args:
        llm: The primary LLM to use.
        fallbacks: Ordered list of fallback LLMs.
        failover_on: Exception types that trigger failover.
            Defaults to ``(ConnectionError, TimeoutError)``.
            Add provider-specific exceptions as needed.
        circuit_breaker: Optional circuit breaker for tracking
            provider health. When provided, unhealthy providers
            are skipped without attempting a request.

    Examples:
        .. code-block:: python

            from llama_index.core.llms import FallbackLLM
            from llama_index.llms.openai import OpenAI
            from llama_index.llms.anthropic import Anthropic

            llm = FallbackLLM(
                llm=OpenAI(model="gpt-4o"),
                fallbacks=[Anthropic(model="claude-sonnet-4-20250514")],
            )
            response = llm.chat([ChatMessage(role="user", content="Hi")])
    """

    llm: SerializeAsAny[BaseLLM] = Field(description="Primary LLM.")
    fallbacks: list[SerializeAsAny[BaseLLM]] = Field(
        description="Ordered list of fallback LLMs."
    )
    failover_on: tuple[type[BaseException], ...] = Field(
        default=DEFAULT_FAILOVER_EXCEPTIONS,
        description="Exception types that trigger failover.",
        exclude=True,
    )
    circuit_breaker: Optional[Any] = Field(
        default=None,
        description="Optional CircuitBreaker instance.",
        exclude=True,
    )

    @classmethod
    def class_name(cls) -> str:
        return "fallback_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """Delegate metadata to the primary LLM."""
        return self.llm.metadata

    def _providers(self) -> list[BaseLLM]:
        """Return all providers in priority order."""
        return [self.llm, *self.fallbacks]

    def _try_providers(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Try calling method_name on each provider in order.

        Skips providers whose circuit breaker is open. Records
        success/failure with the circuit breaker when present.
        Raises the primary provider's exception if all fail.
        """
        providers = self._providers()
        first_error = None

        for provider in providers:
            provider_id = id(provider)

            # Skip providers with open circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.is_available(provider_id):
                logger.debug(
                    "Skipping %s (circuit open)", provider.__class__.__name__
                )
                continue

            try:
                result = getattr(provider, method_name)(*args, **kwargs)
                if self.circuit_breaker:
                    self.circuit_breaker.record_success(provider_id)
                return result
            except BaseException as e:
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)

                if not isinstance(e, self.failover_on):
                    raise

                if first_error is None:
                    first_error = e

                logger.warning(
                    "Provider %s failed with %s: %s. Trying next.",
                    provider.__class__.__name__,
                    type(e).__name__,
                    str(e)[:200],
                )

        # All providers failed: raise the first error
        raise first_error  # type: ignore[misc]

    async def _atry_providers(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Async variant of _try_providers."""
        providers = self._providers()
        first_error = None

        for provider in providers:
            provider_id = id(provider)

            if self.circuit_breaker and not self.circuit_breaker.is_available(provider_id):
                logger.debug(
                    "Skipping %s (circuit open)", provider.__class__.__name__
                )
                continue

            try:
                result = await getattr(provider, method_name)(*args, **kwargs)
                if self.circuit_breaker:
                    self.circuit_breaker.record_success(provider_id)
                return result
            except BaseException as e:
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(provider_id)

                if not isinstance(e, self.failover_on):
                    raise

                if first_error is None:
                    first_error = e

                logger.warning(
                    "Provider %s failed with %s: %s. Trying next.",
                    provider.__class__.__name__,
                    type(e).__name__,
                    str(e)[:200],
                )

        raise first_error  # type: ignore[misc]

    # ===== Sync Endpoints =====

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self._try_providers("chat", messages, **kwargs)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self._try_providers("complete", prompt, formatted=formatted, **kwargs)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        return self._try_providers("stream_chat", messages, **kwargs)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return self._try_providers("stream_complete", prompt, formatted=formatted, **kwargs)

    # ===== Async Endpoints =====

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return await self._atry_providers("achat", messages, **kwargs)

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return await self._atry_providers("acomplete", prompt, formatted=formatted, **kwargs)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        return await self._atry_providers("astream_chat", messages, **kwargs)

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        return await self._atry_providers("astream_complete", prompt, formatted=formatted, **kwargs)
```

### 2.4 Streaming Failover Semantics

Streaming is the hardest case. Two strategies:

**Strategy A (proposed): Fail before yield.** The stream generator is created by calling `provider.stream_chat(messages)`. If the provider is down, the generator creation itself throws (connection refused, timeout on first chunk). `_try_providers` catches this and moves to the next provider. Once a generator starts yielding tokens, mid-stream failures are not caught by the failover layer. They propagate to the caller.

This is what LangChain does. It is simple and predictable: if you get any tokens, the response is committed to that provider.

**Strategy B (future): Mid-stream retry.** Wrap the generator to catch mid-stream errors and restart on the next provider. This requires buffering partial tokens and is complex. Not proposed for v1.

### 2.5 Integration Points

The following table shows where `FallbackLLM` fits in the existing call chain:

| Layer | Component | What it does |
|-------|-----------|-------------|
| Application | `Settings.llm = FallbackLLM(...)` | User configures failover |
| FallbackLLM | `_try_providers("chat", ...)` | Routes to available provider |
| CircuitBreaker | `is_available(provider_id)` | Skips unhealthy providers |
| Callback | `@llm_chat_callback()` | Rate limiter acquire + instrumentation |
| Provider retry | `@llm_retry_decorator` (e.g. tenacity) | Provider-level retries with backoff |
| Provider | `openai.ChatCompletion.create()` | Actual API call |

The key ordering: `FallbackLLM` wraps the outer call. Each provider's own retry logic runs inside. So if OpenAI is configured with `max_retries=3`, it retries 3 times with exponential backoff before the exception reaches `FallbackLLM`, which then tries the next provider. This prevents unnecessary failover on transient blips that a single retry would fix.

### 2.6 Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `llama_index/core/llms/fallback_llm.py` | Create | `FallbackLLM` class |
| `llama_index/core/llms/circuit_breaker.py` | Create | `CircuitBreaker` class |
| `llama_index/core/llms/__init__.py` | Modify | Export `FallbackLLM`, `CircuitBreaker` |
| `tests/core/llms/test_fallback_llm.py` | Create | Unit tests for FallbackLLM |
| `tests/core/llms/test_circuit_breaker.py` | Create | Unit tests for CircuitBreaker |

### 2.7 Test Plan

**Unit tests (mocked, fast):**
- Primary succeeds: returns primary's response, no fallback attempted
- Primary fails with failover exception: falls back to secondary, returns secondary's response
- Primary fails with non-failover exception (e.g. `ValueError`): raises immediately, no fallback
- All providers fail: raises the primary's exception
- Circuit breaker skips unhealthy provider
- Circuit breaker half-open: allows probe request after recovery timeout
- Circuit breaker resets on success
- Streaming: failover on generator creation error
- Async variants for all of the above
- `metadata` delegates to primary LLM
- Serialization round-trip (Pydantic model_dump/model_validate)

**Integration tests (real providers, manual):**
- OpenAI primary + Anthropic fallback: kill OpenAI API key, verify Anthropic serves requests
- Rate limit scenario: verify failover triggers on 429 after provider retries are exhausted

### 2.8 Rate Limiter Interaction

Each provider in a `FallbackLLM` can have its own `rate_limiter`. The `FallbackLLM` itself does not need a rate limiter because it delegates to providers who handle their own throttling:

```python
openai_limiter = TokenBucketRateLimiter(requests_per_minute=100)
anthropic_limiter = TokenBucketRateLimiter(requests_per_minute=50)

Settings.llm = FallbackLLM(
    llm=OpenAI(model="gpt-4o", rate_limiter=openai_limiter),
    fallbacks=[Anthropic(model="claude-sonnet-4-20250514", rate_limiter=anthropic_limiter)],
)
```

---

## Design Decisions

| Decision | Chosen | Alternatives | Rationale |
|----------|--------|-------------|-----------|
| Class hierarchy | `FallbackLLM(LLM)` | Standalone wrapper, mixin | Follows `StructuredLLM` pattern, works with `Settings.llm` |
| Exception filter | User-configurable tuple | Catch all, hardcoded list | Different providers have different exception hierarchies |
| Default exceptions | `(ConnectionError, TimeoutError)` | `(Exception,)` | Conservative default; catches network issues without masking bugs |
| Circuit breaker | Optional separate class | Built into FallbackLLM, no circuit breaker | Separation of concerns; users who don't need it pay no complexity |
| Provider identity | `id(llm)` | String name, model name | Handles same provider class with different configs |
| Streaming failover | Fail before first yield only | Mid-stream retry | Simple, predictable; matches LangChain behavior |
| All-fail behavior | Raise primary's exception | Raise last exception, aggregate | Primary's error is most informative to the user |

---

## Future Work

- **`FallbackEmbedding`**: Same pattern applied to `BaseEmbedding`. Separate PR to keep scope focused.
- **Load-balanced routing**: Round-robin or weighted distribution across healthy providers. Requires different abstraction (router, not failover).
- **Cost-aware routing**: Pick the cheapest available provider. Requires token cost metadata per provider.
- **Distributed circuit breaker**: Redis-backed state for multi-process deployments.
- **Metrics and observability**: Emit structured events for failover attempts, circuit state changes.
- **Mid-stream retry**: Buffer partial streaming responses and retry from scratch on a different provider if the stream fails mid-response.

---

## References

1. [LlamaIndex Issue #19631: Built-in LLM Failover for Reliability](https://github.com/run-llama/llama_index/issues/19631)
2. [LlamaIndex Issue #15649: Indexing hit rate limit error and keeps endless retrying](https://github.com/run-llama/llama_index/issues/15649)
3. [LlamaIndex PR #20712: Token-bucket rate limiter](https://github.com/run-llama/llama_index/pull/20712)
4. [LlamaIndex PR #20813: Respect Retry-After header in retry decorator](https://github.com/run-llama/llama_index/pull/20813)
5. [LangChain RunnableWithFallbacks](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.fallbacks.RunnableWithFallbacks.html)
6. [LangChain fallbacks.py source](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/runnables/fallbacks.py)
7. [Semantic Kernel Error Handling and Resilience](https://skgraph.dev/how-to/error-handling-and-resilience/)
8. [LlamaIndex StructuredLLM source](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/llms/structured_llm.py)
9. [LlamaIndex BaseLLM source](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/llms/base.py)
10. [LlamaIndex TokenBucketRateLimiter source](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/rate_limiter.py)
