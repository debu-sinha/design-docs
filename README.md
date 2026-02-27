# Design Documents

My technical design proposals for open source projects. These are personal contributions submitted to various open source communities for review and feedback.

## Documents

| Document | Project | Status | Issue |
|----------|---------|--------|-------|
| [UV Package Manager Support v2](mlflow/uv-support-design-doc-v2.md) | MLflow | In Review | [#12478](https://github.com/mlflow/mlflow/issues/12478), [PR #20344](https://github.com/mlflow/mlflow/pull/20344) |
| [UV Package Manager Support v1](mlflow/mlflow-uv-support-design-doc-v1.md) | MLflow | Superseded | [#12478](https://github.com/mlflow/mlflow/issues/12478) |
| [uv Compatibility](guardrails/guardrails-uv-compatibility.md) | Guardrails AI | Implemented | [#1392](https://github.com/guardrails-ai/guardrails/issues/1392), [PR #1416](https://github.com/guardrails-ai/guardrails/pull/1416) |
| [Usage Tracker](instructor/usage-tracker-design-doc.md) | Instructor | Closed | [#2080](https://github.com/567-labs/instructor/issues/2080) (completed by maintainers) |
| [Online Evals](phoenix/phoenix-online-evals-design-doc.md) | Phoenix (Arize) | Draft | [#11642](https://github.com/Arize-ai/phoenix/issues/11642) |
| [LLM Failover and Circuit Breaker](llamaindex/llm-failover-circuit-breaker.md) | LlamaIndex | Draft | [#19631](https://github.com/run-llama/llama_index/issues/19631) |

## Structure

```
design-docs/
├── mlflow/                    # MLflow project designs
│   ├── images/
│   └── *.md
├── guardrails/                # Guardrails AI project designs
│   ├── images/
│   └── *.md
├── instructor/                # Instructor project designs
│   └── *.md
├── phoenix/                   # Phoenix (Arize) project designs
│   ├── images/
│   └── *.md
├── llamaindex/                # LlamaIndex project designs
│   ├── images/
│   └── *.md
└── README.md
```

## Document Template

Each design document follows a standard structure:

1. **Metadata** - Author, status, related issues
2. **Part I: Design Sketch** - Motivation, requirements, proposal overview, alternatives
3. **Part II: Detailed Design** - Implementation details, code examples, testing
4. **Appendix** - Design decisions, references

## Status Definitions

| Status | Description |
|--------|-------------|
| Draft | Initial proposal, seeking feedback |
| In Review | Under active review by maintainers |
| Approved | Accepted, ready for implementation |
| Implemented | Merged into target project |
| Rejected | Not accepted (with rationale) |

## Author

Debu Sinha ([@debu-sinha](https://github.com/debu-sinha))
