# Design Documents

Technical design documents for open source contributions and feature proposals.

## Documents

| Document | Project | Status | Issue |
|----------|---------|--------|-------|
| [UV Package Manager Support](mlflow/mlflow-uv-support-design-doc-v1.md) | MLflow | Draft | [#12478](https://github.com/mlflow/mlflow/issues/12478) |

## Structure

```
design-docs/
├── mlflow/                    # MLflow project designs
│   ├── images/
│   └── *.md
├── <project>/                 # Future project designs
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
