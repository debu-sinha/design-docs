# Design Documents

My technical design proposals for open source projects. These are personal contributions submitted to various open source communities for review and feedback.

## Documents

| Document | Project | Status | Issue |
|----------|---------|--------|-------|
| [UV Package Manager Support](mlflow/mlflow-uv-support-design-doc-v1.md) | MLflow | Draft | [#12478](https://github.com/mlflow/mlflow/issues/12478) |
| [Usage Tracker](instructor/usage-tracker-design-doc.md) | Instructor | Draft | [#613](https://github.com/jxnl/instructor/issues/613), [#267](https://github.com/jxnl/instructor/issues/267) |

## Structure

```
design-docs/
├── mlflow/                    # MLflow project designs
│   ├── images/
│   └── *.md
├── instructor/                # Instructor project designs
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
