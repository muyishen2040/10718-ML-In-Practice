# Interface and backend placement

This directory is the home for the product layer. No extension or backend has
been implemented by this organization change. Proposed implementation paths:

```text
apps/
  browser-extension/    Teammate A: browser client and results panel
  backend/              Teammate A: HTTP boundary and integration
```

The local reader and hosted-provider adapters stay in shared Python code under
`src/verification/`; retrieval stays under `src/retrieval/`. The backend calls
those components instead of duplicating their model/evaluation logic inside the
extension. Teammates B and C supply adapters to A's common boundary.

Follow [the integration contract](../docs/INTEGRATION_CONTRACT.md) and
[A's task brief](../docs/tasks/TEAMMATE_A_INTERFACE.md). Declare frontend/backend
dependencies when actual implementations are added. Credentials remain on the
backend, and a mounted personal Drive is a development input location rather
than an assumed deployment filesystem.
