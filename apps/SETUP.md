# Run the prototype (development only)

Status: mock interface plus a fixture-replay backend. No ML model, retrieval or
hosted API is connected yet. Contract: [INTEGRATION_CONTRACT.md](../docs/INTEGRATION_CONTRACT.md).

## Requirements

Chrome or another Chromium browser; Python 3.9+. The backend uses only the
standard library, and the extension has no build step. Set no environment
variables and no secrets are needed.

## Start the backend

```bash
python3 apps/backend/server.py            # http://127.0.0.1:8000
python3 -m unittest discover -s apps/backend/tests
```

Routes: `GET /health`, `GET /benchmark/claims`, `POST /verify`.

## Load the extension

1. Open `chrome://extensions`, enable Developer mode.
2. Choose **Load unpacked** and select `apps/browser-extension/`.
3. Select text on any normal web page, click the extension icon, press
   **Check this claim**. Chrome's own pages block selection; paste instead.

## Three result sources (always labeled in the popup)

| Mode | Needs backend | What it does |
| --- | --- | --- |
| Mock | no | Built-in synthetic samples for every UI state. Not a check. |
| Benchmark demo | yes | Replays saved fixture results by claim ID. Not new inference. |
| Live | yes | Returns "not enough sources": no retrieval path exists for new claims. |

## Known limits

- Fixtures in `apps/backend/fixtures/demo_claims.json` are synthetic (example.com
  URLs), not AVeriTeC records. Replace with saved benchmark predictions once B or C
  provides them, via an `Adapter` in `apps/backend/server.py`.
- The backend adds an optional `provenance` field (`fixture_replay`) to
  responses. This is an additive proposal to the contract, not yet agreed.
- Nothing here shows that the interface improves user decisions.
