# Teammate C live synthetic smoke test

- Owner is Teammate C. Run date is November 16, 2026.
- The run completed after resuming with available API credits.
- The local branch is `team/teammate-c-api` based on commit
  `0a3a8585ab07759a77f5b8c597b3243ebd769c8b`.
  The runner used source SHA-256
  `52f498f7ecbe1fdc66ee496d2329383de85633767fd1d262f5921ca1d7eedbe8`.

## Inputs and run settings

The run used three synthetic claims, including one with an empty evidence list.
The shared dataset and its evaluation split were outside the scope of this run.
The model was `gpt-5.4-mini-2026-03-17` with standard service tier,
reasoning effort `none`, temperature 0, and a 512 completion-token ceiling.
The runner reserved up to ten billable attempts across restarts with zero
automatic retries. It used the supplied fixture passages with external search disabled.

## Artifacts and results

The run directory is `local/pilot/runs/synthetic-smoke-v1`. The binding file
records the input hashes and full run settings. The request, failure, and
prediction journals are in the same directory. A portable summary with
per-claim verdicts, citation IDs, timing, usage, configuration and hashes is
included in [the recorded results](2026-11-16_teammate-c_synthetic-smoke-results.json).

- The first invocation encountered HTTP 429 with `credit_balance_exhausted`
  after 2.59 seconds. It saved `synthetic:3` locally as **Not Enough Evidence**.
- The resumed invocation reused that prediction. OpenAI returned **Supported**
  for `synthetic:1` in 3.33 seconds and **Refuted** for `synthetic:2` in 1.22
  seconds. Both hosted outputs passed citation validation and matched the
  expected synthetic verdicts.
- Coverage was three of three with zero unresolved errors. The journal
  retained one initial billing failure and recorded two successful API
  requests. They used 755 input and 133 output tokens. Applying [OpenAI's
  standard model rates](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
  gives an estimated usage charge of $0.00116475.

## Reproduction

With Python 3.11 or newer, install `requirements/hosted.txt` and run
`python scripts/run_hosted_pilot.py` for the offline preflight.
Use `python scripts/run_hosted_pilot.py --check-key` to check model access
and `python scripts/run_hosted_pilot.py --live` to generate replies.
The same run name reuses all three saved predictions in the original checkout.

The packaged runner and verifier test suite passed 133 tests with four optional neural
tests skipped. The separate local branch `team/teammate-c-interface-check`
also exercised A's backend and popup against six synthetic scenarios on
November 16. That check used a popup page served locally.
The formal validation and official-dev comparison await the shared split.
