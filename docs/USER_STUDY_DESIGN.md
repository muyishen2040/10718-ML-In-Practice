# User study design (pilot draft)

Owner: Teammate A (interface). Status: draft for the team, pilot not yet run. Written 2026-11-14.
Course feedback this responds to: the real test of the system is whether users decide not to
trust claims that are refuted; retrieval and classifier metrics are only proxies. Nothing here
claims a benefit yet; a working page is not evidence that the interface helps.

## Question

Does seeing the screening tool (a preliminary verdict plus three source passages) change whether
people correctly decide to **accept** a claim (gold label Supported) or **withhold trust** (gold label
Refuted, Not Enough Evidence or Conflicting Evidence)?

Secondary: confidence, time per claim, how often a wrong verdict is followed (over-reliance),
whether participants open the sources, and short survey ratings.

## Design

- Within-subject, two conditions. **Unassisted**: the claim only. **Assisted**: the claim plus the
  tool panel (same layout as the browser extension popup, including its caution that the baseline agreed
  with reference labels about half the time). Both conditions forbid searching the web. Reason: the
  benchmark claims come from fact-checking sites, so a web search would usually reveal the answer for
  both conditions. The study therefore simulates a person who cannot or will not look elsewhere.
- Each participant sees 12 claims: 6 unassisted, 6 assisted, in two blocks of six, after one made-up
  practice claim. Order within a block is shuffled with a seed from the participant ID.
- Counterbalancing: claims are split into sets A and B (6 each). Four groups:

| Group | Tool block first? | Claim set shown with the tool | Set shown without it |
| --- | --- | --- | --- |
| 1 | yes | A | B |
| 2 | no | A | B |
| 3 | yes | B | A |
| 4 | no | B | A |

  Every claim is seen with and without the tool across participants, and block order is balanced.
  Give each participant a link with `?g=<1-4>&p=<id>` (target 6 people per group, 24 total; 20-30 is the
  range in the proposal). Without `g` the page picks a group at random, which does not stay balanced.
- Answer per claim: one of the four labels, plus confidence 1-5. Time is recorded from display to
  answer. Which sources were opened is logged. After the tasks: three 5-point items about the tool,
  one optional comment. One background item (how often people check claims).
- Single model for the whole study: the saved baseline (DistilBERT trained on BM25 top-3; 0.52
  accuracy and 0.347 macro-F1 on the 500 official dev claims). Its outputs are replayed from saved
  predictions, so the interface labels them "Saved model output (not a live check)". If B's new model
  becomes the study model, regenerate the 12 outputs (below) and say so in the report.

## The 12 claims

Chosen with `scripts/select_study_claims.py` (seed 718; eligible = three passages of at least 80
characters, claim of 40-220 characters), then curated by hand from three candidates per (gold label x
model output) cell. IDs and sets are in `configs/user_study_claims.json`. The aim is about half
correct, matching the baseline's dev accuracy, so over-reliance can be observed. This is a deliberate
sample, not a random sample of dev, and gold labels were used to choose it. Gold labels are not in the
participant-facing files. The page shows source URLs as plain text, not links, so participants are
not drawn into searching.

| Claim | Set | Gold | Model | Model right (4-way) | Claim |
| --- | --- | --- | --- | --- | --- |
| 00058 | A | Conflicting | Refuted | no | Russia sent medical aid to the US, in the form of medical equipment to… |
| 00093 | A | Supported | Refuted | no | US Judge Amy Coney Barrett graduated at the top of her law school clas… |
| 00112 | A | Refuted | Supported | no | The Median White Household Wealth In The US Is 81 Times that of The Bl… |
| 00141 | A | Refuted | Refuted | yes | China seek for court’s approval to kill the over 20,000 coronavirus pa… |
| 00257 | A | Supported | Supported | yes | A Maryland man was sentenced to a year in jail for throwing parties. |
| 00375 | A | Not enough evidence | Not enough evidence | yes | Nigeria was the third most terrorised in the world in 2020, after Afgh… |
| 00047 | B | Conflicting | Conflicting | yes | Joe Biden said he wants to ban fracking in the US. |
| 00135 | B | Not enough evidence | Supported | no | No one lost their health care because of the Wellmark and Iowa Farm Bu… |
| 00246 | B | Refuted | Conflicting | no | A “new virus” disease called Novel Bunya V!rus that is caused by ticks… |
| 00357 | B | Supported | Supported | yes | Edo State in Nigeria has a 19% unemployment rate. |
| 00467 | B | Refuted | Refuted | yes | Suresh Raina, the Chennai Super Kings (CSK) cricketer, has withdrawn f… |
| 00499 | B | Refuted | Supported | no | The first night of the US Republican National Convention had 128.4 mil… |

Under the binary decision (accept vs not accept) the model is wrong on four of the twelve:
three false reassurances (00112, 00499, 00135: it says Supported, gold is not) and one false alarm
(00093). 00058 and 00246 are wrong labels but the right decision. Report both views.

Known weaknesses of this selection: all claims are from 2020 (few participants will remember them,
which helps, but some political ones are familiar); several have weak passages
(00375, 00112); the correct-versus-wrong mix is chosen, so these results show how people behave
with a tool that is wrong this often, not a deployment estimate.

## Measures and analysis plan (for the analysis owner)

- Primary: decision correct = (answered Supported and gold Supported) or (answered anything else and
  gold not Supported). Compare assisted vs unassisted. Split by gold label, and report false
  reassurance (accepted a non-Supported claim) and false alarm separately.
- Secondary: four-way accuracy; confidence and calibration; time; over-reliance = share of
  assisted trials where the answer equals a wrong model verdict, against the unassisted rate for the
  same claims; sources opened; survey items.
- With about 24 people and 288 assisted/unassisted judgments, favour effect sizes with confidence
  intervals (for example a logistic mixed model with participant and claim effects) over
  significance tests, and say that claim-level variation is large with only 12 claims.

## Running it

```bash
cd apps/study && python3 -m http.server 8766    # then open http://127.0.0.1:8766/index.html?g=1&p=P01
```

The page also works by opening `index.html` from disk. To host it for remote participants, any static
host works (for example GitHub Pages from the branch). Data is in `study_claims.js`.
To regenerate after changing the model or the selection:

```bash
local/venv-backend/bin/python scripts/select_study_claims.py build
```

The build checks that the adapter reproduces the logged dev prediction for every claim and fails otherwise.

### Collecting responses

The page POSTs each finished session to a Google Apps Script web app that writes to a Google Sheet
(`apps/study/apps_script/Code.gs`; deployment steps in [USER_STUDY_PILOT.md](USER_STUDY_PILOT.md)). The
Sheet gets a `trials` tab (one row per judgement), a `participants` tab (group, survey, comment) and a
`raw` tab (full JSON per person, which `scripts/analyze_study_results.py --raw-csv` reads). The page also
always offers a file download as a fallback. No names, emails, IP addresses or browser strings are stored.
The shared token in `study.js` only keeps casual junk out; it is visible in the page source and is not a secret.

## Before recruiting

1. Confirm with course staff whether approval or an exemption is needed, and finalize the consent text
   (the page shows draft wording).
2. Pilot with 2-3 people: check instructions, total time (target about 20 minutes), that all fields
   are saved, and whether the practice claim is clear. Pilot data is not used in results.
3. Decide the study model. If it changes, regenerate outputs and update the table above.
4. Fix the analysis plan before looking at results.

## Not covered

- The selection-to-check flow inside a web page (the study page replaces the extension for remote use).
- Live claims: there is no retrieval path, so every claim is a benchmark claim.
- Any claim that the tool improves decisions; that depends on the results.
