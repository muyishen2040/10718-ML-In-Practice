# Pilot run: checklist

Purpose: find problems in the study page and procedure with 2-3 people before recruiting.
Design and rationale: [USER_STUDY_DESIGN.md](USER_STUDY_DESIGN.md). Pilot data is not used in results.

## Before you start

1. Pick participants who have not seen the claims, and give each a code `P01-<group>`
   (for example `P01-1`, `P02-2`, `P03-3`; the number after the dash is the group, 1-4).
   Use different groups for the pilot people, so that both block orders and both claim sets get tried.
2. Give them the page, either way:
   - Zip and send: `zip -r claimcheck_study.zip apps/study -x "apps/study/__pycache__/*"`. They unzip it, open
     `index.html` in Chrome or Safari, and type their code on the first screen.
   - Or sit next to them and use your own laptop with the link
     `index.html?g=1&p=P01` (open from disk, or `cd apps/study && python3 -m http.server 8766`).
3. Tell them: about 20 minutes, no searching, there are no trick questions, a made-up practice claim comes first.

## What to watch for

- Do they understand the four answer options? Do they hesitate between "Not enough evidence" and "Conflicting"?
- Do they find and open the sources? Do they read the caution line about the model?
- Is anything confusing in the instructions, the block-change screen or the survey?
- Total time (target about 20 minutes). Any claim that took far longer than the rest.
- Does it work on their browser and screen size, including a phone if you expect that?
- Ask afterwards: what did you think the tool was? Did you trust it? What would you change?

## After they finish

They click "Download my answers" on the last screen and send you the file `claimcheck_<code>.json`.
Put the files in one folder and run:

```bash
python3 scripts/analyze_study_results.py path/to/folder --csv path/to/folder/all.csv
```

It marks each file OK or BAD (missing trials, wrong condition for the group, no answer, no survey),
joins gold labels from `local/baseline/final_dev_predictions.jsonl` (needs the shared baseline file),
and writes one row per judgement. With 2-3 people the printed summary is only a pipeline check.

## Decide after the pilot

- Wording changes to instructions or the consent text.
- Whether 12 claims is the right length.
- Whether to replace any claim (too familiar, confusing evidence).
- Whether to receive data automatically (see "Collecting responses" in the design doc) before recruiting.
