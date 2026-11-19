"""Decisive-evidence pilot sheets and a small selector over saved BM25 pools.

Commands (all inputs relative to your DATA_ROOT; outputs go to a personal run dir):

  sheet  Draw ~20 training-fit claims x 10 candidates for passage-level review.
  audit  Validate a filled sheet (labels, exact quotes, fit-only claims).
  train  Fit the selector on reviewed labels from released fit claims.
  apply  Write three-passage rankings in selector order for any split.

Then feed the applied rankings to the same frozen verifier, e.g.
``python scripts/run_evidence_reader.py infer --rankings OUT/dev_rankings.jsonl ...``.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.evidence import index_claims  # noqa: E402
from src.utils.io import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.utils.progress import progress  # noqa: E402
from src.utils.runs import sha256_file  # noqa: E402


def fit_ids(args, records):
    if args.release_config:
        from src.evaluation.team_release import load_release_fold
        fold = load_release_fold(args.release_config, args.data_root, args.fold_id, records,
                                 train_claims_path=args.claims)
        return set(fold.fit_claim_ids), fold.binding()
    if getattr(args, "provisional_all_train", False):
        return {r["claim_id"] for r in records}, {"provisional": "release pending; reconcile with released fit IDs before training"}
    raise SystemExit("Pass --release-config and --fold-id (or --provisional-all-train for a pre-release pilot sheet)")


def load_inputs(args):
    records = read_jsonl(args.claims)
    rankings = index_claims(read_jsonl(args.rankings))
    return records, rankings


def nli_cache(args):
    if not getattr(args, "prepared_journal", None):
        return None
    from src.retrieval.evidence_selector import nli_cache_from_prepared
    return nli_cache_from_prepared(read_jsonl(args.prepared_journal))


def cmd_sheet(args):
    from src.retrieval.decisive_evidence import build_sheet, write_csv

    records, rankings = load_inputs(args)
    ids, binding = fit_ids(args, records)
    rows, context = build_sheet(records, rankings, ids, claims=args.claims_count, per_claim=args.per_claim, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output / "decisive_sheet.csv")
    write_jsonl(rows, args.output / "decisive_sheet.jsonl")
    write_jsonl(context, args.output / "claim_context_annotators_only.jsonl")
    write_json({"claims": str(args.claims), "claims_sha256": sha256_file(args.claims), "rankings": str(args.rankings),
                "rankings_sha256": sha256_file(args.rankings), "claims_count": len(context), "rows": len(rows),
                "per_claim": args.per_claim, "seed": args.seed, "fit_ids": binding,
                "instructions": "Label each passage alone; quote exact text for direct_support/direct_contradiction; "
                                "record missing qualifications. Set review_status to reviewed only after a second look."},
               args.output / "sheet_config.json")
    print(f"Wrote {len(rows)} candidate rows for {len(context)} claims to {args.output}")


def cmd_audit(args):
    from src.retrieval.decisive_evidence import read_annotations, validate_annotations

    records, rankings = load_inputs(args)
    ids, binding = fit_ids(args, records)
    _, summary = validate_annotations(read_annotations(args.annotations), rankings, ids, allow_silver=args.allow_silver)
    summary["fit_ids"] = binding
    print(json.dumps(summary, indent=2))
    if args.output:
        write_json(summary, args.output)


def cmd_train(args):
    import joblib
    from src.retrieval.decisive_evidence import read_annotations, selector_targets, validate_annotations
    from src.retrieval.evidence_selector import fit_selector, training_matrix

    if not args.release_config:
        raise SystemExit("Selector training needs released fit IDs (--release-config/--fold-id)")
    records, rankings = load_inputs(args)
    ids, binding = fit_ids(args, records)
    usable, audit = validate_annotations(read_annotations(args.annotations), rankings, ids, allow_silver=args.allow_silver)
    targets = selector_targets(usable)
    by_id = {r["claim_id"]: r for r in records}
    x, y, groups, ranks = training_matrix(targets, by_id, rankings, depth=args.depth, nli_cache=nli_cache(args))
    model, report = fit_selector(x, y, groups, ranks, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "selector.joblib")
    write_json({**report, "annotation_audit": audit, "release": binding, "depth": args.depth, "seed": args.seed,
                "annotations_sha256": sha256_file(args.annotations), "rankings_sha256": sha256_file(args.rankings),
                "allow_silver": args.allow_silver, "nli_cache": str(args.prepared_journal) if args.prepared_journal else None,
                "implementation_sha256": sha256_file(PROJECT_ROOT / "src/retrieval/evidence_selector.py")},
               args.output / "selector_report.json")
    print(json.dumps({k: report[k] for k in ("selected_C", "cv_trials", "bm25_order_same_candidates")}, indent=2))


def cmd_apply(args):
    import joblib
    from src.retrieval.evidence_selector import select_passages

    records, rankings = load_inputs(args)
    model = joblib.load(args.selector / "selector.joblib")
    cache = nli_cache(args)
    out = []
    with progress(total=len(records), description="Select evidence", unit="claim") as bar:
        for record in records:
            ranking = rankings.get(record["claim_id"])
            if ranking is None:
                raise ValueError(f"Rankings lack {record['claim_id']}")
            out.append({"claim_id": record["claim_id"], "claim": record["claim"],
                        "retrieved": select_passages(model, record, ranking, depth=args.depth, nli_cache=cache,
                                                     distinct_sources=not args.allow_same_source)})
            bar.update(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, args.output)
    write_json({"selector": str(args.selector), "selector_sha256": sha256_file(args.selector / "selector.joblib"),
                "claims_sha256": sha256_file(args.claims), "rankings_sha256": sha256_file(args.rankings), "depth": args.depth,
                "distinct_sources": not args.allow_same_source, "output_sha256": sha256_file(args.output)},
               args.output.with_suffix(".config.json"))
    changed = sum([p["passage_id"] for p in row["retrieved"]] != [p["passage_id"] for p in (rankings[row["claim_id"]].get("retrieved") or [])[:3]] for row in out)
    print(f"Wrote {len(out)} claims; top three differ from BM25 for {changed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sheet", "audit", "train", "apply"):
        p = sub.add_parser(name)
        p.add_argument("--claims", type=Path, required=True, help="processed/averitec/{train,dev}.jsonl")
        p.add_argument("--rankings", type=Path, required=True, help="Saved BM25 rankings, e.g. train_bm25_top100_v2/train_rankings.jsonl")
        if name != "apply":
            p.add_argument("--data-root", type=Path, help="DATA_ROOT; required with --release-config")
            p.add_argument("--release-config", type=Path)
            p.add_argument("--fold-id")
        if name in ("audit", "train"):
            p.add_argument("--annotations", type=Path, required=True, help="Filled decisive_sheet.csv or .jsonl")
            p.add_argument("--allow-silver", action="store_true", help="Also use unreviewed-but-proposed silver labels")
        if name in ("train", "apply"):
            p.add_argument("--depth", type=int, default=100)
            p.add_argument("--prepared-journal", type=Path, help="Optional prepared {split}.partial.jsonl with text-checked NLI scores")
        p.add_argument("--seed", type=int, default=718)
    sheet = sub.choices["sheet"]
    sheet.add_argument("--output", type=Path, required=True)
    sheet.add_argument("--claims-count", type=int, default=20)
    sheet.add_argument("--per-claim", type=int, default=10)
    sheet.add_argument("--provisional-all-train", action="store_true",
                       help="Before the release: sample from all train claims; rows outside later fit IDs are dropped at audit/train")
    sub.choices["audit"].add_argument("--output", type=Path)
    sub.choices["audit"].add_argument("--provisional-all-train", action="store_true",
                                      help="Audit a pre-release sheet against all train claims (training still needs the release)")
    sub.choices["train"].add_argument("--output", type=Path, required=True)
    apply = sub.choices["apply"]
    apply.add_argument("--selector", type=Path, required=True, help="Directory containing selector.joblib")
    apply.add_argument("--output", type=Path, required=True, help="Output rankings JSONL, e.g. OUT/selected/dev_rankings.jsonl")
    apply.add_argument("--allow-same-source", action="store_true")
    args = parser.parse_args()
    if getattr(args, "release_config", None) and not (args.fold_id and args.data_root):
        parser.error("--release-config needs --fold-id and --data-root")
    if getattr(args, "depth", 3) < 3:
        parser.error("--depth must be at least 3")
    {"sheet": cmd_sheet, "audit": cmd_audit, "train": cmd_train, "apply": cmd_apply}[args.command](args)


if __name__ == "__main__":
    main()
