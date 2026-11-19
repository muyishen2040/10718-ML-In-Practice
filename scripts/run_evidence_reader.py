"""CLI for the evidence-quality and NLI pair-reader pipeline."""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("train-claims", "dev-claims", "train-rankings", "dev-rankings", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--model", default="cross-encoder/nli-deberta-v3-small")
    prep.add_argument("--revision")
    prep.add_argument("--nli-label-order", help="Documented model logit order; auto-detected when config names are explicit")
    prep.add_argument("--max-candidates", type=int, default=24)
    prep.add_argument("--candidate-depth", type=int, default=100)
    prep.add_argument("--max-length", type=int, default=384)
    prep.add_argument("--claim-budget", type=int, default=96)
    prep.add_argument("--batch-size", type=int, default=16)
    prep.add_argument("--device", default="cuda")
    prep.add_argument("--limit", type=int, default=0, help="Smoke test only; use a separate output directory")
    for command in ("fast", "train", "frozen", "compare"):
        p = sub.add_parser(command)
        for name in ("data-root", "prepared", "output"):
            p.add_argument("--" + name, type=Path, required=True)
        if command == "compare":
            p.add_argument("--qrels", type=Path)
            p.add_argument("--reference-root", type=Path)
            continue
        modes = ("raw", "clean", "clean_nli", "qa_same3", "qa_selected3") if command == "train" else ("raw", "clean", "clean_nli")
        p.add_argument("--mode", choices=modes, required=True)
        if command == "frozen":
            p.add_argument("--baseline", type=Path, required=True)
            p.add_argument("--device", default="cuda")
            continue
        p.add_argument("--validation-policy", choices=("grouped", "stratified"), default="grouped")
        p.add_argument("--validation-seed", type=int, default=718)
        p.add_argument("--selection-only", action="store_true")
        if command == "train":
            p.add_argument("--learning-rates", type=float, nargs="+", default=[1e-5, 3e-5])
            p.add_argument("--epochs", type=int, default=3)
            p.add_argument("--batch-size", type=int, default=2)
            p.add_argument("--accumulation", type=int, default=8)
            p.add_argument("--seed", type=int, default=718)
            p.add_argument("--distillation-weight", type=float, default=.1)
            p.add_argument("--loss-normalization", choices=("sample_mean", "batch_weight_mean"), default="sample_mean")
            p.add_argument("--unfreeze-embeddings", action="store_true", help="Also train vocabulary embeddings; increases optimizer memory and checkpoint size")
            p.add_argument("--device", default="cuda")
            p.add_argument("--release-config", type=Path, help="configs/team_evaluation.json; trains on a released fit fold instead of an internal holdout")
            p.add_argument("--fold-id", help="Released fold to fit and score; required with --release-config")
            p.add_argument("--fold-epoch-policy", choices=("fixed", "best"), default="fixed",
                           help="fixed: report the last epoch (no validation selection); best: select epoch on fold validation")
    infer = sub.add_parser("infer", help="Frozen saved reader on any claims + three-passage rankings (input intervention)")
    infer.add_argument("--model", type=Path, required=True, help="Saved final_model directory")
    infer.add_argument("--claims", type=Path, required=True)
    infer.add_argument("--rankings", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--preparation", type=Path, help="preparation.json supplying max_length/claim_budget used in training")
    infer.add_argument("--batch-size", type=int, default=8)
    infer.add_argument("--device", default="cuda")
    mech = sub.add_parser("mechanics", help="Inspect a saved reader run and optionally re-score saved predictions")
    mech.add_argument("--run", type=Path, required=True, help="Saved pair-reader run directory, e.g. .../pair/raw_sample_d01")
    mech.add_argument("--output", type=Path, required=True, help="Personal output directory for the mechanics report")
    mech.add_argument("--data-root", type=Path, help="Needed to join saved predictions to claims/rankings")
    mech.add_argument("--prepared", type=Path, help="Prepared evidence directory used by the run")
    mech.add_argument("--rescore-limit", type=int, default=0, help="Re-run frozen inference on the first N dev claims and compare probabilities")
    mech.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for name in ("max_candidates", "candidate_depth", "batch_size", "epochs", "accumulation"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(name + " must be positive")
    if args.command == "prepare":
        if args.max_length < args.claim_budget + 32 or args.claim_budget < 1 or args.limit < 0:
            parser.error("Invalid text length or smoke-test limit")
        from src.retrieval.evidence_pipeline import prepare_pipeline
        prepare_pipeline(args)
    elif args.command == "infer":
        from src.verification.local_reader import infer_rankings
        print(infer_rankings(args)["macro_f1"])
    elif args.command == "mechanics":
        if args.rescore_limit < 0 or (args.rescore_limit and not (args.data_root and args.prepared)):
            parser.error("--rescore-limit needs --data-root and --prepared and must be non-negative")
        from src.verification.reader_mechanics import mechanics_check
        mechanics_check(args)
    elif args.command == "train":
        if any(lr <= 0 for lr in args.learning_rates) or args.distillation_weight < 0:
            parser.error("Invalid learning rates or distillation weight")
        if bool(args.release_config) != bool(args.fold_id):
            parser.error("--release-config and --fold-id must be given together")
        from src.verification.pair_training import train_reader
        train_reader(args)
    else:
        from src.verification.evidence_experiments import fast_aggregate, frozen_distil, compare_results
        {"fast": fast_aggregate, "frozen": frozen_distil, "compare": compare_results}[args.command](args)


if __name__ == "__main__":
    main()
