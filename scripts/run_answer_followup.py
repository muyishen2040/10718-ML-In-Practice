"""Prepare resumable claim-only questions and exact-source answer contexts."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("questions", "prepare"):
        p = sub.add_parser(command)
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--model", default="Qwen/Qwen3-0.6B" if command == "questions" else "deepset/roberta-base-squad2")
        p.add_argument("--revision")
        p.add_argument("--device", default="cuda")
        if command == "questions":
            p.add_argument("--train-question-cache", type=Path)
            p.add_argument("--dev-question-cache", type=Path)
            p.add_argument("--max-new-tokens", type=int, default=160)
        else:
            p.add_argument("--parent-prepared", type=Path, required=True)
            p.add_argument("--questions", type=Path, required=True)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--max-length", type=int, default=384)
            p.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            parser.error("Choose a Colab GPU runtime")
    for name in ("batch_size", "max_new_tokens"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.command == "prepare" and (args.limit < 0 or args.max_length < 256):
        parser.error("Use nonnegative limit and max-length >= 256")
    from src.retrieval.answer_followup import prepare_answers, prepare_questions
    {"questions": prepare_questions, "prepare": prepare_answers}[args.command](args)


if __name__ == "__main__":
    main()
