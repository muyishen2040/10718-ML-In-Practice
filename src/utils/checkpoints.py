"""Small atomic artifacts and interruption-safe append-only JSONL journals."""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.io import write_json, write_jsonl


def atomic_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    write_jsonl(rows, temporary)
    temporary.replace(path)


def bind_run(path: Path, config: dict) -> None:
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError(f"Inputs or settings changed; use a new run directory: {path.parent}")
    write_json(config, path)


def read_journal(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    if not path.exists():
        return {}
    found = {}
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    last = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
    offset = 0
    repaired = False
    for i, raw in enumerate(lines):
        start = offset
        offset += len(raw)
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if i != last:
                raise ValueError(f"Corrupt checkpoint before final line: {path}") from error
            with path.open("r+b") as handle:
                handle.truncate(start)
            repaired = True
            break
        cid = str(row.get("claim_id") or "")
        if cid not in expected_ids or cid in found:
            raise ValueError(f"Unexpected or duplicate journal claim ID: {cid!r}")
        found[cid] = row
    if not repaired and data and not data.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return found


def append_journal(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def latest_complete_trainer_checkpoint(directory: Path) -> str | None:
    candidates = []
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    for path in directory.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if (any(path.glob("model*.safetensors")) or any(path.glob("pytorch_model*.bin"))) and all((path / k).is_file() for k in required):
            candidates.append((step, path))
    return str(max(candidates, key=lambda p: p[0])[1]) if candidates else None
