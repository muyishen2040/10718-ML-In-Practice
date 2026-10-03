import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_bm25.py"
SPEC = importlib.util.spec_from_file_location("run_bm25_script", SCRIPT_PATH)
assert SPEC and SPEC.loader
RUN_BM25 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_BM25)


def test_per_claim_groups_preserve_candidate_boundaries(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    rows = [
        {"passage_id": "c1-p1", "text": "one", "metadata": {"candidate_claim_id": "c1"}},
        {"passage_id": "c1-p2", "text": "two", "metadata": {"candidate_claim_id": "c1"}},
        {"passage_id": "c2-p1", "text": "three", "metadata": {"candidate_claim_id": "c2"}},
    ]
    corpus_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    groups = list(RUN_BM25.per_claim_candidate_groups(corpus_path))

    assert [claim_id for claim_id, _ in groups] == ["c1", "c2"]
    assert [item.passage_id for item in groups[0][1]] == ["c1-p1", "c1-p2"]


def test_checkpoint_rankings_round_trip(tmp_path) -> None:
    checkpoint_path = tmp_path / "dev_rankings.partial.jsonl"
    rankings = [
        {"claim_id": "c1", "claim": "claim one", "retrieved": []},
        {"claim_id": "c2", "claim": "claim two", "retrieved": [{"passage_id": "c2-p1"}]},
    ]

    RUN_BM25.append_checkpoint_rows(checkpoint_path, rankings)
    loaded = RUN_BM25.load_checkpoint_rankings(checkpoint_path, known_claim_ids={"c1", "c2"})

    assert loaded == {"c1": rankings[0], "c2": rankings[1]}


def test_checkpoint_loader_repairs_only_a_truncated_final_record(tmp_path, capsys) -> None:
    checkpoint_path = tmp_path / "dev_rankings.partial.jsonl"
    first = {"claim_id": "c1", "claim": "claim one", "retrieved": []}
    checkpoint_path.write_bytes(
        (json.dumps(first) + "\n" + '{"claim_id": "c2", "retrieved": ').encode("utf-8")
    )

    loaded = RUN_BM25.load_checkpoint_rankings(checkpoint_path, known_claim_ids={"c1", "c2"})

    assert loaded == {"c1": first}
    assert checkpoint_path.read_text(encoding="utf-8") == json.dumps(first) + "\n"
    assert "removed malformed final checkpoint record" in capsys.readouterr().out


def test_checkpoint_loader_rejects_mid_file_corruption(tmp_path) -> None:
    checkpoint_path = tmp_path / "dev_rankings.partial.jsonl"
    checkpoint_path.write_text(
        '{"claim_id": "c1", "retrieved": \n'
        + json.dumps({"claim_id": "c2", "claim": "claim two", "retrieved": []})
        + "\n",
        encoding="utf-8",
    )

    try:
        RUN_BM25.load_checkpoint_rankings(checkpoint_path, known_claim_ids={"c1", "c2"})
    except ValueError as error:
        assert "corrupt before its final record" in str(error)
    else:
        raise AssertionError("Expected middle checkpoint corruption to be rejected")


def test_checkpoint_repair_keeps_valid_rows_and_makes_backup(tmp_path) -> None:
    checkpoint_path = tmp_path / "dev_rankings.partial.jsonl"
    first = {"claim_id": "c1", "claim": "claim one", "retrieved": []}
    second = {"claim_id": "c2", "claim": "claim two", "retrieved": []}
    checkpoint_path.write_text(
        json.dumps(first) + "\nnot-json\n" + json.dumps(second) + "\n", encoding="utf-8"
    )

    report = RUN_BM25.repair_checkpoint_rankings(checkpoint_path, known_claim_ids={"c1", "c2"})

    assert report["repaired"] is True
    assert report["valid_row_count"] == 2
    assert report["dropped_line_numbers"] == [2]
    assert Path(report["backup_path"]).read_text(encoding="utf-8").splitlines()[1] == "not-json"
    assert RUN_BM25.load_checkpoint_rankings(checkpoint_path, known_claim_ids={"c1", "c2"}) == {
        "c1": first,
        "c2": second,
    }


def test_checkpoint_repair_removes_only_identical_duplicate_rows(tmp_path) -> None:
    checkpoint_path = tmp_path / "dev_rankings.partial.jsonl"
    first = {"claim_id": "c1", "claim": "claim one", "retrieved": []}
    second = {"claim_id": "c2", "claim": "claim two", "retrieved": []}
    checkpoint_path.write_text(
        json.dumps(first) + "\n" + json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
    )

    report = RUN_BM25.repair_checkpoint_rankings(checkpoint_path, known_claim_ids={"c1", "c2"})

    assert report["repaired"] is True
    assert report["dropped_line_numbers"] == []
    assert report["dropped_duplicate_line_numbers"] == [2]
    assert RUN_BM25.load_checkpoint_rankings(checkpoint_path, known_claim_ids={"c1", "c2"}) == {
        "c1": first,
        "c2": second,
    }


def test_checkpoint_repair_rejects_conflicting_duplicate_rows(tmp_path) -> None:
    checkpoint_path = tmp_path / "dev_rankings.partial.jsonl"
    checkpoint_path.write_text(
        json.dumps({"claim_id": "c1", "claim": "first", "retrieved": []})
        + "\n"
        + json.dumps({"claim_id": "c1", "claim": "changed", "retrieved": []})
        + "\n",
        encoding="utf-8",
    )

    try:
        RUN_BM25.repair_checkpoint_rankings(checkpoint_path, known_claim_ids={"c1"})
    except ValueError as error:
        assert "conflicting duplicate" in str(error)
    else:
        raise AssertionError("Expected conflicting duplicate rows to be rejected")
