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
