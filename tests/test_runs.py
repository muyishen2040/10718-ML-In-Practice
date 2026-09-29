import pytest

from src.utils.runs import resolve_run_directory


def test_run_name_creates_an_isolated_child_directory(tmp_path) -> None:
    assert resolve_run_directory(tmp_path, "dev_bm25_v1") == tmp_path / "dev_bm25_v1"


def test_run_name_rejects_path_segments(tmp_path) -> None:
    with pytest.raises(ValueError, match="run_name"):
        resolve_run_directory(tmp_path, "../outside")
