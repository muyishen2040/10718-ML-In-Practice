"""Optional notebook-friendly progress bars with a dependency-free fallback."""

from __future__ import annotations

from typing import Any


class _NoOpProgress:
    def update(self, _: int | float = 1) -> None:
        return None

    def set_postfix(self, **_: Any) -> None:
        return None

    def __enter__(self) -> "_NoOpProgress":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def progress(*, total: int | None, description: str, unit: str = "item", **kwargs: Any) -> Any:
    """Return `tqdm.auto` when installed; scripts remain usable without it."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NoOpProgress()
    return tqdm(total=total, desc=description, unit=unit, **kwargs)
