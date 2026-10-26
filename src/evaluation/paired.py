"""Paired uncertainty conditional on the observed class counts."""
import numpy as np

from src.data.averitec import CANONICAL_LABELS
from src.data.evidence import index_claims


def paired_delta(reference, candidate, trials=1000, seed=718):
    a, b = index_claims(reference), index_claims(candidate)
    if set(a) != set(b) or any(a[c]["true_label"] != b[c]["true_label"] for c in a):
        raise ValueError("Paired evaluations must cover identical claims and gold labels")
    keys = sorted(a)
    label = {k: i for i, k in enumerate(CANONICAL_LABELS)}
    y = np.array([label[a[c]["true_label"]] for c in keys])
    old = np.array([label[a[c]["predicted_label"]] for c in keys])
    new = np.array([label[b[c]["predicted_label"]] for c in keys])

    def f1(gold, predicted):
        counts = np.bincount(4 * gold + predicted, minlength=16).reshape(4, 4)
        denom = counts.sum(0) + counts.sum(1)
        return np.divide(2 * counts.diagonal(), denom, out=np.zeros(4), where=denom > 0).mean()

    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == i) for i in range(4)]
    differences = []
    for _ in range(trials):
        sample = np.concatenate([rng.choice(g, len(g)) for g in groups if len(g)])
        differences.append(f1(y[sample], new[sample]) - f1(y[sample], old[sample]))
    return {"delta_macro_f1": float(f1(y, new) - f1(y, old)), "interval_95pct": np.quantile(differences, [.025, .975]).tolist(),
            "changed_predictions": int((old != new).sum()), "fixed": int(((old != y) & (new == y)).sum()),
            "broken": int(((old == y) & (new != y)).sum()),
            "caveat": "Class-stratified paired bootstrap; excludes training randomness, related claims and repeated dev inspection."}
