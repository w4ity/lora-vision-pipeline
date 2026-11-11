"""Head-to-head comparison of two runs on the same test manifest.

Produces: overall metrics, per-class F1 deltas, McNemar-style exact test on
paired disagreements, per-example fixed/regressed lists, and a stress benchmark
comparison. This is the report that backs every improvement claim.
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision_pipeline.data.datasets import Manifest, ManifestDataset
from vision_pipeline.data.transforms import build_eval_transform
from vision_pipeline.evaluation.corrupt import CORRUPTIONS, CorruptedDataset
from vision_pipeline.evaluation.evaluate import load_trained
from vision_pipeline.utils import get_logger, md_table

log = get_logger()


@torch.inference_mode()
def _collect(model, dataset, device: torch.device, batch_size: int, num_workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    logits_all, labels_all = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(y.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test (McNemar simplification with p=0.5)."""
    if n == 0:
        return 1.0
    probs = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = abs(k - n / 2)
    return float(sum(p for i, p in enumerate(probs) if abs(i - n / 2) >= obs - 1e-9))


def compare_runs(
    run_a: str | Path,
    run_b: str | Path,
    split: str = "test",
    severities: list[int] | None = None,
    device: str = "auto",
    num_workers: int = 2,
    max_examples: int = 25,
    out_root: str | Path | None = None,
) -> Path:
    severities = severities or [1, 3, 5]
    dev = torch.device(device) if device != "auto" else None
    model_a, classes_a, cfg_a, meta_a = load_trained(run_a, dev)
    model_b, classes_b, cfg_b, meta_b = load_trained(run_b, dev)
    name_a, name_b = cfg_a.name, cfg_b.name

    if cfg_a.data.name != cfg_b.data.name or classes_a != classes_b:
        raise RuntimeError("Both runs must be trained on the same dataset/classes.")
    if cfg_a.data.img_size != cfg_b.data.img_size:
        raise RuntimeError("Runs use different img_size; not comparable.")
    if cfg_a.data.root != cfg_b.data.root:
        raise RuntimeError("Runs use different data roots; manifests may differ.")

    manifest = Manifest.load(
        Path(cfg_a.data.root) / "manifests" / cfg_a.data.name / f"{split}.json"
    )
    ds = ManifestDataset(manifest, build_eval_transform(cfg_a.data.img_size))

    logits_a, y_true = _collect(model_a, ds, model_a_head_dev(model_a), 64, num_workers)
    logits_b, _ = _collect(model_b, ds, model_a_head_dev(model_b), 64, num_workers)

    pred_a, pred_b = logits_a.argmax(1), logits_b.argmax(1)
    correct_a, correct_b = pred_a == y_true, pred_b == y_true

    per_class_f1_a = {
        c: v["f1"] for c, v in _summary(logits_a, y_true, classes_a)["per_class"].items()
    }
    per_class_f1_b = {
        c: v["f1"] for c, v in _summary(logits_b, y_true, classes_b)["per_class"].items()
    }

    fixed_idx = np.where(~correct_a & correct_b)[0]
    regressed_idx = np.where(correct_a & ~correct_b)[0]
    p_value = binom_two_sided(int(len(fixed_idx)), int(len(fixed_idx) + len(regressed_idx)))

    # ---- stress comparison ---- #
    raw_ds = ManifestDataset(manifest, transform=None)
    stress_rows = []
    for corruption in CORRUPTIONS:
        for sev in severities:
            ds_c = CorruptedDataset(raw_ds, corruption, sev, build_eval_transform(cfg_a.data.img_size))
            la, ly = _collect(model_a, ds_c, model_a_head_dev(model_a), 64, num_workers)
            lb, _ = _collect(model_b, ds_c, model_a_head_dev(model_b), 64, num_workers)
            stress_rows.append({
                "corruption": corruption, "severity": sev,
                f"acc_{name_a}": round(float((la.argmax(1) == ly).mean()), 4),
                f"acc_{name_b}": round(float((lb.argmax(1) == ly).mean()), 4),
            })

    # ---- report ---- #
    out_dir = Path(out_root or Path(cfg_a.output_dir)) / "compare" / f"{name_a}__vs__{name_b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def examples(idx: np.ndarray, n: int) -> list[dict]:
        out = []
        for i in idx[:n]:
            out.append({
                "path": ds.paths[i],
                "label": classes_a[int(y_true[i])],
                f"pred_{name_a}": classes_a[int(pred_a[i])],
                f"pred_{name_b}": classes_b[int(pred_b[i])],
            })
        return out

    overall = [
        ["accuracy", round(float(correct_a.mean()), 4), round(float(correct_b.mean()), 4),
         round(float(correct_b.mean() - correct_a.mean()), 4)],
        ["macro_f1", round(_summary(logits_a, y_true, classes_a)["macro_f1"], 4),
         round(_summary(logits_b, y_true, classes_b)["macro_f1"], 4), ""],
        ["top5_accuracy", round(float((logits_a.argsort(1)[:, -5:] == y_true[:, None]).any(1).mean()), 4),
         round(float((logits_b.argsort(1)[:, -5:] == y_true[:, None]).any(1).mean()), 4), ""],
    ]
    per_class_rows = [
        [c, per_class_f1_a[c], per_class_f1_b[c], round(per_class_f1_b[c] - per_class_f1_a[c], 4)]
        for c in classes_a
    ]
    stress_table = md_table(
        ["corruption", "severity", f"acc {name_a}", f"acc {name_b}"],
        [[r["corruption"], r["severity"], r[f"acc_{name_a}"], r[f"acc_{name_b}"]] for r in stress_rows],
    )

    report = f"""# Head-to-head: `{name_a}` vs `{name_b}` ({split} split, n={len(y_true)})

## Overall
{md_table(['metric', name_a, name_b, 'delta'], overall)}

## Per-class F1
{md_table(['class', f'f1 {name_a}', f'f1 {name_b}', 'delta'], per_class_rows)}

## Paired analysis
- {name_b} fixed {len(fixed_idx)} examples {name_a} got wrong.
- {name_b} regressed on {len(regressed_idx)} examples {name_a} got right.
- Exact binomial p-value on disagreements: **{p_value:.5f}**
  (significant at 0.05 means the difference is unlikely to be chance).

## Stress benchmark
{stress_table}

## Examples fixed by {name_b} (up to {max_examples})
```json
{json.dumps(examples(fixed_idx, max_examples), indent=2)}
```

## Examples regressed by {name_b} (up to {max_examples})
```json
{json.dumps(examples(regressed_idx, max_examples), indent=2)}
```
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({
            "runs": [name_a, name_b], "split": split, "n": int(len(y_true)),
            "overall": overall, "per_class": per_class_rows,
            "fixed": int(len(fixed_idx)), "regressed": int(len(regressed_idx)),
            "p_value": p_value, "stress": stress_rows,
        }, indent=2), encoding="utf-8"
    )
    log.info("Comparison report -> %s", out_dir / "report.md")
    print(overall_table_md(name_a, name_b, overall))
    return out_dir


def _summary(logits: np.ndarray, y_true: np.ndarray, classes: list[str]) -> dict:
    from vision_pipeline.training.metrics import summarize

    return summarize(logits, y_true, classes)


def model_a_head_dev(model):
    """Device of the model's parameters (kept simple: single device)."""
    return next(model.parameters()).device


def overall_table_md(name_a: str, name_b: str, overall: list) -> str:
    return md_table(["metric", name_a, name_b, "delta"], overall)
