"""Hand-computed metric checks."""
import numpy as np

from vision_pipeline.training.metrics import (
    confusion_matrix,
    prf_from_cm,
    summarize,
    topk_accuracy,
)


def test_confusion_and_prf():
    y = np.array([0, 0, 1, 2, 2, 2])
    p = np.array([0, 1, 1, 2, 2, 0])
    cm = confusion_matrix(y, p, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]
    prf = prf_from_cm(cm)
    # class 0: tp=1, fp=1, fn=1 -> p=r=f1=0.5 ; class 1: p=0.5 r=1 f1=2/3 ; class 2: p=1 r=2/3 f1=0.8
    assert abs(prf["f1"][0] - 0.5) < 1e-6
    assert abs(prf["f1"][1] - 2 / 3) < 1e-6
    assert abs(prf["f1"][2] - 0.8) < 1e-6
    assert abs(prf["macro_f1"] - np.mean([0.5, 2 / 3, 0.8])) < 1e-6


def test_topk():
    logits = np.array([[0.0, 10.0, 1.0], [10.0, 1.0, 2.0]])
    y = np.array([2, 0])
    # argmax: sample 0 -> class 1 (wrong, truth 2); sample 1 -> class 0 (right)
    assert topk_accuracy(y, logits, 1) == 0.5
    # top-3 covers everything here
    assert topk_accuracy(y, logits, 3) == 1.0


def test_summarize_shape():
    logits = np.random.default_rng(0).normal(size=(20, 3))
    y = np.random.default_rng(1).integers(0, 3, 20)
    out = summarize(logits, y, ["a", "b", "c"], loss=0.5)
    assert abs(out["n_examples"] - 20) < 1
    assert out["n_examples"] == 20
    assert len(out["confusion_matrix"]) == 3
    assert set(out["per_class"]) == {"a", "b", "c"}
    assert "loss" in out
