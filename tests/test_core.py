"""Unit and integration tests for Milestones 1 through 6."""
import os
from pathlib import Path
import pytest

from app import core


def test_milestone_1():
    """Milestone 1: Prepare synthetic shopping basket data for training."""
    res = core.milestone_1(n_baskets=10000, seed=42)
    assert res["status"] == "completed"
    assert res["n_baskets"] == 10000
    assert res["n_train"] == 8000
    assert res["n_test"] == 2000
    assert res["vocab_size"] > 0
    assert res["catalog_size"] > 0
    assert res["n_frequent_itemsets"] > 0

    assert Path("data/vocab.json").exists()
    assert Path("data/catalog.json").exists()
    assert Path("data/train_baskets.json").exists()
    assert Path("data/test_baskets.json").exists()
    assert Path("data/truth.json").exists()
    assert Path("data/meta.json").exists()


def test_milestone_2():
    """Milestone 2: Continue pretraining a small LM on the basket data."""
    res = core.milestone_2(epochs=1, batch_size=64)
    assert res["status"] == "completed"
    assert res["final_loss"] > 0
    assert res["total_parameters"] > 0
    assert Path(res["model_path"]).exists()
    assert res["model_size_mb"] > 0


def test_milestone_3():
    """Milestone 3: Implement next-item prediction and itemset probing."""
    res = core.milestone_3(top_k=5, n_probe_samples=20)
    assert res["status"] == "completed"
    assert "sample_basket" in res
    assert len(res["recommendations"]) > 0

    first_rec = res["recommendations"][0]
    assert "item" in first_rec
    assert "score" in first_rec
    assert "rank" in first_rec
    assert 0.0 <= first_rec["score"] <= 1.0

    probe = res["probe_stats"]
    assert "n_generated_baskets" in probe
    assert "recovered_itemsets_count" in probe
    assert "precision" in probe


def test_milestone_4():
    """Milestone 4: Measure the baseline size and accuracy of the model."""
    res = core.milestone_4(n_eval_cases=20)
    assert res["status"] == "completed"
    metrics = res["baseline_metrics"]

    assert metrics["model_size_mb"] > 0
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["top5_accuracy"] <= 1.0
    assert metrics["top5_accuracy"] >= metrics["top1_accuracy"]
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert metrics["latency_ms"] > 0
    assert metrics["n_evaluated_cases"] == 20


def test_milestone_5():
    """Milestone 5: Apply dynamic and static post-training quantization to the model."""
    res = core.milestone_5(n_calibration_baskets=20)
    assert res["status"] == "completed"

    assert Path(res["dynamic_model_path"]).exists()
    assert Path(res["static_model_path"]).exists()

    assert res["original_size_mb"] > 0
    assert res["dynamic_size_mb"] > 0
    assert res["static_size_mb"] > 0

    assert res["dynamic_size_mb"] < res["original_size_mb"]
    assert res["static_size_mb"] < res["original_size_mb"]
    assert res["dynamic_reduction_pct"] > 0
    assert res["static_reduction_pct"] > 0


def test_milestone_6():
    """Milestone 6: Demonstrate size reduction and evaluate accuracy impact."""
    res = core.milestone_6(n_eval_cases=20)
    assert res["status"] == "completed"
    assert "baseline" in res
    assert "dynamic_int8" in res
    assert "static_int8" in res

    table = res["comparison_table"]
    assert len(table) == 3

    model_names = [row["model"] for row in table]
    assert "Baseline FP32" in model_names
    assert "Dynamic INT8" in model_names
    assert "Static INT8 (PTQ)" in model_names

    for row in table:
        assert row["size_mb"] > 0
        assert row["latency_ms"] > 0
        assert 0.0 <= row["top1_acc"] <= 1.0
        assert 0.0 <= row["top5_acc"] <= 1.0
        assert 0.0 <= row["mrr"] <= 1.0
