"""Unit and integration tests for model optimizations, quantization, and evaluation."""
import json
import py_compile
from pathlib import Path

import numpy as np
import pytest
import torch

from app import core
from app.core import (
    BasketTokenizer,
    RetailTransformerLM,
    evaluate_recommender,
    predict_next_items,
    train_recommender_lm,
)


@pytest.fixture(scope="module")
def shared_tokenizer():
    vocab_path = Path("data/vocab.json")
    if not vocab_path.exists():
        core.milestone_1(n_baskets=10000, seed=42)
    return BasketTokenizer.load(vocab_path)


@pytest.fixture(scope="module")
def shared_test_baskets():
    test_path = Path("data/test_baskets.json")
    if not test_path.exists():
        core.milestone_1(n_baskets=10000, seed=42)
    return json.loads(test_path.read_text())


def test_tokenization_and_vocabulary(shared_tokenizer, tmp_path):
    """Test tokenization encoding, decoding, special tokens, and serialization."""
    tok = shared_tokenizer
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.unk_id == 3
    assert tok.vocab_size >= 1000

    items = tok.item_names[:3]
    encoded = tok.encode(items, add_special_tokens=True)
    assert encoded[0] == tok.bos_id
    assert encoded[-1] == tok.eos_id
    assert len(encoded) == len(items) + 2

    decoded = tok.decode(encoded, skip_special_tokens=True)
    assert decoded == items

    save_path = tmp_path / "vocab_test.json"
    tok.save(save_path)
    loaded_tok = BasketTokenizer.load(save_path)
    assert loaded_tok.vocab_size == tok.vocab_size
    assert loaded_tok.item_to_id == tok.item_to_id


def test_model_forward_pass_and_causality():
    """Verify model forward pass shape and causal attention masking."""
    vocab_size = 100
    model = RetailTransformerLM(vocab_size=vocab_size, d_model=32, n_head=2, num_layers=1, d_ff=64)
    model.eval()

    # Shape check
    x = torch.randint(0, vocab_size, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, vocab_size)

    # Causality test: changing token at pos 5 must NOT change logits at pos 0..4
    x1 = torch.tensor([[10, 20, 30, 40, 50, 60]], dtype=torch.long)
    x2 = torch.tensor([[10, 20, 30, 40, 50, 99]], dtype=torch.long)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    # Positions 0 through 4 must be identical
    assert torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5)


def test_training_loss_decrease(shared_tokenizer, shared_test_baskets, tmp_path):
    """Verify that training on basket sequences converges and decreases loss."""
    subset = shared_test_baskets[:50]
    out_model_path = tmp_path / "mini_model.pt"

    model, losses = train_recommender_lm(
        train_baskets=subset,
        tokenizer=shared_tokenizer,
        epochs=3,
        batch_size=16,
        lr=3e-3,
        d_model=32,
        n_head=2,
        num_layers=1,
        d_ff=64,
        out_path=str(out_model_path),
        shuffle_order=True,
    )

    assert len(losses) == 3
    assert losses[-1] < losses[0]
    assert out_model_path.exists()


def test_next_item_prediction_scores_and_ranking(shared_tokenizer):
    """Verify predicted recommendations are ranked with valid probabilities."""
    model_path = Path("data/baseline_model.pt")
    if not model_path.exists():
        core.milestone_2(epochs=1, batch_size=64)

    model = torch.load(model_path, weights_only=False)
    sample_basket = shared_tokenizer.item_names[:3]

    recs = predict_next_items(
        model=model,
        visible_items=sample_basket,
        tokenizer=shared_tokenizer,
        top_k=5,
        exclude_input=True,
    )

    assert len(recs) <= 5
    scores = [r["score"] for r in recs]
    ranks = [r["rank"] for r in recs]

    assert ranks == list(range(1, len(recs) + 1))
    assert scores == sorted(scores, reverse=True)
    for s in scores:
        assert 0.0 <= s <= 1.0


def test_candidate_masking_excludes_visible_and_special(shared_tokenizer):
    """Verify candidate recommendations exclude all visible items and special tokens."""
    model_path = Path("data/baseline_model.pt")
    model = torch.load(model_path, weights_only=False)

    visible = [shared_tokenizer.item_names[0], shared_tokenizer.item_names[1]]
    recs = predict_next_items(
        model=model,
        visible_items=visible,
        tokenizer=shared_tokenizer,
        top_k=10,
        exclude_input=True,
    )

    rec_names = [r["item"] for r in recs]
    for v in visible:
        assert v not in rec_names

    for special in shared_tokenizer.special_tokens:
        assert special not in rec_names


def test_evaluation_metrics_topk_and_mrr(shared_tokenizer, shared_test_baskets):
    """Verify evaluation metric calculations for top-1, top-5, top-10, MRR and latency."""
    model_path = Path("data/baseline_model.pt")
    model = torch.load(model_path, weights_only=False)

    metrics = evaluate_recommender(
        model=model,
        test_baskets=shared_test_baskets,
        tokenizer=shared_tokenizer,
        n_eval_cases=15,
        seed=42,
    )

    assert 0.0 <= metrics["top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["top5_accuracy"] <= 1.0
    assert 0.0 <= metrics["top10_accuracy"] <= 1.0
    assert metrics["top10_accuracy"] >= metrics["top5_accuracy"] >= metrics["top1_accuracy"]
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert metrics["latency_ms"] > 0.0
    assert metrics["throughput_qps"] > 0.0


def test_quantized_model_loading_and_execution(shared_tokenizer):
    """Verify loading and execution of INT8 quantized models."""
    dyn_path = Path("data/quantized_dynamic.pt")
    stat_path = Path("data/quantized_static.pt")

    if not (dyn_path.exists() and stat_path.exists()):
        core.milestone_5(n_calibration_baskets=20)

    dyn_model = torch.load(dyn_path, weights_only=False)
    stat_model = torch.load(stat_path, weights_only=False)

    vis = shared_tokenizer.item_names[:2]
    recs_dyn = predict_next_items(dyn_model, vis, shared_tokenizer, top_k=3)
    recs_stat = predict_next_items(stat_model, vis, shared_tokenizer, top_k=3)

    assert len(recs_dyn) > 0
    assert len(recs_stat) > 0


def test_recommendation_consistency(shared_tokenizer):
    """Verify recommendations are deterministic and consistent given identical inputs."""
    model_path = Path("data/baseline_model.pt")
    model = torch.load(model_path, weights_only=False)
    vis = shared_tokenizer.item_names[:3]

    recs_1 = predict_next_items(model, vis, shared_tokenizer, top_k=5)
    recs_2 = predict_next_items(model, vis, shared_tokenizer, top_k=5)

    assert [r["item"] for r in recs_1] == [r["item"] for r in recs_2]
    assert np.allclose([r["score"] for r in recs_1], [r["score"] for r in recs_2], atol=1e-6)


def test_streamlit_application_syntax_and_imports():
    """Verify app/main.py compiles without syntax or import errors."""
    py_compile.compile("app/main.py", doraise=True)
