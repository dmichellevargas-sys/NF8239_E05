import numpy as np
import pandas as pd
import pytest

from inf8239_u03.metrics import (
    catalog_coverage,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)


def test_ranking_metrics():
    recs = {1: [3, 2], 2: [4, 1]}
    truth = pd.DataFrame({"userId": [1, 2], "movieId": [3, 5]})
    assert hit_rate_at_k(recs, truth, 2) == 0.5
    assert catalog_coverage(recs, 5) == 0.8


def test_precision_recall_ndcg_known_values():
    recs, relevant = [1, 2, 3, 4], {2, 4, 9}
    assert precision_at_k(recs, relevant, 4) == 0.5
    assert recall_at_k(recs, relevant, 4) == pytest.approx(2 / 3)
    expected = (1 / np.log2(3) + 1 / np.log2(5)) / (1 + 1 / np.log2(3) + 1 / np.log2(4))
    assert ndcg_at_k(recs, relevant, 4) == pytest.approx(expected)
    assert ndcg_at_k([2, 4, 9], relevant, 3) == pytest.approx(1.0)


def test_novelty_is_higher_for_rare_items():
    popularity = {1: 0.5, 2: 0.01}
    assert novelty([2], popularity) > novelty([1], popularity)


def test_diversity_zero_for_identical_and_one_for_orthogonal_items():
    same = {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])}
    ortho = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    assert intra_list_diversity([1, 2], same) == pytest.approx(0.0)
    assert intra_list_diversity([1, 2], ortho) == pytest.approx(1.0)
