import time

import numpy as np
import pytest

from gpubench.timing import measure, summarize


def test_summarize_matches_numpy():
    x = [1.0, 2.0, 3.0, 4.0, 100.0]
    s = summarize(x)
    assert s.n == 5
    assert s.median == 3.0
    assert s.q1 == np.percentile(x, 25)
    assert s.q3 == np.percentile(x, 75)
    assert s.iqr == pytest.approx(s.q3 - s.q1)
    assert s.n_outliers == 1  # 100 is far outside 1.5 * IQR
    assert s.min == 1.0 and s.max == 100.0


def test_summarize_single_sample_has_zero_std():
    assert summarize([0.5]).std == 0.0


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_measure_calls_warmup_and_sync():
    calls = {"fn": 0, "sync": 0}

    def fn():
        calls["fn"] += 1

    def sync():
        calls["sync"] += 1

    samples = measure(fn, sync, warmup=2, repeats=5)
    assert len(samples) == 5
    assert calls["fn"] == 2 + 5
    # one sync after warm-up, then one before and one after every sample
    assert calls["sync"] == 1 + 2 * 5


def test_measure_stops_early_on_time_budget():
    samples = measure(lambda: time.sleep(0.02), warmup=0, repeats=1000,
                      min_repeats=3, max_seconds=0.05)
    assert 3 <= len(samples) < 10


def test_measure_reflects_duration():
    samples = measure(lambda: time.sleep(0.01), warmup=0, repeats=3)
    assert min(samples) >= 0.009
