import numpy as np
import pytest

from gpubench.backends import NumpyBackend, available_backends, make_backend, torch_available
from gpubench.workloads import AXPY, MATMUL, MEAN, get_workload


def test_flop_and_byte_counts():
    assert MATMUL.flops(10) == 2000
    assert MATMUL.bytes_moved(10, "float32") == 3 * 100 * 4
    assert AXPY.flops(1000) == 2000
    assert AXPY.bytes_moved(1000, "float16") == 3 * 1000 * 2
    assert MEAN.bytes_moved(1000, "float64") == 8000


def test_inputs_are_quantized_to_dtype():
    rng = np.random.default_rng(0)
    a, b = MATMUL.make_inputs(rng, 8, "float16")
    assert a.dtype == np.float16 and b.shape == (8, 8)


def test_matmul_reference_uses_sampled_rows():
    rng = np.random.default_rng(0)
    inputs = MATMUL.make_inputs(rng, 200, "float64")
    out = inputs[0] @ inputs[1]
    actual, expected = MATMUL.reference(inputs, out, rng)
    assert actual.shape == (64, 200)
    np.testing.assert_allclose(actual, expected)


def test_numpy_backend_ops():
    be = NumpyBackend()
    x = be.from_numpy(np.arange(4.0), "float32")
    assert x.dtype == np.float32
    np.testing.assert_allclose(be.to_numpy(be.axpy(2.0, x, x)), [0, 3, 6, 9])
    assert float(be.mean(x)) == 1.5
    assert be.cast(x, "float16").dtype == np.float16


def test_backend_registry():
    assert available_backends()[0] == "numpy"
    assert make_backend("numpy").name == "numpy"
    with pytest.raises(ValueError):
        make_backend("tpu")
    with pytest.raises(ValueError):
        get_workload("fft")


@pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
def test_torch_cpu_backend_matches_numpy():
    be = make_backend("torch-cpu")
    rng = np.random.default_rng(1)
    a, b = MATMUL.make_inputs(rng, 32, "float32")
    out = be.to_numpy(be.matmul(be.from_numpy(a, "float32"), be.from_numpy(b, "float32")))
    np.testing.assert_allclose(out, a.astype(np.float64) @ b, rtol=1e-4, atol=1e-4)
