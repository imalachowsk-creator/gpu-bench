"""Compute backends: the same operations implemented with NumPy and PyTorch.

A backend hides *where* the computation runs (CPU or GPU) behind one small
interface, so the benchmark code can treat every device the same way.

The most important method is :meth:`Backend.sync`. GPU work in PyTorch is
asynchronous: ``a @ b`` returns as soon as the kernel is *queued*, not when it
has finished. Without a synchronisation point before reading the clock you
measure how long it takes to queue work, not how long the work takes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Dtypes are passed around as plain strings so configs stay simple.
NUMPY_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}


class Backend:
    """Common interface. Subclasses implement the device-specific parts."""

    name: str = "base"
    is_gpu: bool = False
    supported_dtypes: tuple[str, ...] = ()

    # --- data movement -------------------------------------------------
    def from_numpy(self, array: np.ndarray, dtype: str) -> Any:
        raise NotImplementedError

    def to_numpy(self, value: Any) -> np.ndarray:
        """Return a float64 NumPy copy (used only for verification)."""
        raise NotImplementedError

    def sync(self) -> None:
        """Block until all queued work on this device has finished."""

    def cast(self, value: Any, dtype: str) -> Any:
        """Convert an on-device value to another dtype (stays on the device)."""
        raise NotImplementedError

    # --- operations ----------------------------------------------------
    def matmul(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def axpy(self, alpha: float, x: Any, y: Any) -> Any:
        """``alpha * x + y`` -- a classic memory-bound kernel."""
        raise NotImplementedError

    def mean(self, x: Any) -> Any:
        """Full reduction to a scalar."""
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        return {"backend": self.name}

    def __repr__(self) -> str:
        return f"<Backend {self.name}>"


class NumpyBackend(Backend):
    """CPU baseline using NumPy (matmul is delegated to the BLAS library)."""

    name = "numpy"
    is_gpu = False
    # float16 matmul in NumPy has no BLAS fast path and is extremely slow,
    # so it would not be a fair baseline. Keep it out by default.
    supported_dtypes = ("float32", "float64")

    def from_numpy(self, array: np.ndarray, dtype: str) -> np.ndarray:
        return np.ascontiguousarray(array, dtype=NUMPY_DTYPES[dtype])

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)

    def cast(self, value, dtype):
        return np.asarray(value).astype(NUMPY_DTYPES[dtype])

    def matmul(self, a, b):
        return a @ b

    def axpy(self, alpha, x, y):
        return alpha * x + y

    def mean(self, x):
        return x.mean()

    def describe(self) -> dict[str, str]:
        return {"backend": self.name, "numpy": np.__version__}


class TorchBackend(Backend):
    """PyTorch on CPU or CUDA. Imported lazily so NumPy-only installs work."""

    def __init__(self, device: str = "cuda"):
        import torch  # noqa: WPS433 (lazy import is intentional)

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but no CUDA GPU is available")
        self.is_gpu = self.device.type == "cuda"
        self.name = f"torch-{self.device.type}"
        self.supported_dtypes = ("float16", "float32") if self.is_gpu else ("float32",)
        self._dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }

    def from_numpy(self, array, dtype):
        t = self.torch.from_numpy(np.ascontiguousarray(array))
        return t.to(device=self.device, dtype=self._dtypes[dtype])

    def to_numpy(self, value):
        return value.detach().to(device="cpu", dtype=self.torch.float64).numpy()

    def sync(self) -> None:
        if self.is_gpu:
            self.torch.cuda.synchronize(self.device)

    def cast(self, value, dtype):
        return value.to(dtype=self._dtypes[dtype])

    def matmul(self, a, b):
        return a @ b

    def axpy(self, alpha, x, y):
        return alpha * x + y

    def mean(self, x):
        return x.mean()

    def describe(self) -> dict[str, str]:
        info = {"backend": self.name, "torch": self.torch.__version__}
        if self.is_gpu:
            info["gpu"] = self.torch.cuda.get_device_name(self.device)
            info["cuda"] = str(self.torch.version.cuda)
        return info


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def cuda_available() -> bool:
    if not torch_available():
        return False
    import torch

    return torch.cuda.is_available()


def make_backend(name: str) -> Backend:
    """Create a backend from its name: ``numpy``, ``torch-cpu`` or ``torch-cuda``."""
    if name == "numpy":
        return NumpyBackend()
    if name == "torch-cpu":
        return TorchBackend("cpu")
    if name == "torch-cuda":
        return TorchBackend("cuda")
    raise ValueError(f"Unknown backend {name!r}. Use numpy, torch-cpu or torch-cuda.")


def available_backends() -> list[str]:
    names = ["numpy"]
    if torch_available():
        names.append("torch-cpu")
    if cuda_available():
        names.append("torch-cuda")
    return names
