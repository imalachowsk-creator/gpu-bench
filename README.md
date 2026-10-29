# gpubench — honest CPU vs GPU benchmarking

[![tests](https://github.com/Ignacy_Malachowski/gpu-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/Ignacy_Malachowski/gpu-bench/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ignacy_Malachowski/gpu-bench/blob/main/notebooks/colab_gpu_benchmark.ipynb)

"The GPU is 50× faster" is easy to claim and easy to get wrong. A forgotten
`torch.cuda.synchronize()`, a cached result or a silent drop to half precision
can all produce impressive and meaningless numbers.

**gpubench** measures the same kernels on NumPy (CPU), PyTorch CPU and PyTorch
CUDA, and **refuses to report a result it cannot verify**. Beyond raw timings it
answers three practical questions:

1. **From what problem size does the GPU actually win**, and by how much?
2. **How expensive is moving data** between host and GPU? (latency + bandwidth model)
3. **How much GPU memory will a workload need?** (predicted *before* running, then checked)

## What is measured

| Workload | Character | Reported as |
|---|---|---|
| `matmul` (n × n) | compute-bound, 2n³ FLOPs | time, GFLOP/s, speed-up |
| `axpy` (`a·x + y`) | memory-bound, 2 FLOPs per 3 values moved | time, GB/s, speed-up |
| `mean` | reduction (many values → one) | time, GB/s, speed-up |
| host ↔ device copies | PCIe transfers, pinned vs pageable memory | GB/s, fitted latency & bandwidth |
| peak GPU memory | matmul and MLP inference, float32 vs float16 | predicted vs measured MiB |

## Methodology

**Timing** (`gpubench/timing.py`)
- warm-up runs are discarded (CUDA context creation, kernel selection, allocations, cold caches);
- the device is synchronised before *and* after every timed call, because CUDA kernels run asynchronously;
- every configuration is repeated (up to 30 times, time-budgeted) and summarised with the **median and IQR**, which are robust to the occasional slow run. Outliers are counted with the 1.5 × IQR rule.

**Verification and anti-cheating guards** (`gpubench/verify.py`)

Speed is only reported for implementations that pass all three checks:

| Check | Catches |
|---|---|
| Output vs float64 reference, dtype-aware tolerance (matmul: random row sample) | wrong results, silent precision loss |
| Re-run on **fresh** random inputs | cached / memoised / input-independent outputs |
| Achieved FLOP/s and bytes/s below physical limits | "work" that never happened |

`gpubench/cheats.py` contains deliberately broken implementations (cached result,
skipped work, hidden float16). The test suite and CI fail if any of them slips
through — the guards are themselves tested.

**Transfer model** (`gpubench/transfer.py`): copy time is modelled as
`time = latency + bytes / bandwidth` and fitted by weighted least squares
(weights `1/time`, so 4 KiB and 256 MiB copies count equally).

**Memory model** (`gpubench/memory.py`): tracks which tensors are alive at each
step of an `nn.Sequential` forward pass (weights, caller's input, current
layer's input and output), rounds each allocation to PyTorch's 512-byte blocks,
and compares the predicted peak with `torch.cuda.max_memory_allocated()`.

## Results

> Measured on a Google Colab **NVIDIA Tesla T4** (16 GB, PyTorch 2.11, CUDA 13.0)
> against the Colab VM's CPU. All 104 timed configurations passed verification.
> Full tables: [`results/colab-t4/summary.md`](results/colab-t4/summary.md).

| | |
|---|---|
| ![time vs size](results/colab-t4/figures/time_vs_size.png) | ![speed-up](results/colab-t4/figures/speedup.png) |
| ![transfer](results/colab-t4/figures/transfer.png) | ![memory](results/colab-t4/figures/memory.png) |

**Key findings**

- **Small problems belong on the CPU.** Every GPU call costs a fixed ~20–40 µs
  regardless of size (launch + synchronisation), so the GPU only wins from a 128×128
  matmul, 65 536-element axpy and 262 144-element mean onwards.
- **Large matmul: ~29× faster in float32, ~158× in float16** than PyTorch on
  the CPU (n = 4096; 1.08 s → 37 ms → 6.8 ms). float16 reaches 39 TFLOP/s at
  n = 2048 thanks to Tensor Cores, but falls to 20 TFLOP/s at n = 4096 — the
  70 W T4 is power-limited under sustained load.
- **Memory-bound kernels gain far less than compute-bound ones.** axpy peaks at
  ~145 GB/s and mean at ~263 GB/s (T4 spec: 320 GB/s), giving 23–34× speed-ups over
  NumPy in float32 at 2²⁶ elements; float16 roughly doubles that only because
  it halves the bytes moved.
- **Transfers can erase the gain.** Pinned memory reaches 12–13 GB/s over PCIe,
  pageable only 4–6 GB/s, with ~22 µs fixed latency per copy. Copying two
  4096×4096 float32 matrices to the GPU (128 MiB) takes ~11 ms pinned —
  longer than the float16 matmul itself.
- **Peak memory is predictable.** The tensor-liveness model matched
  `torch.cuda.max_memory_allocated()` exactly in 13 of 14 cases (within 1 MiB
  in the last), and confirms float16 halves the footprint.
- **The simple transfer model has limits.** It fits pinned copies almost
  perfectly (R² ≈ 1.00) but not pageable host-to-device copies (R² = 0.92),
  which go through the driver's staging buffers in chunks.
- **Caveat:** the NumPy baseline on a shared Colab VM was noisy (54–123 GFLOP/s
  for large matmul), which inflates some matmul speed-ups in the NumPy plot (up to ~700× in
  float16); the matmul figures above are quoted against the steadier PyTorch
  CPU baseline instead.

## Quick start

```bash
pip install -e ".[dev]"          # add ".[gpu]" or use a CUDA build of PyTorch for GPU runs
python -m gpubench env           # which backends are available
python -m gpubench cheats        # see the guards reject broken implementations
python -m gpubench run --preset full --out results/my-gpu --report
pytest                           # GPU tests are skipped automatically without CUDA
```

On a machine without a GPU everything runs on the CPU backends; transfer and
memory benchmarks are skipped. For a real GPU, open the
[Colab notebook](notebooks/colab_gpu_benchmark.ipynb).

## Project structure

```
gpubench/
  backends.py   NumPy / PyTorch CPU / PyTorch CUDA behind one interface (incl. sync)
  workloads.py  matmul, axpy, mean: inputs, reference, FLOP and byte counts
  timing.py     warm-up, synchronised repeated timing, median / IQR statistics
  verify.py     correctness, fresh-input and plausibility checks
  cheats.py     deliberately broken implementations the guards must catch
  transfer.py   host<->device copies + latency/bandwidth fit
  memory.py     peak-memory prediction and measurement
  runner.py     runs a suite, saves CSV + environment metadata
  report.py     figures and summary.md
  cli.py        `python -m gpubench ...`
tests/          CPU tests (run in CI) and GPU tests (run on Colab)
notebooks/      Colab notebook for the real GPU run
```

## Limitations and next steps

- Wall-clock timing with explicit synchronisation includes kernel-launch
  overhead; CUDA events or Nsight Systems would isolate pure kernel time.
- Only single-GPU, single-stream execution; overlapping copies with compute
  (CUDA streams) is a natural extension.
- The memory model covers inference of plain MLPs; extending it to attention
  layers and the KV cache, and to quantised (int8/int4) weights, would make it
  useful for LLM serving.
- More backends (CuPy, Triton, `torch.compile`) would plug into the same
  `Backend` interface.

## License

MIT
