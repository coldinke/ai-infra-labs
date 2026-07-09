# Agent Instructions

This repo is a CUDA / GPU infra lab workspace. Keep labs small, measurable, and
consistent across operators.

## Core Principles

- Correctness first.
- Benchmark before optimizing.
- Profile before guessing.
- Record curated conclusions in docs or lab notes.
- Prefer small complete labs over large unfinished rewrites.

## Lab Structure

Each operator lab should keep the same shape:

```text
labs/<id>-<topic>/
  <op>_lab.py
  benchmarks/<op>.py
  benchmarks/<variant_wrapper>.py
  tests/test_<op>.py
  README.md
  notes/
```

- `<op>_lab.py` owns shapes, dtype parsing, variants, reference
  implementation, input factories, correctness helpers, and byte estimates.
- `benchmarks/<op>.py` is the main benchmark/profile entrypoint.
- Benchmark wrapper files should only import `main` and set `default_variant`.
- Tests should discover variants from `lab.VARIANTS` so future CUDA/Triton
  variants are covered without adding one test file per variant.

## Variant Rules

- Use stable variant names such as `torch`, `manual_eager`, `cuda_naive`, and
  later explicit optimized names.
- Keep baseline semantics stable. Add new optimized variants instead of
  changing the meaning of an existing variant.
- `--variant all` is valid for benchmark mode.
- Profile mode must run exactly one variant at a time.

## Justfile Interaction

Generic recipes are the source of truth:

```bash
just remote-test <lab>
just remote-bench <lab> <type> --variant all
just remote-profile <lab> <type> --variant <variant>
```

Lab shortcuts should only wrap the generic recipes and should support `*args`
when extra pytest or benchmark arguments are useful.

Do not add one-off recipes for individual variants. Prefer:

```bash
just softmax-profile --variant manual_eager
```

over a dedicated recipe such as `softmax-manual-profile`.

## Build, Test, Bench, Profile

Keep build/test/bench/profile separate. Test recipes should not implicitly build
CUDA extensions.

Typical remote flow:

```bash
just push
just <lab>-build
just <lab>-test
just <lab>-bench
just <lab>-profile
```

When a new lab or untracked file is added locally, run `just push` before remote
test/bench/profile. The remote copy intentionally has no `.git` directory.

## CUDA Extension Scaffold

CUDA extension labs should follow the RMSNorm / Softmax pattern:

```text
setup.py
src/<op>.cpp
src/<op>_kernel.cu
```

- `setup.py` should use `torch.utils.cpp_extension.CUDAExtension`.
- C++ binding files validate tensor device, contiguity, dtype, rank, and shape.
- CUDA files own kernels, launches, current stream lookup, and
  `C10_CUDA_KERNEL_LAUNCH_CHECK()`.
- Pybind modules should expose at least `forward` and `block_size`.
- Initial CUDA variants should be simple and correct before optimized variants
  are added.

## Benchmark And Profile

- Use `labkit.bench` for wall-clock and CUDA-event timings.
- Use `labkit.cuda` for CUDA availability, environment printing, and NVTX
  profile loops.
- Use `nsys` first to inspect timeline, launch overhead, kernel count, and NVTX
  range coverage.
- Use `ncu` only after the hot kernel is known.
- Keep `ncu` iteration counts small because it may replay kernels.

Benchmark argument parsers should use:

```python
argparse.ArgumentParser(allow_abbrev=False)
```

This prevents truncated flags such as `--batch-` from being accepted as
`--batch-size`.

## Verification Boundaries

If the local machine has no CUDA, do not claim GPU correctness or performance
was verified locally. Run local static checks instead:

```bash
python3 -m py_compile <files>
bash -n scripts/run_lab.sh scripts/gpu_remote_sync.sh
just --list
git diff --check
```

CUDA correctness, build, benchmark, and profile checks must run on the remote
GPU machine.

## Artifact Policy

- Do not commit `__pycache__`, build output, compiled extensions, raw profiler
  files, or profile databases.
- Keep raw profile artifacts under `labs/*/profiles/`.
- Pull raw profiler databases only when GUI/offline inspection is needed.
- Write durable conclusions into lab notes or docs instead of relying on raw
  profile directories.
