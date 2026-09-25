# Testing and upstream parity

The C++ library, CLI and `gliner-bench` do not embed or invoke Python. Python tooling is limited to one-time conversion and developer verification.

## Test layers

| Layer | Dependencies | Coverage |
|---|---|---|
| `gliner-core` | C/C++ and GGML | C-compiled API smoke/error contracts |
| `gliner-cuda-precision-config`, `gliner-metal-precision-config` | CMake and a C++ compiler; no GPU needed | Source-overlay correctness, repeatability, rejection of unknown/mixed dispatch layouts, target wiring and unmodified GGML checkouts |
| `gliner-conversion` | Python 3.10+ standard library | Streaming conversion, tensor preservation, C API classification/extraction, CLI errors and checked-in F32/F16 token/layer/span-logit goldens |
| `gliner-benchmark` | Python 3.10+ standard library | Native benchmark output/statistics, modes, warmup/sample counts and result consistency |
| `gliner-real-parity` (explicit opt-in) | Python plus `tests/requirements-parity.txt`, local weights and GGUF | Live upstream reference comparison for all 13 single/joint cases |
| `gliner-real-span-parity` (explicit opt-in) | Same reference environment, span-enabled GGUF | Six extraction cases: word/label routing, every candidate logit, count gate and exact decoded spans |
| `gliner-real-record-parity` (explicit opt-in) | Same reference environment, span-enabled GGUF | Seven repeated-record cases: counts, every recurrent slot/field logit, decoded fields and original offsets |
| `gliner-real-boundary-parity` (explicit opt-in) | Same reference environment, shared-pool boundary checkpoint/GGUF | Eleven entity cases: exact pool/routing, start/end/null/proposal/pair logits, weighted flat decoding and multilingual offsets |
| `gliner-real-boundary-record-parity` (explicit opt-in) | Same reference environment, record-enabled boundary GGUF | Sixteen cases across anchorless/latent modes: raw pair/object/assignment scores, canonical slots and decoded records |

The full C API checks run against tiny generated models during `gliner-conversion`; the standalone native smoke test does not replace this inference coverage.

Run the normal suite with no ML package installation:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

CMake adds the two standard-library integration tests when Python 3.10+ is found. If it is absent, configuration reports that they were not registered. Disable Python discovery explicitly with:

```sh
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release -DGLINER_PYTHON_TESTS=OFF
cmake --build build-native --config Release -j
ctest --test-dir build-native -C Release --output-on-failure
```

This runs the three native tests, not the synthetic or real-model suites. It does not change inference functionality. `BUILD_TESTING=OFF` disables all test targets.

## One optional upstream reference tool

Keep ML packages in a separate Python 3.12 environment. Use an approved package index for your environment; this example uses the proxy used during Windows validation:

```powershell
py -3.12 -m venv build-oracle
.\build-oracle\Scripts\python.exe -m pip install -r .\tests\requirements-parity.txt `
  --index-url https://packagefeedproxy.microsoft.io/pypi/simple/

.\build-oracle\Scripts\python.exe .\tests\test_parity.py `
  --checkpoint .\models\GLiNER2.5-Decide `
  --gguf .\models\decide.gguf `
  --binary .\build\Release\gliner-classify.exe `
  --expect-backend cpu
```

The Python reference runs on CPU; it can compare a C++ CPU, CUDA or strict Metal binary without installing GPU-enabled PyTorch. Pass the matching executable and `--expect-backend cuda` or `metal` for those builds. Single-configuration generators put the executable directly in the build directory rather than `Release`.

The tool compares exact token IDs, marker positions and task ranges; embeddings plus every encoder layer; contextual label states; raw logits; and task-local probabilities/decisions. The real F32 tolerance is `atol=rtol=3e-4`. The default is all 13 cases; `--single-only`, `--batch-only` and `--case N` narrow the run.

The classification oracle supports both span and boundary checkpoint classifier layouts, including the output-layer index shifted by dropout. `--multilingual-only` selects four additional French, Chinese, Arabic and Spanish cases. The [model matrix](models.md) records real CPU results for the four additional boundary-family checkpoints. Synthetic coverage verifies version-3 metadata, long-name alias reversibility, byte-preserved F32/F16 tensors, dropout/no-dropout classifier layouts, shared-pool extraction, old classification-only GGUF compatibility, and explicit rejection of unsupported boundary variants or missing record metadata.

For CTest integration, set `GLINER_PARITY_CHECKPOINT` and `GLINER_PARITY_GGUF` to absolute local paths and `Python3_EXECUTABLE` to the oracle environment's interpreter. Leave `GLINER_PYTHON_TESTS=ON`. No model is downloaded automatically.

For extraction, run the same tool with `--spans-only` or `--records-only`, a span-capable GGUF and `gliner-extract` instead of `gliner-classify`. Set `GLINER_SPAN_PARITY_GGUF` alongside `GLINER_PARITY_CHECKPOINT` to register both in CTest. The reference uses upstream `SpanRepLayer`, `CountLSTM`, the count head and overlap decoder; there is no separate Python extraction implementation in production. Structured scoring checks the recurrent state and positional embedding for every predicted slot, not a repeated copy of the first entity step.

Boundary models use `--boundary-spans-only`, not either span-architecture mode. For CTest, set `GLINER_BOUNDARY_PARITY_CHECKPOINT` and `GLINER_BOUNDARY_PARITY_GGUF` to absolute paths, independently of the English Decide settings above. The test runs upstream boundary refinement, marginal, shared-pool, content/FiLM and abstention components; it compares selected candidate order and every raw pair/proposal/start/end/null logit. Cases include empty text, Unicode, URLs, truncation, long inputs crossing the local-attention window, the Cloudflare request, and French/Chinese/Arabic/Spanish text. Real F32 uses the same `3e-4` combined absolute/relative tolerance.

## Fixtures

Boundary records use `--boundary-records-only` in the same reference tool. Set `GLINER_BOUNDARY_RECORD_PARITY_GGUF` alongside `GLINER_BOUNDARY_PARITY_CHECKPOINT` to register `gliner-real-boundary-record-parity`. Upstream `RecordHead`/`decode_group` supply the numerical and decoding oracle. List and optional scalar types are nonexclusive; latent identities canonicalize repeated shared-pool seeds across fields.

The checked-in fixtures are deterministic reference outputs, not downloaded model weights. Standard-library tests generate matching tiny checkpoints, convert them, then replay the goldens through the native runtime.

```powershell
.\build-oracle\Scripts\python.exe .\tests\test_parity.py --single-only `
  --write-golden .\tests\fixtures\tiny_parity.json
.\build-oracle\Scripts\python.exe .\tests\test_parity.py --batch-only `
  --write-golden .\tests\fixtures\tiny_batch_parity.json
.\build-oracle\Scripts\python.exe .\tests\test_parity.py --kernel-only `
  --write-golden .\tests\fixtures\tiny_cuda_parity.json
.\build-oracle\Scripts\python.exe .\tests\test_parity.py --spans-only `
  --write-golden .\tests\fixtures\tiny_spans_parity.json
.\build-oracle\Scripts\python.exe .\tests\test_parity.py --records-only `
  --write-golden .\tests\fixtures\tiny_records_parity.json
.\build-oracle\Scripts\python.exe .\tests\test_parity.py --boundary-spans-only `
  --write-golden .\tests\fixtures\tiny_boundary_parity.json
.\build-oracle\Scripts\python.exe .\tests\test_parity.py --boundary-spans-only --synthetic-dtype F16 `
  --write-golden .\tests\fixtures\tiny_boundary_f16_parity.json
.\build-oracle\Scripts\python.exe .\tests\test_parity.py --boundary-records-only `
  --write-golden .\tests\fixtures\tiny_boundary_records_parity.json
.\build-oracle\Scripts\python.exe .\tests\test_parity.py --boundary-records-only --synthetic-dtype F16 `
  --write-golden .\tests\fixtures\tiny_boundary_records_f16_parity.json
```

Fixtures record upstream/tool provenance. The 32-wide, 10/30-token CUDA fixtures exercise a small-matrix dispatch boundary absent from the original 8-wide fixtures. Tiny F32 uses `atol=rtol=5e-5`; F16 storage uses `5e-3` against the F32 goldens. Do not loosen tolerances to hide backend precision changes. Real-checkpoint F16 parity remains separate work.

Span fixtures compare all raw logits under the same numerical tolerances. Exact greedy decoded-span agreement is checked for synthetic F32; F16 tests compare raw scores and result/offset validity because close scores can reorder after weight rounding. Both dtypes exercise C API ownership/cleanup, empty results, thresholds, overlap/top-k and single-pass behavior. Native offsets are half-open UTF-8 bytes; the reference converts Python character offsets and excludes synthetic suffix tokens.

Record fixtures use deterministic uncorrelated synthetic weights and controlled count logits for counts 0, 1, 2, 3 and 19. This exercises the count ceiling and GRU recurrence without relying on a checkpoint to naturally predict every count. Tests compare all slot logits under the existing F32/F16 tolerances, F32 decoded field membership, list/single outputs, empty records, Unicode offsets, repeated state reuse and classification/entity interleaving. Callback counts prove the encoder runs once, including the maximum-count case. A max-record safety-cap violation must clear results rather than truncate.

Boundary F32 fixtures verify exact pool order and decoded spans. F16 execution can reorder near-tied boundaries, so its golden additionally scores every valid word-boundary pair upstream: each native-selected pair is compared by coordinates, rather than incorrectly aligning different pools or skipping unmatched candidates. Both use the existing `5e-5`/`5e-3` F32/F16 budgets. Controlled models cover abstention (including equality at the cutoff), empty pools, zero attention/refinement layers, weighted-interval ties, temperature, and buffer/partial-stream loading of aliased head tensors. No learned tensor math is duplicated in the standard-library test runner.

## Backend policies

Boundary record fixtures include all short-input pool candidates so F16 near-ties only reorder, rather than change membership. F16 tests align pool/latent axes by word boundaries and compare all pair/object/assignment logits at `5e-3`; F32 checks exact pool order, canonical slots and decoded fields at `5e-5`. Controlled tests cover distinct learned-instance assignments, max-record failures, no-object logits and empty candidate pools in both modes. The C test verifies one encoder traversal, result clearing, invalid modes, UTF-8 ownership and record equivalence after buffer/partial-stream loading. Existing English Decide record behavior remains covered.

CUDA CTest tests and standalone reference/integration scripts set `NVIDIA_TF32_OVERRIDE=0` and `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` for their child processes. They do not change the invoking shell. Normal application launches still need the [documented CUDA precision settings](../README.md#backends-cpu-cuda-and-metal).

Strict Metal uses the precision overlay and runs the encoder goldens. `GLINER_METAL_STRICT_F32=OFF` preserves the explicit faster/lower-precision mode: `gliner-conversion-fast` checks format/API behavior without claiming encoder parity, and CTest does not register real-model parity for that build. CUDA/Metal overlay configuration checks run in either mode, on any host.

A build needing a missing GPU fails rather than silently switching backends. Expected malformed-GGUF messages from negative tests are not failures by themselves; tracebacks, failed assertions and CTest failures are. The synthetic tests report the first failing layer, fixture and numerical error.

## Remaining Python footprint

- `convert/convert_hf_to_gguf.py`: standard-library-only, streaming converter.
- `tests/test_conversion.py`, `tests/tiny_checkpoint.py`: synthetic conversion/inference coverage and fixture construction.
- `tests/test_benchmark.py`: integration coverage for the native `gliner-bench`.
- `tests/parity_environment.py`: shared test-only CUDA precision settings.
- `tests/test_parity.py`: the sole optional upstream ML reference and golden generator.

The exploratory Python performance harness, its tests and the redundant label-state export example have been removed. Historical CPU/CUDA/Metal comparison results and links to the original harness remain in the [benchmark reports](benchmarks/README.md). The native C++ benchmark remains supported.
