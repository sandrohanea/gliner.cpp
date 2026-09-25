# gliner.cpp

GGML/GGUF inference for [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) text classification. CPU is the reference backend. Real-checkpoint F32 CUDA parity has been reported on an RTX 4060 Laptop GPU under the precision settings below, and F32 Metal parity has been verified on an Apple M4 Pro. Broader hardware, F16, memory, and performance validation remain pending. The CLI runs entirely in C++ from one GGUF file: schema construction, Unicode/Unigram tokenization, the 24-layer DeBERTa-v3-large encoder, contextual `[L]` marker gathering, and the shared classification head.

Despite its name, the published checkpoint uses the **span** architecture. Its classification head is `Linear(H, 2H)`, ReLU, `Linear(2H, 1)` on contextual marker states, not standalone label embeddings.

## Build

Requires CMake 3.20+, a C++17 compiler, GGML and utf8proc. Missing dependencies are fetched at pinned revisions: GGML v0.24.0 and utf8proc v2.10.0. Python 3.10+ is needed only for conversion and the synthetic tests.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

On Windows with Visual Studio, the executable is `build\Release\gliner-classify.exe`. With a single-configuration generator it is `build/gliner-classify`. Fetched dependency DLLs are placed alongside the executable on Windows.

GGML resolution is: a local `ggml/` checkout (or `-DGLINER_GGML_SOURCE_DIR=...`), an installed GGML CMake package, installed libraries, then the pinned fetch fallback. For installed dependencies:

```sh
cmake -S . -B build -DGLINER_FETCH_GGML=OFF -DGLINER_FETCH_UTF8PROC=OFF -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Installed shared libraries must be discoverable by the platform loader (`PATH` on Windows). Third-party sources are not bundled; see [GGML's license](https://github.com/ggml-org/ggml/blob/master/LICENSE) and [utf8proc's license](https://github.com/JuliaStrings/utf8proc/blob/master/LICENSE.md).

## Backends: CPU, CUDA and Metal

The backend is selected **at build time**, using the same `GGML_CUDA` and `GGML_METAL` CMake options used by whisper.cpp. There are no runtime backend/device parameters or `--backend`/`--device` switches. With both options off, the binary uses CPU; CUDA is off by default, and Metal defaults on for Apple platforms. Enable only one GPU backend per build.

Read the binary's fixed backend without loading a checkpoint or initializing a device:

```powershell
.\build\Release\gliner-classify.exe --build-info
```

Output is JSON, for example `{"backend":"cpu"}`. At initialization the runtime uses the first available device of that compiled backend. `--inspect` and `--debug` report the backend and the actual device. Switching backends requires using another build.

The model's weights, all state scratch buffers and graph operations use that device. Tokenization remains on the CPU. GPU matrix multiplications request F32 activation inputs and accumulation unless the explicit fast Metal build option below is selected; `--threads` controls CPU execution only. There is **no silent CPU fallback**, partial offload or multi-GPU splitting. A missing device, allocation failure or unsupported operation is an error. Weights plus inference scratch must fit on the device; the real F32 model alone needs roughly 1.8 GiB, and attention memory grows quadratically with token count.

For an explicit CPU build on any platform:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF
cmake --build build-cpu --config Release -j
```

**CUDA (NVIDIA, Windows/Linux):** install a compatible NVIDIA driver, CUDA toolkit and host C++ compiler, then enable GGML's CUDA backend when building GGML from source:

```powershell
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_METAL=OFF
cmake --build build-cuda --config Release -j
$env:NVIDIA_TF32_OVERRIDE = "0"
$env:GGML_CUDA_CUBLAS_COMPUTE_TYPE = "f32"
.\build-cuda\Release\gliner-classify.exe --model .\models\decide.gguf --task intent --labels refund_request,other --text "Refund my order"
```

For a single-configuration generator, omit `Release` from the executable path. Use `CMAKE_CUDA_ARCHITECTURES` when targeting a specific GPU architecture.

**CUDA precision requirement:** the pinned GGML cuBLAS backend enables TF32 math, even when graph operations request F32 accumulation. TF32 can round multiplication inputs enough to exceed the F32 parity tolerance. Set `NVIDIA_TF32_OVERRIDE=0` **before launching the application** to disable it; setting `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` also prevents an inherited GGML override from selecting F16/BF16 cuBLAS computation. On Linux, prefix the invocation with `NVIDIA_TF32_OVERRIDE=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`.

The CLI and C library do not modify their host's process-wide CUDA environment. Applications embedding gliner must set these variables before initializing CUDA/cuBLAS, including through other libraries. Running without these settings retains GGML's faster reduced-precision defaults and is not covered by the strict parity claim. Disabling TF32 may reduce throughput.

On **2026-09-25**, the user reported all **13 real-checkpoint F32 parity cases passing on Windows with an RTX 4060 Laptop GPU (compute capability 8.9, 8187 MiB VRAM, VMM enabled)** after the MMF fix below, with both CUDA precision settings shown above. Token IDs, marker positions and task offsets matched exactly; all 25 hidden-state outputs, contextual label states, logits, per-task probabilities and decisions passed the unchanged `atol=rtol=3e-4` comparisons. The maximum absolute hidden-state difference across the run was `0.00010967255` (about `1.10e-4`); the previously failing 12-token case was `1.1920929e-5`. This is a user-supplied hardware result, not a run on this development host. It does not establish real-checkpoint F16 parity, Linux/other-GPU coverage or performance.

The same host also passed all three expanded CUDA CTest tests: `gliner-core`, `gliner-cuda-precision-config` and `gliner-conversion`, including the 32-wide short/long-sequence fixtures. The reported suite duration was 49.59 seconds; this is test-harness duration, not an inference benchmark.

| Recorded build detail | Value |
|---|---|
| gliner.cpp revision | `ee9c8a495c224c4fb9b0aab2ca61f1bb0af09765` |
| GGML revision | `456172ec733a135778adcd32d00e576a58232e45` (v0.24.0, with the generated dispatch overlay) |
| CUDA compiler/toolkit | CUDA 13.4, `nvcc V13.4.59` |
| NVIDIA Windows driver | `32.0.15.9155` (reported by `Win32_VideoController`) |

**Short-sequence CUDA fix:** the pinned GGML also has a custom small-matrix (MMF) kernel using explicit TF32 instructions, outside cuBLAS. The cuBLAS environment override does not affect it. A real 22-token case passed on the RTX 4060 with those settings, but the 12-token case failed at encoder layer 1. CMake now builds a generated copy of GGML's CUDA dispatch file that bypasses MMF when F32 activation inputs are requested, allowing the FP32-capable cuBLAS path instead. The original GGML source checkout is not modified. This targeted build-directory fix requires local/fetched GGML source; prebuilt CUDA GGML packages are rejected until an upstream equivalent can be relied on. Unknown dispatch layouts fail configuration rather than silently omitting the fix. CPU and Metal builds do not use the overlay.

After updating these changes, **reconfigure and rebuild** the CUDA build; changing only the environment does not repair an older binary's MMF dispatch. Configuration prints `gliner: applied CUDA F32-input precision fix in the build directory`. Keep the cuBLAS precision environment settings as well. The new 32-wide synthetic fixtures test sequences below and above the small-matrix cutoff. The full real-model run reported above confirms the rebuilt runtime's short-sequence fix on the RTX 4060 Laptop GPU.

**Metal (Apple GPUs, macOS):** use the Xcode command-line tools and Metal toolchain:

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build-metal -j
```

Run that build's `gliner-classify` with the normal model/task/text arguments; it uses Metal without an additional flag. Embedding the Metal library avoids a separate shader-file deployment step.

**Metal F32 precision fix:** GGML v0.24.0's `kernel_mul_mm_f32_f32` converts both F32 operands to F16. This exceeded the unchanged synthetic F32 tolerance in a long attention case. The default `GLINER_METAL_STRICT_F32=ON` build marks matrix products for F32 accumulation and F32 activation inputs, and generates a build-directory overlay of GGML's Metal matrix dispatch so F32×F32 products use its F32 matrix-vector kernel. The GGML source checkout is not modified. This path can be slower than GGML's default F16-based matrix kernel. Strict Metal builds require a local or fetched GGML source checkout; prebuilt Metal GGML packages cannot receive the fix. The overlay checks the pinned dispatch layout and fails configuration if it changes.

For faster Metal inference when strict F32 parity is not required, use a **separate build directory**:

```sh
cmake -S . -B build-metal-fast -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGLINER_METAL_STRICT_F32=OFF
cmake --build build-metal-fast -j
```

This option removes both the Metal dispatch overlay and the graph's F32 matrix precision requests. The model file remains F32, but GGML's stock Metal matrix kernel narrows multiplication operands to F16. It is a build-time choice; benchmark JSON records `metal_precision_mode` as `strict_f32` or `fast`. On the M4 Pro two-task workload, a build using this option averaged 38.94 ms joint and 65.41 ms separate versus 215.92 ms and 343.76 ms with the precision fix in an earlier run. The fast build **failed** the synthetic F32 encoder parity test at layer 1. Its CTest conversion/API checks still run, while strict encoder goldens and optional real-checkpoint parity are excluded from fast-build CTest. Run a strict build for parity validation. See the [M4 Pro benchmark report](docs/benchmarks/2026-09-25-macos-m4-pro.md) for the workload and accuracy comparison.

`GGML_CUDA`/`GGML_METAL` configure **local or fetched GGML source builds** and fix gliner's execution backend. CUDA and strict Metal builds require GGML source for their precision fixes; a Metal-enabled installed GGML CMake package can be used by the fast mode. Installed GGML and unconfigured bare GGML libraries remain supported for CPU builds. F32 parity has been exercised on an M4 Pro for strict Metal and reported on an RTX 4060 Laptop GPU for CUDA. Further release gates are tracked in [NEXT_STEPS.md](NEXT_STEPS.md).

## Convert

Download the model repository into a local directory containing `config.json`, `encoder_config/config.json`, `tokenizer.json`, and `model.safetensors` (or sharded safetensors and its index):

```sh
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="fastino/GLiNER2.5-Decide", revision="7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6", local_dir="models/GLiNER2.5-Decide", allow_patterns=["config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors", "model.safetensors.index.json"])'
python convert/convert_hf_to_gguf.py models/GLiNER2.5-Decide models/decide.gguf
build/gliner-classify --model models/decide.gguf --inspect
```

The download command requires `huggingface_hub`; conversion itself has no Python package dependencies. Checkpoints and GGUF outputs are ignored by Git.

The converter is streaming and **standard-library-only**. It preserves every F32/F16 tensor, including unused span/counting weights, without quantization. It validates tensor shapes, dtypes, offsets, shard indexes, required encoder/classifier tensors and the supported tokenizer/encoder configuration. GGUF v3 contains application format version 2, including exact Unigram scores and Python Unicode lowercasing/word-character tables.

Older GGUF files remain usable with `--states`, but must be **reconverted** for text inference. Incomplete or unsupported version-2 models fail at load time rather than returning plausible scores.

## Classify text

```sh
build/gliner-classify --model models/decide.gguf \
  --task intent --labels refund_request,order_status,other \
  --text "Please refund my order" --threads 4
```

The verified F32 checkpoint selects `refund_request`, with logits approximately `2.57794`, `-2.82484`, and `-2.92263`. Output is JSON containing the selected label and each label's raw logit and probability.

Use repeated `--label` for labels containing commas, or `--text-file` for UTF-8 files. Empty text is supported, matching the upstream processor's punctuation normalization. Text words are lowercased; task names, label names and prompts retain their case. No `[CLS]`/`[SEP]` wrapper is inserted, matching upstream's segment-by-segment tokenization.

```sh
build/gliner-classify --model models/decide.gguf \
  --task topic --labels billing,delivery,other \
  --prompt "Choose all relevant topics" \
  --description "billing=Invoices, payments and refunds" \
  --description "delivery=Shipping and order tracking" \
  --text "Refund the delivery charge" --multi-label --threshold 0.5
```

Descriptions are serialized in label order; match that insertion order in Python's `label_descriptions` dictionary when comparing logits. An explicitly empty description is preserved. Literal `[SEP_TEXT]`/`[SEP_STRUCT]` task and label names are rejected because they collide with upstream schema routing.

| Option | Behavior |
|---|---|
| `--build-info` | Report the fixed build backend as JSON, without loading a model. Use alone. |
| `--task NAME` | Starts a classification task; repeat for multiple questions on the same text. |
| `--activation sigmoid\|softmax` | Current task: text defaults to exclusive softmax. `--multi-label` and `--states` default to independent sigmoid. |
| `--multi-label --threshold T` | Current task: returns a `labels` array, in input order, with probabilities at least `T` (default 0.5). |
| `--temperature T` | Current task's positive probability temperature; raw logits are unchanged. |
| `--max-words N` | Upstream `max_len`-style word truncation before subword encoding; 0 means unlimited. |
| `--max-tokens N` | Hard limit over all task schemas plus the shared text, default 512, supported range 1..4096. Exceeding it is an error, not silent schema truncation. |
| `--debug` | Adds exact token IDs, label positions, flattened contextual label states and task score offsets to JSON. |
| `--dump-hidden FILE` | Writes embeddings and every encoder layer for parity diagnostics. Header: LE u32 token count and hidden size; payload: LE float32 `[layers+1, tokens, hidden]`. Refuses to overwrite an existing file. |

Long sequences can require substantial memory because attention is quadratic. This implementation does not chunk long documents automatically.

## Multiple questions on one input

Repeat `--task` to classify the same input jointly, in **one encoder pass**:

```powershell
.\build\Release\gliner-classify.exe --model .\models\decide.gguf `
  --text "Please refund my order; the delivery was late and I am upset." `
  --task intent --labels refund_request,order_status,other `
  --task sentiment --labels positive,neutral,negative `
  --task urgent --labels yes,no --threads 4
```

This example returns `refund_request`, `negative`, and `yes` on the verified F32 model. Multiple tasks produce a JSON `tasks` array in request order; each object has `task`, `label` (or `labels` for multi-label mode), and its own `scores`. A single task retains the existing top-level `label`/`scores` output.

Task-local options (`--labels`/`--label`, `--prompt`, `--description`, `--multi-label`, `--threshold`, `--temperature`, `--activation`) apply to the most recent `--task` and reset for the next one. Options before the first `--task` belong to that first task, preserving the single-task CLI. Text, thread count, word/token limits and diagnostics apply to the entire request. Each task needs a unique nonempty name and its own nonempty label list; labels must be unique within a task but can repeat across tasks.

For example, one task can use multi-label sigmoid while another uses exclusive softmax:

```powershell
.\build\Release\gliner-classify.exe --model .\models\decide.gguf `
  --text-file .\message.txt `
  --task topics --labels billing,delivery,other --multi-label --threshold 0.5 `
  --task sentiment --labels positive,neutral,negative --threads 4
```

The CLI loads the **model once** and reads the input file once. It joins schemas using `[SEP_STRUCT]`, appends `[SEP_TEXT]` and one tokenized copy of the text, then gathers every task's contextual label states from one encoder result. Softmax/threshold decisions are applied separately within each task, never across all labels. `--debug` adds `task_offsets` of length `task_count + 1`; consecutive entries delimit each task's labels in the flattened diagnostics.

**Joint scores are not independent-call scores.** Tasks, labels and text all attend to one another; adding, removing or reordering a task may change another task's logits. This matches upstream's multi-task schema behavior. All schemas share the token budget; exceeding it fails rather than splitting or dropping tasks. A larger joint schema still increases attention memory and may not be faster than separate calls for every workload.

This is one document with multiple questions, not a padded batch of multiple documents. In the C API, reuse the same context and state across requests to retain model weights and reuse scratch allocations. Input text is not cached across calls, and changing the schema requires a fresh encoder pass.

## Performance benchmark

`gliner-bench` is built alongside `gliner-classify` and uses the same synchronous C API for **CPU, CUDA and Metal**, selected at build time. It loads the model once, reads the text once and reuses execution states. Published device measurements are collected in [Benchmark results](docs/benchmarks/README.md), including the [Windows CPU/CUDA and Python comparison](docs/benchmarks/2026-09-25-windows-rtx4060.md) and [Apple M4 Pro CPU/Metal and Python comparison](docs/benchmarks/2026-09-25-macos-m4-pro.md).

Start with a Release CPU build:

```powershell
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF
cmake --build build-cpu --config Release -j 4

.\build-cpu\Release\gliner-bench.exe --model .\models\decide.gguf `
  --text "Please refund my order; the delivery was late." `
  --task intent --labels refund_request,order_status,other `
  --task sentiment --labels positive,neutral,negative `
  --threads 4 --mode both --warmup 3 --iterations 20 `
  > .\build-cpu\bench-cpu.json

$bench = Get-Content .\build-cpu\bench-cpu.json -Raw | ConvertFrom-Json
$bench | Select-Object backend, device, build_type, model_load_ms
$bench.results | Select-Object mode, tokens_per_request,
  @{Name="median_ms"; Expression={$_.latency_ms.median}},
  @{Name="p95_ms"; Expression={$_.latency_ms.p95}},
  requests_per_second, questions_per_second | Format-Table
```

With a single-configuration generator (typically Linux/macOS), use the executable directly under the build directory, without `Release`. For CUDA/Metal comparisons, run that backend's `gliner-bench` with **the same GGUF, text, task order, labels, prompts, limits and repetition counts**. The CUDA benchmark needs the documented `NVIDIA_TF32_OVERRIDE=0` and `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` launch environment. It records both variables in JSON and warns if those settings are absent; it never modifies them. Backend/device selection remains build-time only.

| Option | Meaning |
|---|---|
| `--mode joint\|separate\|both` | Default `both`. Joint answers all questions in one pass; separate runs each question independently against the same loaded model. |
| `--warmup N` | Additional unmeasured requests after the first request; default 3, may be 0. |
| `--iterations N` | Measured requests **per mode**; default 20, must be positive. Use more repetitions for meaningful tail percentiles. |
| `--text` / `--text-file` | Exactly one input, including an empty text if desired. File I/O occurs once, outside inference timing. |
| `--task`, `--labels` / `--label`, `--prompt`, `--description` | Same task grouping as `gliner-classify`. Decoding options are not accepted: the benchmark measures raw classification, not decisions or formatting. |
| `--threads`, `--max-words`, `--max-tokens` | Same inference controls as the CLI. The token limit applies per encoder pass, so joint mode must fit all schemas together. |

**What the numbers measure:**

- `model_load_ms`: one model initialization, including metadata/tokenizer/weights and backend initialization/transfers. It is not a guaranteed cold-disk measurement; OS caches may already contain the model.
- `state_init_ms` and `first_request_ms`: reported separately for each mode. The first request includes initial graph/scratch allocation and possible kernel initialization, and is excluded from the measured samples. Then come `warmup_requests` additional requests, also excluded.
- `latency_ms`: min, mean, median, p95 and max across `iterations` samples, using a monotonic clock. Percentiles linearly interpolate at `(n - 1) * p`; a one-sample run reports that same value for every percentile. `samples_ms` retains every measured latency.
- `requests_per_second`: `1000 / mean_latency_ms`. One request answers **all questions**, so separate-mode latency includes all sequential calls. `questions_per_second` multiplies by task count. `encoded_tokens_per_second` counts actual schema-plus-text encoder work, including repeated text in separate mode; it is not unique-document-token throughput.

Timing includes tokenization, graph construction/allocation reuse, synchronized backend execution and the C API's result transfers. GPU timings therefore include completion, not just asynchronous kernel submission. Result checks, JSON formatting, file reading, model loading and state initialization are outside the measured request samples. The output also records backend/device, Metal precision mode, compiler/build configuration, model path/file size, dimensions, thread count, task definitions, limits, token counts and last logits. It reports text byte length but does not embed the text; retain the input separately for reproducibility.

Joint mode uses one state. Separate mode reuses one state per question to avoid cycling different shapes through the same allocator, while sharing the **same immutable model weights**. Modes run sequentially (`joint` then `separate` for `both`) and their states are freed between modes. The later mode may benefit from process/device caches initialized earlier; its first request is not a fresh-process cold start. For isolated measurements, use separate `--mode joint` and `--mode separate` processes and keep the machine idle. Joint and separate outputs can differ by design; their timing comparison is not a semantic-equivalence claim.

The benchmark does **not** yet sample peak RAM/VRAM, measure concurrent-request serving, generate token-length sweeps or compare result files automatically. `model_file_bytes` is file size, not resident or peak memory. Use external memory monitoring and explicit input files for short/medium/long workloads. Keep CPU/GPU precision settings consistent before interpreting speedups.

The [dated Windows report](docs/benchmarks/2026-09-25-windows-rtx4060.md) and [M4 Pro report](docs/benchmarks/2026-09-25-macos-m4-pro.md) contain the exact two-question workload, results, loading/first-request timings, correctness evidence, reproduction commands and comparison caveats.

### Optional Python CPU/CUDA comparison

`tests/bench_python.py` is an optional comparison helper using the existing `tests/requirements-parity.txt` environment. It reuses the upstream processor, Transformers encoder and classification head from the parity loader, once, in **float32 eval/inference mode**. The default is CPU; the Python-only `--device cuda` or `--device cuda:N` selects a CUDA GPU and fails explicitly if unavailable, without falling back to CPU. It accepts the same text, task, prompt, description, thread/limit, mode, warmup and iteration options as `gliner-bench`; supply the local Hugging Face directory with `--checkpoint` instead of a GGUF with `--model`. This does not change the C++ runtime's build-time backend selection.

Run on the **same machine**, with other benchmarks/builds stopped:

```powershell
$workload = @(
  "--text", "Please refund my order; the delivery was late.",
  "--task", "intent", "--labels", "refund_request,order_status,other",
  "--task", "sentiment", "--labels", "positive,neutral,negative",
  "--threads", "4", "--mode", "both", "--warmup", "3", "--iterations", "20"
)
.\build-cpu\Release\gliner-bench.exe --model .\models\decide.gguf @workload `
  > .\build-cpu\bench-cpu-comparison.json
.\build-oracle\Scripts\python.exe .\tests\bench_python.py `
  --checkpoint .\models\GLiNER2.5-Decide @workload `
  > .\build-oracle\bench-python-cpu.json

$cpp = Get-Content .\build-cpu\bench-cpu-comparison.json -Raw | ConvertFrom-Json
$python = Get-Content .\build-oracle\bench-python-cpu.json -Raw | ConvertFrom-Json
foreach ($row in $python.results) {
  $baseline = $cpp.results | Where-Object mode -eq $row.mode
  [pscustomobject]@{
    Mode = $row.mode
    CppMeanMs = $baseline.latency_ms.mean
    PythonMeanMs = $row.latency_ms.mean
    CppSpeedup = $row.latency_ms.mean / $baseline.latency_ms.mean
  }
}
```

Use a C++ result file from those exact arguments: token counts should be `[40]` for joint and `[27,23]` for separate, and each mode's `last_logits` should agree within the established F32 tolerance. `CppSpeedup` above 1 means C++ was faster. Different computers or different task/text configurations are not comparable.

The Python request timer includes collation/tokenization, input transfers, encoder execution, upstream marker extraction, per-task classification and materialization of logits/sigmoid probabilities on the host. It excludes imports, model loading, initial schema construction, result validation and JSON formatting. CUDA runs synchronize the selected device before starting and before stopping each timer; warmups are also synchronized. Model weights move to the GPU once during loading, not per request. Like C++, it reports the first request separately, then runs additional warmups followed by measured samples, using the same median/p95 definition and per-request throughput formulas. Each request answers all questions; separate mode uses multiple encoder passes without reloading the model. No per-layer outputs, autocast, tracing or compilation are enabled.

Python sets PyTorch intra-op threads to `--threads` and inter-op threads to 1; the JSON records the effective counts, package versions and PyTorch build configuration. Upstream's normal per-segment tokenizer cache remains active across requests and modes (C++ currently has no equivalent cache). There is no explicit Python state allocator: `state_count` is 0 and `state_init_ms` is null rather than claiming equivalence to C++ state initialization.

**Compare steady-state inference, not loader timings, across implementations.** This helper loads only the upstream classification components, not unused span/counting modules or the full `AutoExtractor` facade. Safetensors can map weights lazily, whereas GGUF loading copies the required weights; imports and first-use page faults also fall in different phases. CUDA model loading includes device initialization and synchronized weight transfers. `model_load_ms` therefore does not measure equivalent full-application startup costs. Peak memory, Python Metal/MPS benchmarking, reduced-precision modes and a general cross-machine speedup claim are out of scope for this helper.

For Python CUDA, first check your environment:

```powershell
.\build-oracle\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If it is CPU-only, use a separate environment to preserve the existing CPU baseline. In the user's restricted environment, use the approved package index `https://packagefeedproxy.microsoft.io/pypi/simple/`, not the blocked external PyTorch registry.

**CUDA wheel availability:** when checked on 2026-09-25, the proxy listed `torch-2.6.0-cp312-cp312-win_amd64.whl` but no CUDA-tagged PyTorch wheels, including `2.6.0+cu126`. The plain version is not proof of CUDA support. An approved feed must mirror the CUDA wheel (or supply an approved local wheel) before the CUDA install below can succeed; otherwise pip should fail rather than substitute the CPU build.

```powershell
py -3.12 -m venv build-oracle-cuda
# Requires torch 2.6.0+cu126 to have been mirrored into the approved feed.
.\build-oracle-cuda\Scripts\python.exe -m pip install --upgrade "torch==2.6.0+cu126" `
  --index-url https://packagefeedproxy.microsoft.io/pypi/simple/
.\build-oracle-cuda\Scripts\python.exe -m pip install -r .\tests\requirements-parity.txt `
  --index-url https://packagefeedproxy.microsoft.io/pypi/simple/
.\build-oracle-cuda\Scripts\python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The explicit `+cu126` suffix matters when repairing an existing environment: `torch==2.6.0` alone can already be satisfied by `2.6.0+cpu`. Expect `2.6.0+cu126`, `12.6`, and `True` from the check, and use that same interpreter for the benchmark. A `+cpu` version with `torch.version.cuda == None` means the wrong wheel or interpreter, not a failed C++ CUDA build. The parity requirements accept the CUDA wheel and do not require replacing it with the CPU build. Do not add the blocked registry as an extra index to work around its unavailability.

The wheel supplies its CUDA runtime (12.6 here); it need not match the locally installed `nvcc` toolkit used to compile GGML. The NVIDIA driver must support it. The benchmark records PyTorch's CUDA runtime, GPU name, compute capability and precision settings so that this stack difference is visible.

Using the same `$workload` array from the CPU example:

```powershell
$env:NVIDIA_TF32_OVERRIDE = "0"
$env:GGML_CUDA_CUBLAS_COMPUTE_TYPE = "f32"
.\build-cuda\Release\gliner-bench.exe --model .\models\decide.gguf @workload `
  > .\build-cuda\bench-cuda-comparison.json
.\build-oracle-cuda\Scripts\python.exe .\tests\bench_python.py `
  --checkpoint .\models\GLiNER2.5-Decide --device cuda @workload `
  > .\build-oracle-cuda\bench-python-cuda.json
```

Use `build-oracle` instead if it already has a working CUDA-enabled PyTorch. The Python CUDA path explicitly selects PyTorch's highest F32 matmul precision and disables TF32 for matmul and cuDNN; keep the environment settings above for the C++ comparison. It does not mutate the invoking shell's environment. Compare CUDA against CUDA, including token counts and last logits, and run the programs sequentially. Python CUDA has unit coverage for timing/transfers plus the [user-supplied RTX 4060 hardware benchmark](docs/benchmarks/2026-09-25-windows-rtx4060.md); this macOS host has no CUDA device.

Standard-library parser/timing tests run in CTest without ML dependencies. The optional live Python/C++ comparison can be run with:

```powershell
.\build-oracle\Scripts\python.exe .\tests\test_python_benchmark.py `
  --cpp-bench .\build-cpu\Release\gliner-bench.exe `
  --converter .\convert\convert_hf_to_gguf.py
```

On CUDA hardware, the same live comparison accepts `--device cuda` with the CUDA `gliner-bench` and a CUDA-enabled Python environment; it fails rather than skipping when CUDA is unavailable.

## Python parity

Real-checkpoint F32 parity has been exercised against checkpoint revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`, upstream GLiNER2 commit `1a80c9c85272cd6809009b102b770c452e928196`, Transformers 4.48.1/4.48.2 and PyTorch 2.6.0. Eight single-task cases cover ordinary and empty text, Unicode, punctuation, URLs/email, prompts/descriptions containing special markers, empty descriptions, word truncation and logarithmic relative-position buckets. Five additional joint-schema cases cover multiple tasks with different label counts, repeated labels across tasks, prefix-overlapping task names, task order, mixed decoding modes and shared text truncation. The current processor/scorer and span inference semantics were also checked at upstream commit `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`.

Token IDs, marker positions and task score offsets match exactly. All 25 hidden-state outputs (embeddings plus 24 layers), contextual label states, raw logits, per-task probabilities and chosen labels are compared. On CPU, the observed maximum absolute hidden-state difference was below `3e-5` for single tasks and `1e-4` for joint schemas. The user-reported RTX 4060 Laptop CUDA run passed all 13 cases with a maximum difference of `0.00010967255`. On an Apple M4 Pro with pinned GGML v0.24.0 and the Metal precision overlay, all 13 F32 cases passed with maximum difference `8.87e-5`. Both GPU runs use the unchanged live comparison tolerance `atol=rtol=3e-4`. This is numerical parity for the supported classification path and documented configurations, not a claim that every upstream API or decoder is implemented.

The normal CTest suite needs no ML dependencies or downloaded weights. It replays checked-in, deterministic single-task and joint-schema goldens on a complete tiny encoder, checks F32/F16 storage, the C ABI, task grouping/decoding, conversion preservation and error paths. A callback-count assertion verifies one encoder traversal per multi-task call. It also compares file, buffer and non-seekable short-read stream initialization for both single-task and joint inference, including source lifetimes, malformed/truncated input and loader cleanup. Tiny F32 uses `atol=rtol=5e-5`; F16 storage is compared to F32 goldens with `5e-3`. **Real-checkpoint F16 parity has not been established.**

`tiny_cuda_parity.json` adds 32-wide encoder fixtures with 10- and 30-token inputs, covering CUDA's short-matrix dispatch boundary that the original 8-wide fixtures could not exercise. Regenerate them with `tests/test_parity.py --kernel-only --write-golden tests/fixtures/tiny_cuda_parity.json`. CUDA and Metal source-overlay configuration tests check that GGML checkouts remain unchanged; hardware numerical parity is checked separately.

For the optional live oracle, use a separate Python 3.12 environment:

```sh
python -m pip install -r tests/requirements-parity.txt
python tests/test_parity.py --checkpoint models/GLiNER2.5-Decide \
  --gguf models/decide.gguf --binary build-metal/gliner-classify \
  --expect-backend metal
```

To include that comparison in CTest, configure `GLINER_PARITY_CHECKPOINT` and `GLINER_PARITY_GGUF` with absolute paths and select the oracle environment with `Python3_EXECUTABLE`. The oracle runs all 13 cases by default; `--single-only` and `--batch-only` select a subset. Regenerate synthetic fixtures with `tests/test_parity.py --single-only --write-golden tests/fixtures/tiny_parity.json` or `--batch-only --write-golden tests/fixtures/tiny_batch_parity.json`; provenance is stored inside each fixture.

Parity runs use the binary's compiled backend. Both scripts accept `--expect-backend cpu|cuda|metal` to assert which build is under test; this option does not select or change the backend. CTest supplies that assertion automatically. CUDA CTest runs and standalone parity scripts explicitly set `NVIDIA_TF32_OVERRIDE=0` and `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` for their child processes, without changing the invoking shell. CPU/Metal runs leave those environment settings alone.

Keep separate CPU/CUDA/Metal build directories and run CTest in each; a GPU build without its required device fails rather than skipping. The synthetic parity test reports backend, storage dtype, fixture, first failing encoder layer and the numerical error/allowed error. Layer 0 is embeddings. GPU tests enforce the same tolerances as CPU; hardware-specific tolerances must be measured and documented before changing them. The reported CUDA result and measured M4 Pro Metal result establish F32 numerical parity for these fixtures and real-checkpoint cases on those devices. They do not establish real-checkpoint F16 parity or a speedup. Malformed-GGUF messages from negative loader tests are expected; a traceback or CTest failure is not.

For classification-head-only comparisons, export contextual states with the upstream model:

```sh
python examples/export_label_states.py --checkpoint models/GLiNER2.5-Decide \
  --text "Please refund my order" --task intent \
  --labels refund_request order_status other --output label_states.txt
build/gliner-classify --model models/decide.gguf \
  --labels refund_request,order_status,other --states label_states.txt
```

Each states-file row contains `hidden_size` floats for one contextual `[L]` marker, in label order. This legacy path retains sigmoid probabilities by default.

## C API

The public [gliner.h](include/gliner/gliner.h) is valid C. A context owns immutable model data; each state owns its backend, allocator, scratch space and results. Keep the context alive until all its states are freed. A state is used by one thread at a time; separate states can share a context.

All initializers use the build's fixed backend. `gliner_build_backend()` reports it without initializing devices; `gliner_model_backend_name(ctx)` and `gliner_model_device_name(ctx)` report a loaded model's identity. No backend selection is exposed through initialization parameters.

```c
struct gliner_context * ctx = gliner_init_from_file("models/decide.gguf");
if (!ctx) { fprintf(stderr, "%s\n", gliner_last_error()); return 1; }
struct gliner_state * state = gliner_init_state(ctx);
if (!state) {
    fprintf(stderr, "%s\n", gliner_last_error());
    gliner_free(ctx);
    return 1;
}
const char * labels[] = {"refund_request", "order_status", "other"};
struct gliner_text_params params = gliner_default_text_params();
params.n_threads = 4;
int status = gliner_classify_text(ctx, state, "Please refund my order",
                                 "intent", labels, 3, &params);
if (status == GLINER_STATUS_OK) {
    const struct gliner_score * scores = gliner_get_scores(state);
    printf("refund logit: %f\n", scores[0].logit);
} else {
    fprintf(stderr, "%s\n", gliner_last_error());
}
gliner_free_state(state);
gliner_free(ctx);
```

Include `gliner/gliner.h` and `stdio.h` in the calling source. `gliner_score_label_states` remains available. C API probabilities always mean independent sigmoid; exclusive softmax is a CLI decoding choice. Diagnostic getters expose the last successful text call's token IDs, marker positions and label states; an optional callback exposes each layer. Result pointers are valid until the next inference call or state destruction. Failures clear previous results and report a thread-local error; exceptions do not cross the ABI.

### Joint classification API

Given an already loaded `ctx` and initialized `state`, supply one text and an ordered array of tasks:

```c
const char * intents[] = {"refund_request", "order_status", "other"};
const char * sentiments[] = {"positive", "neutral", "negative"};
struct gliner_classification_task tasks[] = {
    {"intent", intents, 3, NULL, NULL},
    {"sentiment", sentiments, 3, "How does the customer feel?", NULL}
};
struct gliner_batch_params batch = gliner_default_batch_params();
batch.n_threads = 4;
int status = gliner_classify_text_batch(ctx, state, "Please refund my order",
                                      tasks, 2, &batch);
if (status == GLINER_STATUS_OK) {
    const struct gliner_score * scores = gliner_get_scores(state);
    const struct gliner_task_result * results = gliner_get_task_results(state);
    for (int t = 0; t < gliner_n_tasks(state); ++t) {
        for (int i = 0; i < results[t].n_scores; ++i) {
            printf("%s / %s: %f\n", tasks[t].task, tasks[t].labels[i],
                   scores[results[t].score_offset + i].logit);
        }
    }
} else {
    fprintf(stderr, "%s\n", gliner_last_error());
}
```

`gliner_classification_task` holds the task name, label pointers/count, optional prompt and optional descriptions in label order. `gliner_batch_params` controls CPU threads and the shared token/word limits. All request pointers are borrowed only for the call. Tasks with zero labels, duplicate task names, duplicate labels within a task, malformed input or a combined token-limit violation fail the entire call and clear all results.

`gliner_get_scores` returns all labels flattened in task order, then label order. Each `gliner_task_result` is an offset/count into that array, also applicable to marker positions and contextual label states. `gliner_n_tasks`/`gliner_get_task_results` expose one range after the existing single-task API and none after states-only scoring or a failure. Result pointers expire at the next inference call or state destruction. C API probabilities remain independent sigmoid; apply softmax separately per task if desired. No additional model initialization is needed between calls.

### Initialize from a byte buffer

```c
struct gliner_context * ctx = gliner_init_from_buffer(model_bytes, model_size);
if (!ctx) { fprintf(stderr, "%s\n", gliner_last_error()); return 1; }
/* model_bytes may now be freed; create states with gliner_init_state(ctx). */
```

`model_bytes` is a `const void *` to GGUF bytes and `model_size` is their `size_t` length. The buffer is borrowed only for the synchronous initialization call; metadata and weights are owned by the returned context. A managed `byte[]` must be pinned or marshalled to a stable pointer for that call, but does not need to remain pinned afterwards. This is not a zero-copy/memory-mapped model.

### Initialize from a stream

`gliner_model_loader` follows whisper.cpp's `context`, `read`, `eof`, `close` callback pattern. `gliner_init(&loader)` accepts sequential, non-seekable streams and aggregates short reads. For example, a C `FILE *` adapter:

```c
static size_t model_read(void * context, void * output, size_t count) {
    return fread(output, 1, count, (FILE *)context);
}
static bool model_eof(void * context) {
    return feof((FILE *)context) != 0;
}
static void model_close(void * context) {
    fclose((FILE *)context);
}

/* Inside the caller: */
FILE * input = fopen("model.gguf", "rb");
if (!input) { perror("model.gguf"); return 1; }
struct gliner_model_loader loader = {input, model_read, model_eof, model_close};
struct gliner_context * ctx = gliner_init(&loader);
/* input has been closed, including on initialization failure. */
if (!ctx) { fprintf(stderr, "%s\n", gliner_last_error()); return 1; }
```

Start the stream at the GGUF header. All three callbacks are required; a missing callback rejects the loader without invoking any callbacks. Once accepted, `close` runs exactly once before initialization returns, on success or failure. Neither the loader nor its context is retained, and callbacks must not throw or re-enter initialization. `read` must return at most the requested count; a zero-byte read is an error or end-of-stream, not a temporary "try again" result. `eof` distinguishes premature EOF from a stalled/failed read.

Loading uses GGML's GGUF callback parser and at most 8 MiB tensor-transfer chunks (plus metadata and final model storage), with no temporary file or full-stream copy. Unused tensor payloads are consumed and validated too; the caller must supply the complete GGUF data. Trailing padding or bytes beyond the tensor payload need not be consumed. File, buffer and stream initializers leave state creation to the caller.

## Scope and remaining work

Supported: one text with one or multiple jointly encoded classification tasks per call, ordered labels, per-task prompts and ordered label descriptions, exclusive argmax/softmax and independent multi-label thresholding, with the published Decide DeBERTa configuration.

Not implemented: NER/span/JSON extraction, few-shot examples, constrained/beam/exact decision decoding, calibration fitting, automatic long-document chunking, alternate encoders/tokenizer pipelines, multi-document padded batching, quantization, memory-mapped weights, hybrid CPU/GPU offload or multi-GPU execution. Broader CUDA and Metal hardware validation remains pending. See [NEXT_STEPS.md](NEXT_STEPS.md).

The model checkpoint is [Apache 2.0 licensed](https://huggingface.co/fastino/GLiNER2.5-Decide); review its license when redistributing converted weights. Checkpoints and generated GGUF files stay out of Git.
