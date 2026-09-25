# gliner.cpp

GGML/GGUF inference for [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) text classification and grouped span extraction. CPU is the reference backend. Classification F32 CUDA parity has been reported on an RTX 4060 Laptop GPU under the precision settings below, and classification F32 Metal parity has been verified on an Apple M4 Pro. Native span extraction has CPU reference coverage; its GPU validation remains pending. Inference runs entirely in C++ from one GGUF file, including schema/tokenization, the DeBERTa encoder and task heads.

Despite its name, the published checkpoint uses the **span** architecture. Its classification head is `Linear(H, 2H)`, ReLU, `Linear(2H, 1)` on contextual marker states, not standalone label embeddings.

**No Python is needed to build or run the C++ library, CLI or benchmark from a GGUF model.** Python is limited to one-time Hugging Face conversion and optional developer tests/reference checks.

## Build

Requires CMake 3.20+, a C++17 compiler, GGML and utf8proc. Missing dependencies are fetched at pinned revisions: GGML v0.24.0 and utf8proc v2.10.0. Python 3.10+ is optional for synthetic integration tests and required only when running the converter.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

On Windows with Visual Studio, the executable is `build\Release\gliner-classify.exe`. With a single-configuration generator it is `build/gliner-classify`. Fetched dependency DLLs are placed alongside the executable on Windows.

The C API checks and CUDA/Metal source-overlay checks run natively. CMake also registers standard-library-only conversion/CLI tests when Python is available; pass `-DGLINER_PYTHON_TESTS=OFF` to disable Python test discovery entirely. This reduces test coverage, not runtime functionality. See [Testing and upstream parity](docs/testing.md).

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

Download the [model repository](https://huggingface.co/fastino/GLiNER2.5-Decide/tree/7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6) with your preferred client into a local directory containing `config.json`, `encoder_config/config.json`, `tokenizer.json`, and `model.safetensors` (or sharded safetensors and its index):

```sh
python convert/convert_hf_to_gguf.py models/GLiNER2.5-Decide models/decide.gguf
build/gliner-classify --model models/decide.gguf --inspect
```

Conversion has no Python package dependencies. Once converted, distribute/use the GGUF with the native runtime; no checkpoint directory or Python environment is needed for inference. Checkpoints and GGUF outputs are ignored by Git.

The converter is streaming and **standard-library-only**. It preserves every F32/F16 tensor without quantization. It validates tensor shapes, dtypes, offsets, shard indexes, required encoder/classifier tensors and the supported tokenizer/encoder configuration. Checkpoints declaring a `markerV0` span head also require complete span, count-gate and `count_lstm` conditioning tensors. GGUF v3 contains application format version 2, including exact Unigram scores, Python Unicode lowercasing/word-character tables and additive span metadata where available.

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

## Grouped span extraction

`gliner-extract` returns **verbatim substrings**, their half-open **UTF-8 byte offsets**, raw logits and sigmoid confidence, grouped by supplied extraction labels. All labels share one encoder pass and one loaded model. It follows upstream entity-schema scoring: contextual `[E]` markers, `markerV0` endpoint/output MLPs, the count gate and the first count-conditioned GRU/projector step. It is not the classification head applied to arbitrary words.

Reconvert an existing checkpoint with the updated converter to include the required span metadata:

```powershell
python .\convert\convert_hf_to_gguf.py .\models\GLiNER2.5-Decide .\models\decide-spans.gguf
.\build\Release\gliner-classify.exe --model .\models\decide-spans.gguf --inspect

.\build\Release\gliner-extract.exe --model .\models\decide-spans.gguf `
  --text "Tim Cook works at Apple in California." `
  --labels person,company,location `
  --description "person=A person name" `
  --description "company=A company name" `
  --description "location=A location name" --threads 4
```

This example matches upstream's `Tim Cook`, `Apple` and `California` spans. The output has a `groups` array in label order; each group contains `label` and `spans`, whose entries contain `text`, `start`, `end`, `logit` and `probability`. Empty groups remain present. `span_extraction: yes` in `--inspect` confirms model support. Older GGUF files still classify but cannot extract until reconverted, even if their span tensors were preserved.

To request both search-term groups from one input:

```powershell
.\build\Release\gliner-extract.exe --model .\models\decide-spans.gguf `
  --text-file .\request.txt `
  --label "D1 search term" `
  --description "D1 search term=Words or short phrases needed to search official documentation for the Cloudflare D1 query bound parameter limit. Include the product name and the property whose limit is requested." `
  --label "Email Sending search term" `
  --description "Email Sending search term=Words or short phrases needed to search official documentation for the Cloudflare Email Sending combined recipient limit. Include the product name and the property whose limit is requested." `
  --threshold 0.3 --top-k 5 --threads 4
```

**Quality caveat:** Decide is classification-focused. For the Cloudflare request used during development, those search-term descriptions produced empty groups at threshold 0.3 in both upstream and C++; a broader `product` label returned `Cloudflare` rather than the full desired product names. Native parity does not guarantee useful search keywords. Label/schema tuning or an extraction-focused checkpoint may be needed. The runtime does not fetch documents, discover topic groups automatically, generate missing terms, normalize plurals, or assemble search queries.

| Extraction option | Meaning |
|---|---|
| `--labels a,b` / repeated `--label NAME` | Unique extraction labels/groups, all evaluated together. |
| `--description NAME=TEXT` | Optional per-label guidance, encoded in label order. |
| `--threshold P` | Inclusive sigmoid cutoff in `[0,1]`, default 0.5. Confidence is not a calibrated retrieval-relevance score. |
| `--top-k N` | Maximum returned spans per label **after** overlap suppression; 0 means all, the default. |
| `--allow-overlap` | Disable confidence-first greedy overlap suppression within each label. Different labels may overlap regardless. |
| `--max-tokens N` / `--max-words N` | Same aggregate token budget and explicit text-word truncation as classification. No silent truncation. |
| `--debug` | Token IDs, label/word positions, candidate word ranges, all raw span/count logits and the predicted count. |

Results are ordered by label, then descending confidence with source-position tie breaking. A predicted count of zero suppresses all results, matching the upstream entity gate. Only the first conditioning step is needed for entity lists, even when the gate predicts a larger count. Repeated records use the separate structure mode below.

The published checkpoint permits spans of at most **8 upstream word/punctuation tokens**, not arbitrary-length summaries. Input must fit the schema-plus-text token budget (default 512, configurable up to 4096). This first version does **not** chunk long documents; exceeding the budget returns an error. `--max-words` is explicit truncation, not full-document extraction. The processor's synthetic terminal punctuation is encoded for parity but never returned as a standalone span; suffix offsets inside a URL are clipped to the original text.

CPU checks compare all raw candidate logits, count logits/gate, token/marker/word positions and decoded spans against upstream on six cases, including Unicode, URLs, truncation, empty text and the Cloudflare example. Synthetic F32/F16 coverage is part of the existing integration suite. CUDA/Metal builds use the same GGML graph but require their own new span-path hardware checks; prior classification parity does not establish span parity.

## Count-conditioned structured extraction

Use a **fixed record schema** when the model should discover multiple groups rather than the caller supplying topic labels:

```powershell
.\build\Release\gliner-extract.exe --model .\models\decide-spans.gguf `
  --text-file .\request.txt `
  --structure search_requests `
  --field terms `
  --description "terms=Verbatim search terms for one request needing additional context" `
  --threshold 0.3 --top-k 5 --threads 4
```

`search_requests` and `terms` are structural names used across inputs; they are not caller-invented `d1_terms`/`email_terms` groups. The model predicts a count and conditions the same field query for each record slot. Each surviving record has its own list of term spans. This enables the machinery for fine-tuning retrieval-request grouping, **not** a guarantee that the current Decide weights already do it well: on the supplied Cloudflare text this schema currently predicts one slot but returns no fields above 0.3, matching upstream.

A simple repeated-record example does work with the current checkpoint:

```powershell
.\build\Release\gliner-extract.exe --model .\models\decide-spans.gguf `
  --text "Alice works at Contoso. Bob works at Fabrikam." `
  --structure employment `
  --single-field person --description "person=Employee name" `
  --single-field company --description "company=Company where the employee works" `
  --threads 4
```

It predicts two records and pairs `Alice` with `Contoso`, and `Bob` with `Fabrikam`. Output contains `structure`, `predicted_count` and an ordered `records` array:

```json
{
  "structure": "employment",
  "predicted_count": 2,
  "records": [
    {
      "slot_index": 0,
      "fields": {
        "person": {"text": "Alice", "start": 0, "end": 5, "logit": 4.06787, "probability": 0.983174},
        "company": {"text": "Contoso", "start": 15, "end": 22, "logit": 1.70461, "probability": 0.846136}
      }
    },
    {
      "slot_index": 1,
      "fields": {
        "person": {"text": "Bob", "start": 24, "end": 27, "logit": 6.84442, "probability": 0.998936},
        "company": {"text": "Fabrikam", "start": 37, "end": 45, "logit": 3.76469, "probability": 0.977350}
      }
    }
  ],
  "offset_unit": "utf8_bytes",
  "max_span_width": 8
}
```

Values shown are rounded. `--field NAME` is list-valued and returns an array; `--single-field NAME` returns the highest-confidence surviving span or `null`. Descriptions use `--description NAME=TEXT`. All fields are included in each returned record; records whose fields are all empty are omitted, so `records` may be shorter than `predicted_count`. `slot_index` retains the original predicted slot, including gaps after filtering.

The `[P]` state predicts counts **0..19**. `--max-records N` (default 19) is a safety cap, not a requested count: a larger prediction fails explicitly without truncating slots. Count zero returns no records and does not run the slot head. The runtime encodes the text once, retains word/field features on the selected device, then runs a head-only graph with the learned count-position embeddings and recurrent GRU state for the predicted number of slots. It never reloads the model or reruns the encoder per record.

Thresholding, overlap suppression and `--top-k` apply within each field and slot. Single fields always keep at most one span; top-k limits list fields. Fields and slots are decoded independently: spans may repeat across fields/records and no cross-record deduplication, uniqueness or relational constraints are imposed. The count limit, eight-token span width, UTF-8 byte offsets and aggregate text-token budget still apply; no new model format or reconversion is needed beyond a span-capable GGUF.

This first version supports one structure with ordered list/single extractive fields per call. It does not support nested records, enumerated-choice fields, field validators, mixed classification/entity/structure schemas or automatic long-document chunking. The structure name `entities` is reserved for the entity path. `--debug` exposes count logits and raw `[predicted_count, field_count, candidate_count]` record logits, plus field/word positions and candidate ranges, for parity checks and future fine-tuning evaluation.

CPU reference checks cover real repeated-record examples and all slot logits. Synthetic F32/F16 cases exercise predicted counts 0, 1, 2, 3 and 19, recurrence, Unicode offsets, list/single decoding, empty-record filtering and safety-cap errors. CUDA/Metal hardware checks of the structured path remain pending.

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

The exploratory Python performance harness has been removed from the current tree. Its recorded results, methodology and links to the historical harness remain in the [device reports](docs/benchmarks/README.md). Use the native `gliner-bench` for ongoing performance work.

## Validation

Real-checkpoint F32 parity has been exercised against checkpoint revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`, upstream GLiNER2 commit `1a80c9c85272cd6809009b102b770c452e928196`, Transformers 4.48.1/4.48.2 and PyTorch 2.6.0. Eight single-task cases cover ordinary and empty text, Unicode, punctuation, URLs/email, prompts/descriptions containing special markers, empty descriptions, word truncation and logarithmic relative-position buckets. Five additional joint-schema cases cover multiple tasks with different label counts, repeated labels across tasks, prefix-overlapping task names, task order, mixed decoding modes and shared text truncation. The current processor/scorer and span inference semantics were also checked at upstream commit `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`.

Token IDs, marker positions and task score offsets match exactly. All 25 hidden-state outputs (embeddings plus 24 layers), contextual label states, raw logits, per-task probabilities and chosen labels are compared. On CPU, the observed maximum absolute hidden-state difference was below `3e-5` for single tasks and `1e-4` for joint schemas. The user-reported RTX 4060 Laptop CUDA run passed all 13 cases with a maximum difference of `0.00010967255`. On an Apple M4 Pro with pinned GGML v0.24.0 and the Metal precision overlay, all 13 F32 cases passed with maximum difference `8.87e-5`. Both GPU runs use the unchanged live comparison tolerance `atol=rtol=3e-4`. This is numerical parity for the supported classification path and documented configurations, not a claim that every upstream API or decoder is implemented.

The normal CTest suite needs no ML dependencies or downloaded weights. It replays checked-in, deterministic single-task and joint-schema goldens on a complete tiny encoder, checks F32/F16 storage, the C ABI, task grouping/decoding, conversion preservation and error paths. A callback-count assertion verifies one encoder traversal per multi-task call. It also compares file, buffer and non-seekable short-read stream initialization for both single-task and joint inference, including source lifetimes, malformed/truncated input and loader cleanup. Tiny F32 uses `atol=rtol=5e-5`; F16 storage is compared to F32 goldens with `5e-3`. **Real-checkpoint F16 parity has not been established.**

Native C/CMake checks run without Python. The additional synthetic conversion/CLI suite uses only Python's standard library, not PyTorch. One optional upstream tool, `tests/test_parity.py`, remains for regenerating goldens and comparing real checkpoints; its ML dependencies are never part of the runtime or ordinary build. See [Testing and upstream parity](docs/testing.md) for test layers, fixture regeneration and the explicit real-model check.

The `--states` CLI path still accepts precomputed contextual `[L]` states for classification-head-only scoring: one row of `hidden_size` floats per label, in label order. It retains sigmoid probabilities by default.

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

### Span extraction API

Reuse a span-capable context and a state, just as for classification:

```c
const struct gliner_span_label labels[] = {
    {"person", "A person name"},
    {"company", "A company name"}
};
struct gliner_span_params options = gliner_default_span_params();
options.n_threads = 4;
options.max_spans_per_label = 5;
int status = gliner_extract_spans(ctx, state, "Tim Cook works at Apple.",
                                 labels, 2, &options);
if (status == GLINER_STATUS_OK) {
    const struct gliner_span * spans = gliner_get_spans(state);
    for (int i = 0; i < gliner_n_spans(state); ++i) {
        printf("%s: %s [%zu,%zu) %.6f\n", labels[spans[i].label_index].name,
               spans[i].text, spans[i].start, spans[i].end, spans[i].probability);
    }
} else {
    fprintf(stderr, "%s\n", gliner_last_error());
}
```

`gliner_model_supports_spans` reports capability. Input pointers are borrowed for the call. Returned strings, arrays and `gliner_get_span_scores` diagnostic pointers are state-owned and valid until its next inference call or destruction. Offsets count UTF-8 bytes, not Unicode code points or UTF-16 code units. An empty match set is successful, not a fabricated fallback. Any failure clears results; switching between extraction and classification also clears the previous operation's results. Each state remains single-threaded; multiple states share immutable encoder and head weights.

### Repeated record API

Given the same span-capable `ctx` and an initialized `state`:

```c
const struct gliner_record_field fields[] = {
    {"terms", "Verbatim search terms for one request needing additional context", GLINER_FIELD_LIST}
};
struct gliner_record_params options = gliner_default_record_params();
options.n_threads = 4;
options.max_spans_per_field = 5;
int status = gliner_extract_records(ctx, state, text, "search_requests", fields, 1, &options);
if (status == GLINER_STATUS_OK) {
    const struct gliner_record * records = gliner_get_records(state);
    const struct gliner_span * spans = gliner_get_record_spans(state);
    for (int r = 0; r < gliner_n_records(state); ++r) {
        for (int i = records[r].span_offset; i < records[r].span_offset + records[r].n_spans; ++i) {
            printf("slot %d / %s: %s\n", records[r].slot_index,
                   fields[spans[i].label_index].name, spans[i].text);
        }
    }
} else {
    fprintf(stderr, "%s\n", gliner_last_error());
}
```

`gliner_record` indexes the flat `gliner_get_record_spans` array; each span's `label_index` is its field index. Missing list/single fields have no span entries. `gliner_get_record_scores` provides raw slot logits and the predicted count, including count zero or all-empty decoded records. All results are state-owned until the next inference/free, and a failure clears them. Entity, record and classification result getters remain distinct and are cleared when switching operations.

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

Supported: one text with jointly encoded classification tasks, grouped verbatim span labels, or one count-conditioned repeated-record schema per call; descriptions; list/single extractive fields; classification decoding and thresholded, overlap-resolved spans with the published Decide configuration.

Not implemented: nested/choice-field/relation extraction, generated search queries, mixed classification/extraction or multiple structures in the same call, few-shot examples, constrained/beam/exact decision decoding, calibration fitting, automatic long-document chunking, alternate encoders/tokenizer pipelines, multi-document padded batching, quantization, memory-mapped weights, hybrid CPU/GPU offload or multi-GPU execution. Span/record GPU validation and broader hardware coverage remain pending. See [NEXT_STEPS.md](NEXT_STEPS.md).

The model checkpoint is [Apache 2.0 licensed](https://huggingface.co/fastino/GLiNER2.5-Decide); review its license when redistributing converted weights. Checkpoints and generated GGUF files stay out of Git.
