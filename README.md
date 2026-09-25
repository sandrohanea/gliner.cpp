# gliner.cpp

GGML/GGUF inference for [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) text classification. CPU is the verified reference backend; CUDA and Metal builds are experimental, pending hardware parity. The CLI runs entirely in C++ from one GGUF file: schema construction, Unicode/Unigram tokenization, the 24-layer DeBERTa-v3-large encoder, contextual `[L]` marker gathering, and the shared classification head.

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

The model's weights, all state scratch buffers and graph operations use that device. Tokenization remains on the CPU. GPU matrix multiplications request F32 accumulation; `--threads` controls CPU execution only. There is **no silent CPU fallback**, partial offload or multi-GPU splitting. A missing device, allocation failure or unsupported operation is an error. Weights plus inference scratch must fit on the device; the real F32 model alone needs roughly 1.8 GiB, and attention memory grows quadratically with token count.

For an explicit CPU build on any platform:

```sh
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF
cmake --build build-cpu --config Release -j
```

**CUDA (NVIDIA, Windows/Linux):** install a compatible NVIDIA driver, CUDA toolkit and host C++ compiler, then enable GGML's CUDA backend when building GGML from source:

```powershell
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_METAL=OFF
cmake --build build-cuda --config Release -j
.\build-cuda\Release\gliner-classify.exe --model .\models\decide.gguf --task intent --labels refund_request,other --text "Refund my order"
```

For a single-configuration generator, omit `Release` from the executable path. Use `CMAKE_CUDA_ARCHITECTURES` when targeting a specific GPU architecture.

**Metal (Apple GPUs, macOS):** use the Xcode command-line tools and Metal toolchain:

```sh
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build build-metal -j
```

Run that build's `gliner-classify` with the normal model/task/text arguments; it uses Metal without an additional flag. Embedding the Metal library avoids a separate shader-file deployment step.

`GGML_CUDA`/`GGML_METAL` configure **local or fetched GGML source builds** and fix gliner's execution backend. An installed GGML must already include the requested backend; configuration fails if its CMake package reports otherwise. Unconfigured bare GGML libraries are supported only for CPU builds. Neither CUDA nor Metal has been run in this Windows CPU-only environment. Their release gates are tracked separately in [NEXT_STEPS.md](NEXT_STEPS.md).

## Convert

Download the model repository into a local directory containing `config.json`, `encoder_config/config.json`, `tokenizer.json`, and `model.safetensors` (or sharded safetensors and its index):

```sh
python convert/convert_hf_to_gguf.py models/GLiNER2.5-Decide models/decide.gguf
build/gliner-classify --model models/decide.gguf --inspect
```

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

## Python parity

Real-checkpoint F32 parity has been exercised against checkpoint revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`, upstream GLiNER2 commit `1a80c9c85272cd6809009b102b770c452e928196`, Transformers 4.48.1 and PyTorch 2.6.0. Eight single-task cases cover ordinary and empty text, Unicode, punctuation, URLs/email, prompts/descriptions containing special markers, empty descriptions, word truncation and logarithmic relative-position buckets. Five additional joint-schema cases cover multiple tasks with different label counts, repeated labels across tasks, prefix-overlapping task names, task order, mixed decoding modes and shared text truncation. The current processor/scorer and span inference semantics were also checked at upstream commit `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`.

Token IDs, marker positions and task score offsets match exactly. All 25 hidden-state outputs (embeddings plus 24 layers), contextual label states, raw logits, per-task probabilities and chosen labels are compared. The observed maximum absolute hidden-state difference was below `3e-5` for single tasks and `1e-4` for joint schemas; the live comparison uses `atol=rtol=3e-4`. This is numerical parity for the supported classification path, not a claim that every upstream API or decoder is implemented.

The normal CTest suite needs no ML dependencies or downloaded weights. It replays checked-in, deterministic single-task and joint-schema goldens on a complete tiny encoder, checks F32/F16 storage, the C ABI, task grouping/decoding, conversion preservation and error paths. A callback-count assertion verifies one encoder traversal per multi-task call. It also compares file, buffer and non-seekable short-read stream initialization for both single-task and joint inference, including source lifetimes, malformed/truncated input and loader cleanup. Tiny F32 uses `atol=rtol=5e-5`; F16 storage is compared to F32 goldens with `5e-3`. **Real-checkpoint F16 parity has not been established.**

For the optional live oracle, use a separate Python 3.12 environment:

```sh
python -m pip install -r tests/requirements-parity.txt
python tests/test_parity.py --checkpoint models/GLiNER2.5-Decide \
  --gguf models/decide.gguf --binary build/gliner-classify
```

To include that comparison in CTest, configure `GLINER_PARITY_CHECKPOINT` and `GLINER_PARITY_GGUF` with absolute paths and select the oracle environment with `Python3_EXECUTABLE`. The oracle runs all 13 cases by default; `--single-only` and `--batch-only` select a subset. Regenerate synthetic fixtures with `tests/test_parity.py --single-only --write-golden tests/fixtures/tiny_parity.json` or `--batch-only --write-golden tests/fixtures/tiny_batch_parity.json`; provenance is stored inside each fixture.

Parity runs use the binary's compiled backend. Both scripts accept `--expect-backend cpu|cuda|metal` to assert which build is under test; this option does not select or change the backend. CTest supplies that assertion automatically. Keep separate CPU/CUDA/Metal build directories and run CTest in each; a GPU build without its required device fails rather than skipping. GPU tests initially enforce the same tolerances as CPU; hardware-specific tolerances must be measured and documented before changing them. No GPU speedup or real-checkpoint parity is claimed yet.

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

Not implemented: NER/span/JSON extraction, few-shot examples, constrained/beam/exact decision decoding, calibration fitting, automatic long-document chunking, alternate encoders/tokenizer pipelines, multi-document padded batching, quantization, memory-mapped weights, hybrid CPU/GPU offload or multi-GPU execution. CUDA/Metal hardware validation remains pending. See [NEXT_STEPS.md](NEXT_STEPS.md).

The model checkpoint is [Apache 2.0 licensed](https://huggingface.co/fastino/GLiNER2.5-Decide); review its license when redistributing converted weights. Checkpoints and generated GGUF files stay out of Git.
