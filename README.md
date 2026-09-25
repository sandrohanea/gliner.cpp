# gliner.cpp

GGML/GGUF CPU inference for [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) text classification. The CLI runs entirely in C++ from one GGUF file: schema construction, Unicode/Unigram tokenization, the 24-layer DeBERTa-v3-large encoder, contextual `[L]` marker gathering, and the shared classification head.

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
| `--activation sigmoid\|softmax` | Text defaults to exclusive softmax. `--multi-label` and `--states` default to independent sigmoid. |
| `--multi-label --threshold T` | Returns a `labels` array, in input order, with probabilities at least `T` (default 0.5). |
| `--temperature T` | Positive probability temperature; raw logits are unchanged. |
| `--max-words N` | Upstream `max_len`-style word truncation before subword encoding; 0 means unlimited. |
| `--max-tokens N` | Hard schema-plus-text limit, default 512, supported range 1..4096. Exceeding it is an error, not silent schema truncation. |
| `--debug` | Adds exact token IDs, label marker positions and flattened contextual label states to JSON. |
| `--dump-hidden FILE` | Writes embeddings and every encoder layer for parity diagnostics. Header: LE u32 token count and hidden size; payload: LE float32 `[layers+1, tokens, hidden]`. Refuses to overwrite an existing file. |

Long sequences can require substantial memory because attention is quadratic. This implementation does not chunk long documents automatically.

## Python parity

Real-checkpoint F32 parity has been exercised against checkpoint revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`, upstream GLiNER2 commit `1a80c9c85272cd6809009b102b770c452e928196`, Transformers 4.48.1 and PyTorch 2.6.0. Eight cases cover ordinary and empty text, Unicode, punctuation, URLs/email, prompts/descriptions containing special markers, empty descriptions, word truncation and logarithmic relative-position buckets.

Token IDs and marker positions match exactly. All 25 hidden-state outputs (embeddings plus 24 layers), contextual label states, raw logits, softmax probabilities and chosen labels are compared. The observed maximum absolute hidden-state difference was below `3e-5`; the live comparison uses `atol=rtol=3e-4`. This is numerical parity for the supported classification path, not a claim that every upstream API or decoder is implemented.

The normal CTest suite needs no ML dependencies or downloaded weights. It replays checked-in, deterministic upstream golden fixtures on a complete tiny encoder, checks F32/F16 storage, the C ABI, conversion preservation and error paths. Tiny F32 uses `atol=rtol=5e-5`; F16 storage is compared to F32 goldens with `5e-3`. **Real-checkpoint F16 parity has not been established.**

For the optional live oracle, use a separate Python 3.12 environment:

```sh
python -m pip install -r tests/requirements-parity.txt
python tests/test_parity.py --checkpoint models/GLiNER2.5-Decide \
  --gguf models/decide.gguf --binary build/gliner-classify
```

To include that comparison in CTest, configure `GLINER_PARITY_CHECKPOINT` and `GLINER_PARITY_GGUF` with absolute paths and select the oracle environment with `Python3_EXECUTABLE`. Fixtures can be regenerated with `tests/test_parity.py --write-golden` (see the provenance inside `tests/fixtures/tiny_parity.json`).

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

## Scope and remaining work

Supported: one classification task per call, ordered labels, task prompts, ordered label descriptions, exclusive argmax/softmax and independent multi-label thresholding, with the published Decide DeBERTa configuration.

Not implemented: NER/span/JSON extraction, multiple simultaneous tasks, few-shot examples, constrained/beam/exact decision decoding, calibration fitting, automatic long-document chunking, alternate encoders/tokenizer pipelines, batching, quantization, memory-mapped weights or GPU/Metal execution. See [NEXT_STEPS.md](NEXT_STEPS.md).

The model checkpoint is [Apache 2.0 licensed](https://huggingface.co/fastino/GLiNER2.5-Decide); review its license when redistributing converted weights. Checkpoints and generated GGUF files stay out of Git.
