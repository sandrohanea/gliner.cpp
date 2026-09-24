# gliner.cpp

An early GGML/GGUF runtime for [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide). The checkpoint is a schema driven classifier with a DeBERTa-v3-large encoder. Despite the model name, its published [`config.json`](https://huggingface.co/fastino/GLiNER2.5-Decide/blob/main/config.json) declares the **span** architecture. Classification labels are represented by contextual `[L]` marker states, and the shared classifier scores those states with `Linear(H, 2H) → ReLU → Linear(2H, 1)` ([upstream span model](https://github.com/fastino-ai/GLiNER2/blob/main/gliner2/models/span/model.py), [classification scorer](https://github.com/fastino-ai/GLiNER2/blob/main/gliner2/classification/scoring.py)).

## Current status

This first iteration converts a local safetensors checkpoint to GGUF v3, keeps all F32/F16 tensors and the Unigram vocabulary, validates DeBERTa metadata, and executes the classification head with GGML on CPU. `gliner-classify` accepts **precomputed contextual label states**. It does not yet turn text into those states; full text inference requires the tokenizer and DeBERTa encoder work in [NEXT_STEPS.md](NEXT_STEPS.md). Scores are independent sigmoid probabilities; the CLI selects the largest logit for a single label. It does not yet implement the upstream schema decoder, multi-label thresholding, or calibration.

The public [gliner.h](include/gliner/gliner.h) is a C API following the context/state pattern in [whisper.h](https://github.com/ggml-org/whisper.cpp/blob/master/include/whisper.h). A `gliner_context` owns GGUF metadata and the loaded classifier weights; each `gliner_state` owns a CPU backend, reusable inference allocator, and result scores. Multiple states can share one context. Keep the context alive until its states are freed.

## Build

Requires CMake 3.20+, a C++17 compiler, and GGML. CMake uses `ggml/` if it contains a checkout, then an installed GGML, then downloads pinned GGML v0.24.0. A checkout can be added with:

```sh
git clone --depth 1 --branch v0.24.0 https://github.com/ggml-org/ggml.git ggml
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The checkout is optional. For an installed GGML, pass `-DGLINER_FETCH_GGML=OFF` to avoid a network fetch. This repository does not bundle GGML or its license.

## Convert

Download [the model repository](https://huggingface.co/fastino/GLiNER2.5-Decide) into a local directory containing `config.json`, `encoder_config/config.json`, `tokenizer.json`, and `model.safetensors` (or sharded safetensors plus its index). The converter uses only Python's standard library and streams tensor data, so it does not need PyTorch or NumPy.

```sh
python3 convert/convert_hf_to_gguf.py models/GLiNER2.5-Decide models/decide.gguf
build/gliner-classify --model models/decide.gguf --inspect
```

Conversion currently accepts F32/F16 tensors and a SentencePiece Unigram `tokenizer.json`. It preserves all checkpoint tensors, including encoder and span tensors, even though this runtime only executes the classification head. It does not quantize weights.

## Score label states

`--states` is a UTF-8 text file with one row per label and `hidden_size` whitespace-separated float values per row. Each row must be the encoder output at that label's `[L]` marker after processing the same text and schema. The labels must be in the same order.

```sh
build/gliner-classify --model models/decide.gguf \
  --labels refund_request,order_status,other \
  --states label_states.txt --threads 4
```

For a parity check, `examples/export_label_states.py` can create those rows with the upstream Python implementation. It requires `gliner2` and its PyTorch dependencies in a separate environment:

```sh
python3 examples/export_label_states.py \
  --checkpoint models/GLiNER2.5-Decide \
  --text "Please refund my order" --task intent \
  --labels refund_request order_status other \
  --output label_states.txt
build/gliner-classify --model models/decide.gguf \
  --labels refund_request,order_status,other --states label_states.txt
```

The Python helper prints reference logits so the GGML logits can be compared directly. This helper has not been run against the real checkpoint in this environment because the checkpoint and Python dependencies are unavailable here.

## C API

```c
#include "gliner/gliner.h"
#include <stdio.h>

int score_states(const char * path, const float * states, int n_labels) {
    struct gliner_context * ctx = gliner_init_from_file(path);
    if (!ctx) { fprintf(stderr, "%s\n", gliner_last_error()); return 1; }
    struct gliner_state * state = gliner_init_state(ctx);
    if (!state) {
        fprintf(stderr, "%s\n", gliner_last_error());
        gliner_free(ctx);
        return 1;
    }
    // states contains n_labels * gliner_model_hidden_size(ctx) floats, row major.
    int status = gliner_score_label_states(ctx, state, states, n_labels, 4);
    if (status == GLINER_STATUS_OK) {
        const struct gliner_score * scores = gliner_get_scores(state);
        printf("first label logit: %f\n", scores[0].logit);
    } else {
        fprintf(stderr, "%s\n", gliner_last_error());
    }
    gliner_free_state(state);
    gliner_free(ctx);
    return status;
}
```

The score pointer remains valid until the next score call on that state. A failed call clears its results. A state is used by one thread at a time; separate states can score against the same context. `gliner_last_error()` reports initialization and scoring failures on the calling thread.

## Layout

```text
include/gliner/    Public C API
src/gguf.cpp       GGUF metadata and tensor loading
src/tokenizer.cpp  Classification schema tokens
src/deberta.cpp    Encoder metadata validation (forward graph pending)
src/classify.cpp   GGML classification head
convert/           Streaming safetensors to GGUF converter
examples/          CLI and upstream state export helper
tests/             C API and synthetic conversion/inference checks
```

The model checkpoint is [Apache 2.0 licensed](https://huggingface.co/fastino/GLiNER2.5-Decide); review its license when redistributing converted weights.
