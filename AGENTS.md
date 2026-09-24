# Repository guidance

This project targets **fastino/GLiNER2.5-Decide classification**. Its published config currently declares `architecture: "span"` despite the 2.5 model name. The upstream implementation is under `fastino-ai/GLiNER2`; inspect its current processor, span model, and classification scorer before changing model semantics.

- Keep the GGUF converter streaming and standard-library-only. Preserve every tensor unless a later format version explicitly documents pruning. Validate input shape, dtype, offsets, and required classifier tensors.
- Match raw logits before adding decoding or quantization. The classifier is `Linear(H, 2H)`, ReLU, `Linear(2H, 1)` on contextual `[L]` marker states. A label string by itself is not a valid classifier input.
- Keep `include/gliner/gliner.h` valid C: opaque context/state handles, plain C types, and no exception crossing the ABI. The context owns immutable model data; each state owns execution scratch and scores. Keep the C compiled API test working.
- Never present `gliner-classify` as full text inference until C++ tokenizer and DeBERTa outputs match upstream on golden fixtures. Document unsupported features in README and NEXT_STEPS.md.
- Prefer GGML graph ops and GGUF metadata over custom tensor math. Keep CMake working with a local GGML checkout, an installed GGML, and the pinned fetch fallback.
- Run `cmake -S . -B build -DGLINER_FETCH_GGML=OFF`, `cmake --build build -j`, and `ctest --test-dir build --output-on-failure` when an installed GGML is available. The synthetic conversion test is the baseline for format and classifier wiring; add real-checkpoint parity tests when weights are available.
- Keep checkpoints and converted `.gguf` files out of Git. Do not commit third-party GGML sources unless explicitly requested; a local checkout or pinned CMake fetch is supported.
