# Next steps

## Implemented

- **GGUF-only text classification:** C++ schema construction, Unicode word splitting/lowercasing, NFC/Metaspace Unigram tokenization, marker routing, DeBERTa embeddings and every encoder layer, and contextual classification.
- **Model-format validation:** application format version 2 preserves all tensors and carries required encoder/tokenizer behavior. Legacy files remain head-only; unsupported or incomplete new models fail explicitly.
- **C API and CLI:** `gliner_classify_text`, per-state diagnostics, `--text`/`--text-file`, prompts, descriptions, word truncation and explicit token limits. The precomputed `--states` path is retained.
- **Basic decoding:** exclusive softmax/argmax, independent sigmoid thresholds and probability temperature, with unmodified raw logits.
- **Parity:** committed synthetic token/hidden-state/logit fixtures and optional live checks against the real F32 Decide checkpoint. Eight real cases match token IDs and routing exactly, with maximum observed absolute hidden-state error below `3e-5` across all 24 layers. See README for revisions, tolerances and reproduction commands.

## Remaining

1. **Broader parity coverage.** Add a held-out application corpus, longer contexts, more adversarial Unicode/schema cases and real-checkpoint F16 tests. Preserve intermediate-layer checks when changing kernels. Synthetic F16 coverage is not a substitute for real-model F16 validation.
2. **Upstream schema/decision surface.** Support multiple tasks per encoder pass, few-shot examples, explicit description ordering independent of label order, cross-task constraints and upstream exact/beam decoding. Match upstream decisions and tie-breaking before claiming full Python API parity.
3. **Long documents and batching.** Add documented chunking/aggregation policies, variable-length batches and attention masks. Current token limits deliberately fail rather than silently dropping schema markers.
4. **Performance.** Benchmark cold loading and steady-state CPU inference, then reduce attention-index memory, precompute immutable relative projections, introduce memory mapping and cache schema tokenization. Add backend selection and GPU/Metal after preserving F32 parity.
5. **Quantization and packaging.** Establish separate error budgets for F16 and quantized weights; retain original tensors unless a new format version explicitly documents pruning. Add install/export targets and platform CI with installed, local and fetched GGML.

NER/span/JSON extraction and alternate encoders are not part of the current classification runtime, even though their checkpoint tensors are preserved.
