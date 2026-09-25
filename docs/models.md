# Model compatibility

The GLiNER2.5 family includes different serialized architectures. A model name alone is not a compatibility guarantee. This matrix records checkpoint inspection and CPU parity work performed on **2026-09-25**.

## Current native coverage

| Hugging Face model | Architecture | Encoder hidden/layers | Classification and joint decisions | Entity spans | Repeated records |
|---|---|---|---|---|---|
| [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | `span` | DeBERTa-v3-large, 1024/24 | Supported | Supported with span metadata | Supported with span metadata |
| [fastino/gliner2.5-base-v1](https://huggingface.co/fastino/gliner2.5-base-v1) | `boundary` | DeBERTa-v3-base, 768/12 | Supported; CPU parity passed | Not implemented | Not implemented |
| [fastino/gliner2.5-small-v1](https://huggingface.co/fastino/gliner2.5-small-v1) | `boundary` | DeBERTa-v3-xsmall, 384/12 | Supported; CPU parity passed | Not implemented | Not implemented |
| [fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1) | `boundary` | mDeBERTa-v3-base, 768/12 | Supported; CPU parity passed | Not implemented | Not implemented |
| [fastino/GLiNER2.5-multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide) | `boundary` | mDeBERTa-v3-base, 768/12 | Supported; CPU parity passed | Not implemented | Not implemented |

The four boundary checkpoints support the existing classification C APIs, `gliner-classify`, and classification workloads in `gliner-bench`. Explicit logits, sigmoid/softmax, single-label and multi-label decisions, descriptions and joint tasks use the shared encoder/classifier path. They do **not** use the span architecture's `markerV0`/`count_lstm` extraction path. Native extraction calls reject boundary models rather than returning unrelated or plausible-looking spans.

This is numerical compatibility, not a classification-accuracy benchmark. Output quality and label calibration differ between checkpoints. CUDA/Metal builds share the implementation but have not been hardware-validated with these four boundary checkpoints. The earlier English Decide GPU results do not establish GPU parity for this family.

## Verified checkpoint revisions and results

| Model suffix under `fastino/` | Pinned revision | Cases passed | Maximum absolute hidden-state difference |
|---|---|---:|---:|
| `gliner2.5-base-v1` | `b0c10b23313ec3ff028821dff298dd743e010706` | 13 | `1.0699034e-5` |
| `gliner2.5-small-v1` | `3ec6d3dd7e1e93a7cf9b46096fa47aeda61c711c` | 13 | `1.7166138e-5` |
| `gliner2.5-multi-v1` | `12fc40399dae672ce840c5e3c50a92340bff3c8c` | 17 | `6.4373016e-5` |
| `GLiNER2.5-multi-Decide` | `6bc1d43d201b0691e733626389af8c57eea3ea68` | 17 | `9.5427036e-5` |

All runs used the existing F32 `atol=rtol=3e-4`, with no tolerance changes. Checks cover exact input IDs and marker/task routing, embeddings plus all 12 encoder layers, contextual label states, raw logits, task-local probabilities and decisions. The 13 standard cases include empty/ordinary text, Unicode, descriptions, truncation, joint tasks and reordered tasks. Multilingual models additionally ran French, Chinese and Arabic single tasks and joint Spanish tasks.

The reference was PyTorch 2.6.0 and Transformers 4.48.1, with the existing upstream processor and architecture-appropriate classifier from the parity tool. Although the published encoder configs declare `dtype: float16`, the inspected safetensors checkpoints store **334 F32 tensors each**. Every tensor payload in all four generated GGUFs was compared byte-for-byte with its source; none was pruned or quantized.

The encoder dimensions and relative-attention configuration fit the existing DeBERTa graph. English tokenizers have 128011 total entries; multilingual tokenizers have 250112. All four use the supported Unigram NFC/Metaspace pipeline. Boundary classifier dropout is disabled in eval mode but shifts the stored output layer to `classifier.3` for these checkpoints; the converter validates and records that index instead of assuming `classifier.2`.

## Convert and run

Download a pinned model revision into a local directory containing `config.json`, `encoder_config/config.json`, `tokenizer.json` and its safetensors weights, using the URLs/revisions above and your preferred approved client. Then use the normal standard-library converter:

```powershell
python .\convert\convert_hf_to_gguf.py .\models\gliner2.5-small-v1 .\models\gliner2.5-small-v1.gguf

.\build\Release\gliner-classify.exe --model .\models\gliner2.5-small-v1.gguf --inspect
.\build\Release\gliner-classify.exe --model .\models\gliner2.5-small-v1.gguf `
  --text "Please refund my order" `
  --task intent --labels refund_request,order_status,other `
  --task sentiment --labels positive,neutral,negative --threads 4
```

Substitute another model directory/name without changing the build. `--inspect` reports `architecture: boundary`, `text_inference: yes` and `span_extraction: no`. `gliner_model_architecture(ctx)` provides the same architecture through the C API; `gliner_model_supports_spans(ctx)` is false.

Use the existing optional reference environment to reproduce the classification check:

```powershell
.\build-oracle\Scripts\python.exe .\tests\test_parity.py `
  --checkpoint .\models\gliner2.5-small-v1 `
  --gguf .\models\gliner2.5-small-v1.gguf `
  --binary .\build\Release\gliner-classify.exe --expect-backend cpu

.\build-oracle\Scripts\python.exe .\tests\test_parity.py `
  --checkpoint .\models\GLiNER2.5-multi-Decide `
  --gguf .\models\GLiNER2.5-multi-Decide.gguf `
  --binary .\build\Release\gliner-classify.exe --expect-backend cpu --multilingual-only
```

The second command runs only the four additional language cases. Run the first command with the multilingual checkpoint too for the full 17-case coverage. Do not run `--spans-only`/`--records-only` against these native boundary models; those modes test the implemented span architecture.

## GGUF application format

The GGUF container stays **version 3**. Its application metadata distinguishes capabilities:

| Application format | Metadata | Meaning |
|---|---|---|
| Legacy / version 2 span | `general.architecture=gliner2.5-decide`, `gliner.architecture=span` | Existing files remain compatible; span/record execution additionally requires span metadata |
| Version 3 boundary | `general.architecture=gliner2`, `gliner.architecture=boundary`, `gliner.capabilities=classification` | Shared encoder/classifier execution only; all other weights retained |

Boundary files record `gliner.classifier_output_index` and `deberta.variant=deberta-v3`. Dropout configurations with no intervening layer use output index 2; positive dropout uses index 3. Unsupported architectures, versions, encoder/tokenizer configurations and classifier shapes fail explicitly.

Twenty tensor names in each boundary checkpoint exceed GGML's 63-byte tensor-name limit. Application format 3 stores deterministic short aliases, `gliner.t.` followed by a SHA-256 prefix, for those names. Parallel string arrays `gliner.tensor_name_map.original` and `gliner.tensor_name_map.stored` retain the exact reversible mapping. Tensor shape, dtype, order and payload are unchanged; collisions are rejected. The encoder/classifier names fit the limit and remain unchanged. This is name encoding, not pruning; future boundary-head loading must resolve the mapping when accessing those tensors.

## Remaining boundary-model implementation

1. Port boundary-state refinement/local attention and query-conditioned start/end/inside scoring.
2. Implement the checkpoint's sparse proposal/shared candidate pool, rotary endpoint compatibility, span-content features and pair reranking, including abstention and overlap decoding.
3. Port its separate record-instance/anchor/field head. The boundary record machinery is **not** the span checkpoint's count-conditioned GRU.
4. Compare intermediate outputs and final spans/records on synthetic and real checkpoint fixtures, then validate CUDA and Metal.

Until those steps are complete, the requested family-wide span/record comparison is blocked by unimplemented native heads, not by checkpoint conversion.
