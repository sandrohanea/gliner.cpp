# Model compatibility

The GLiNER2.5 family includes different serialized architectures. A model name alone is not a compatibility guarantee. This matrix records checkpoint inspection and CPU parity work performed on **2026-09-25**.

## Current native coverage

| Hugging Face model | Architecture | Encoder hidden/layers | Classification and joint decisions | Entity spans | Repeated records |
|---|---|---|---|---|---|
| [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | `span` | DeBERTa-v3-large, 1024/24 | Supported | Supported with span metadata | Supported with span metadata |
| [fastino/gliner2.5-base-v1](https://huggingface.co/fastino/gliner2.5-base-v1) | `boundary` | DeBERTa-v3-base, 768/12 | Supported; CPU parity passed | Supported; CPU parity passed | Anchorless/latent; CPU parity passed |
| [fastino/gliner2.5-small-v1](https://huggingface.co/fastino/gliner2.5-small-v1) | `boundary` | DeBERTa-v3-xsmall, 384/12 | Supported; CPU parity passed | Supported; CPU parity passed | Anchorless/latent; CPU parity passed |
| [fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1) | `boundary` | mDeBERTa-v3-base, 768/12 | Supported; CPU parity passed | Supported; CPU parity passed | Anchorless/latent; CPU parity passed |
| [fastino/GLiNER2.5-multi-Decide](https://huggingface.co/fastino/GLiNER2.5-multi-Decide) | `boundary` | mDeBERTa-v3-base, 768/12 | Supported; CPU parity passed | Supported; CPU parity passed | Anchorless/latent; CPU parity passed |

The four boundary checkpoints support the existing classification C APIs, `gliner-classify`, classification workloads in `gliner-bench`, and grouped entity extraction through `gliner_extract_spans`/`gliner-extract`. Explicit logits, sigmoid/softmax, single-label and multi-label decisions, descriptions and joint tasks use the shared encoder/classifier path. Entity extraction uses a separate native boundary graph, **not** the span architecture's `markerV0`/`count_lstm` machinery. Repeated records use the preserved `record_decoder` tensors and candidate endpoint encoder.

This is numerical compatibility, not a classification-accuracy or retrieval-quality benchmark. Output quality and label calibration differ between checkpoints. CUDA/Metal builds share the implementation but have not been hardware-validated with these four boundary checkpoints. The earlier English Decide GPU results do not establish GPU parity for this family.

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

Substitute another model directory/name without changing the build. Fresh conversions report `architecture: boundary`, `text_inference: yes`, `span_extraction: yes` and `record_extraction: yes`. C callers can query `gliner_model_architecture`, `gliner_model_supports_spans` and `gliner_model_supports_records`. Older classification-only or entity-only boundary GGUFs retain their existing capabilities; reconvert to a new filename to enable records.

```powershell
.\build\Release\gliner-extract.exe --model .\models\gliner2.5-small-v1.gguf `
  --text "Tim Cook works at Apple in California." --labels person,company,location `
  --description "person=A person name" --description "company=A company name" `
  --description "location=A location name" --threads 4 --debug
```

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

The second command runs only the four additional classification language cases. Run the first command with the multilingual checkpoint too for the full 17-case classification coverage. Boundary entity extraction has its own mode:

```powershell
.\build-oracle\Scripts\python.exe .\tests\test_parity.py `
  --checkpoint .\models\gliner2.5-small-v1 --gguf .\models\gliner2.5-small-v1.gguf `
  --binary .\build\Release\gliner-extract.exe --boundary-spans-only --expect-backend cpu
```

`--spans-only`/`--records-only` remain span-architecture tests. The boundary reference loads upstream `BoundaryEncoder`, `BoundaryQueryHead`, `DocumentCandidatePool`, `SharedPoolScorer` and the null projection in eval mode. No new Python runtime or separate reference tool is introduced.

## Boundary entity path

One encoder pass constructs contextual entity queries and text-word states. GGML implements learned BOS/EOS boundaries, local self-attention and SwiGLU refinement, start/end/inside marginals, endpoint compatibility, mean-pooled span content, FiLM/GELU pair scoring and null logits. CPU code performs stable boundary selection, per-query quotas, shared-pool deduplication and weighted non-overlap decoding. Features stay on the compiled backend between the encoder/head graphs.

Supported checkpoints use a shared pool, no candidate/query attention layers, mean content pooling, inside evidence, abstention and flat overlap policy. Other extraction variants fail conversion rather than silently approximating them; `--classification-only` can explicitly retain classification access without pruning any weights. All four checkpoints use top-32 start/end boundaries, a 192-candidate pool and an eight-pair reservation per query. Scoring the valid Cartesian pairs before selecting the pool is equivalent here because there is no cross-candidate attention.

Unlike markerV0, this path has no eight-word width ceiling. JSON reports `max_span_width: null`; the aggregate token budget and candidate pool still limit coverage. Null sigmoid above the checkpoint's abstention threshold suppresses that label. Remaining pair scores use the checkpoint temperature, the caller's threshold, weighted flat overlap resolution (or `--allow-overlap`), then top-k. Native offsets are source UTF-8 bytes; synthetic punctuation is encoded but excluded from returned spans.

Debug JSON and `gliner_get_boundary_scores` expose the valid pool only (no padded invalid entries), label-major pair logits, proposal scores, start/end marginals and per-label null logits. They are separate from the span checkpoint's count diagnostics.

| Boundary checkpoint | Eleven-case CPU extraction run: maximum absolute pair-logit difference |
|---|---:|
| `gliner2.5-base-v1` | `2.2914502e-5` |
| `gliner2.5-small-v1` | `2.2936523e-5` |
| `gliner2.5-multi-v1` | `6.6770605e-5` |
| `GLiNER2.5-multi-Decide` | `3.8148535e-5` |

All four passed all eleven cases (44 comparisons), matching exact tokens, word/query routing, ordered candidate pools and decoded spans, and comparing every pair/proposal/start/end/null logit with unchanged `atol=rtol=3e-4`. Cases cover descriptions, Unicode, URLs, empty text, truncation, the Cloudflare request, a 144-word input crossing the local-attention window, and French, Chinese, Arabic and Spanish entity inputs.

## GGUF application format

Boundary records are described [below](#boundary-record-path); they add a capability without invalidating older files.

The GGUF container stays **version 3**. Its application metadata distinguishes capabilities:

| Application format | Metadata | Meaning |
|---|---|---|
| Legacy / version 2 span | `general.architecture=gliner2.5-decide`, `gliner.architecture=span` | Existing files remain compatible; span/record execution additionally requires span metadata |
| Version 3 boundary | `general.architecture=gliner2`, `gliner.architecture=boundary`, `gliner.capabilities=classification` | Shared encoder/classifier execution only; all other weights retained |
| Version 3 boundary with extraction | Same architecture, `gliner.capabilities=classification,spans`, `gliner.boundary.variant=shared-v1` and validated head settings | Shared-pool entity extraction plus classification; records remain unsupported |
| Version 3 boundary with records | `gliner.capabilities=classification,spans,records`, `gliner.boundary.record_variant=instances-v1`, record dimensions/query count/temperature | Classification, entities and anchorless/latent records |

Boundary files record `gliner.classifier_output_index` and `deberta.variant=deberta-v3`. Dropout configurations with no intervening layer use output index 2; positive dropout uses index 3. Unsupported architectures, versions, encoder/tokenizer configurations and classifier shapes fail explicitly.

Twenty tensor names in each boundary checkpoint exceed GGML's 63-byte tensor-name limit. Application format 3 stores deterministic short aliases, `gliner.t.` followed by a SHA-256 prefix, for those names. Parallel string arrays `gliner.tensor_name_map.original` and `gliner.tensor_name_map.stored` retain the exact reversible mapping. Tensor shape, dtype, order and payload are unchanged; collisions are rejected. The encoder/classifier names fit the limit and remain unchanged. This is name encoding, not pruning; native boundary-head loading resolves the mapping when accessing those tensors.

## Remaining boundary-model implementation

1. Add named-anchor/natural mode and explicit required/exclusive field cardinalities, with upstream global-assignment parity. Choices/validators, relations and mixed schemas remain separate work.
2. Validate entity and record graphs on CUDA and strict Metal hardware, including prefix sums, FiLM broadcasts, long inputs and memory pressure.
3. Add per-query pools, cross-candidate/query attention or other variants only with matching reference fixtures. Their rotary/proposal paths differ from the active shared-pool scorer implemented here.
4. Evaluate real-checkpoint F16 accuracy and retrieval-term quality separately from F32 implementation parity.

## Boundary record path

`gliner-extract --structure search_requests --field terms` works with these checkpoints, defaulting to **anchorless** mode. `--record-mode latent` selects learned span-seeded instances. Upstream requires explicit record metadata to activate these modes; the native default corresponds to `record_metadata={structure_name: {"mode": "anchorless"}}`, not upstream's legacy single-instance JSON-structure fallback.

The graph projects concatenated candidate endpoint states to encoder width, runs learned query/candidate attention for anchorless instances (32 per checkpoint) or the latent seed scorer, and evaluates instance-plus-field assignments against candidate states and the learned `ABSENT` vector. Field types are optional scalar or zero-or-more list, nonexclusive. The public `--single-field` type maps to scalar cardinality without changing encoded marker text.

Decode uses the checkpoint's record temperature, object/assignment thresholds, scalar softmax including ABSENT, list sigmoids, pre-filter assignment deduplication, candidate-probability gating, then source-span overlap/top-k filtering. There is no count head. Returned confidence is `min(candidate_probability, assignment_probability)`; returned `logit` is the unscaled assignment logit. Boundary JSON reports `record_count`, not `predicted_count`. Empty records are omitted and max-record safety caps fail rather than truncate.

The native `slot_index` is the learned query index, or the canonical candidate index for latent mode. Upstream repeats the same shared-pool latent seed for every field; tiny kernel-dependent roundoff can choose different equivalent copies. Canonicalization preserves seed identity without requiring an arbitrary duplicate-field index to match.

| Boundary checkpoint | Cases (both modes) | Maximum absolute assignment-logit difference |
|---|---:|---:|
| `gliner2.5-base-v1` | 16 | `2.864646e-5` |
| `gliner2.5-small-v1` | 16 | `3.6258008e-5` |
| `gliner2.5-multi-v1` | 16 | `8.4873772e-5` |
| `GLiNER2.5-multi-Decide` | 16 | `1.2775071e-4` |

All 64 comparisons passed the unchanged real F32 `atol=rtol=3e-4`, including raw pair/object/assignment logits, exact ordered candidate pools, canonical slot identities and decoded record fields. Coverage includes employment, generic search terms, mixed list/scalar fields, Unicode, empty text, truncation, thresholds, overlap/top-k and the full Cloudflare request. The oracle uses upstream `RecordHead.forward_group`, `decode_group` and the overlap decoder. This is not a quality benchmark: the Cloudflare schema returns empty outputs for most model/mode combinations, and many unwanted groups for small-model latent mode.

```powershell
.\build-oracle\Scripts\python.exe .\tests\test_parity.py `
  --checkpoint .\models\gliner2.5-small-v1 --gguf .\models\gliner2.5-small-records.gguf `
  --binary .\build\Release\gliner-extract.exe --boundary-records-only --expect-backend cpu
```
