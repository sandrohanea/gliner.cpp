"""Optional live oracle: compare tokenizer, every DeBERTa layer, states and logits.

The regular CTest suite replays committed synthetic golden fixtures without torch.
This script can regenerate those fixtures or check a real local checkpoint.
"""

import argparse
from dataclasses import replace
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import transformers
from gliner2.processor import SchemaTransformer
from safetensors import safe_open
from transformers import DebertaV2Config, DebertaV2Model, PreTrainedTokenizerFast

from tiny_checkpoint import create_checkpoint
from parity_environment import configure_parity_environment


CASES = [
    {"text": "Please refund my order", "task": "intent", "labels": ["refund_request", "order_status", "other"]},
    {"text": "", "task": "intent", "labels": ["yes", "no"]},
    {"text": "HELLO,world! Caf\u00e9 cafe\u0301 \u0130 \u039f\u03a3 \u4e2d\u6587 \U0001f680\U0001f9ec",
     "task": "language", "labels": ["English", "Other"]},
    {"text": "Email A.B+tag@Example.COM; visit HTTPS://example.com/a-b?x=1 and @USER_NAME. foo-bar foo_bar 3.14",
     "task": "contact", "labels": ["contact", "other"]},
    {"text": "hello\u00a0world\ttext\ntrailing ", "task": "topic", "labels": ["a", "b"],
     "prompt": "Choose [L] wisely.  Caf\u0065\u0301", "descriptions": ["hello [L] world", "other [P] things"]},
    {"text": "one two three four five six seven", "task": "intent", "labels": ["a", "b"], "max_words": 3},
    {"text": "hello " * 140, "task": "intent", "labels": ["a", "b"]},
    {"text": "hello world", "task": "intent", "labels": ["a", "b"], "descriptions": ["", ""]},
]

BATCH_CASES = [
    {"text": "Please refund my order; the delivery was late and I am upset.", "tasks": [
        {"task": "intent", "labels": ["refund_request", "order_status", "other"]},
        {"task": "sentiment", "labels": ["positive", "neutral", "negative"]},
        {"task": "urgent", "labels": ["yes", "no"]},
    ]},
    {"text": "Caf\u00e9 \u0130 \u039f\u03a3 \u4e2d\u6587 \U0001f680. Email A.B@example.com", "tasks": [
        {"task": "intent", "labels": ["yes", "no"], "prompt": "Choose [L] with [SEP_STRUCT] in mind",
         "descriptions": ["", "not [P] applicable"]},
        {"task": "intent detail", "labels": ["yes", "no", "other"], "prompt": "Consider [SEP_TEXT] literally",
         "multi_label": True, "threshold": 0.4, "temperature": 2},
        {"task": "language", "labels": ["other"]},
    ]},
    {"text": "", "tasks": [
        {"task": "topic", "labels": ["other"]},
        {"task": "answer", "labels": ["yes", "no"], "descriptions": ["", ""]},
    ]},
    {"text": "one two three four five six seven", "max_words": 3, "tasks": [
        {"task": "first", "labels": ["yes", "no"], "multi_label": True, "threshold": 0.2},
        {"task": "second", "labels": ["yes", "no"], "activation": "sigmoid", "temperature": 0.7},
    ]},
]
BATCH_CASES.append({"text": BATCH_CASES[0]["text"], "tasks": list(reversed(BATCH_CASES[0]["tasks"]))})

KERNEL_CASES = [
    {"text": "", "task": "a", "labels": ["a"]},
    {"text": "a " * 20, "task": "a", "labels": ["a"]},
]

MULTILINGUAL_CASES = [
    {"text": "Je voudrais un remboursement de ma commande, s'il vous pla\u00eet.",
     "task": "intention", "labels": ["remboursement", "suivi de commande", "autre"]},
    {"text": "\u8bf7\u9000\u8fd8\u6211\u7684\u8ba2\u5355\u8d39\u7528\uff0c\u6211\u5bf9\u914d\u9001\u4e0d\u6ee1\u610f\u3002",
     "task": "\u610f\u56fe", "labels": ["\u9000\u6b3e", "\u8ba2\u5355\u72b6\u6001", "\u5176\u4ed6"]},
    {"text": "\u0623\u0631\u064a\u062f \u0627\u0633\u062a\u0631\u062f\u0627\u062f \u0645\u0628\u0644\u063a \u0637\u0644\u0628\u064a.",
     "task": "\u0627\u0644\u0646\u064a\u0629", "labels": ["\u0627\u0633\u062a\u0631\u062f\u0627\u062f", "\u062d\u0627\u0644\u0629 \u0627\u0644\u0637\u0644\u0628", "\u0623\u062e\u0631\u0649"]},
    {"text": "Quiero un reembolso. La entrega lleg\u00f3 tarde y estoy decepcionado.", "tasks": [
        {"task": "intenci\u00f3n", "labels": ["reembolso", "estado del pedido", "otro"]},
        {"task": "sentimiento", "labels": ["positivo", "neutral", "negativo"]},
    ]},
]

SPAN_CASES = [
    {"text": "Tim Cook works at Apple in California.", "labels": ["person", "company", "location"],
     "descriptions": ["A person name", "A company name", "A location name"], "threshold": 0.5},
    {"text": "Caf\u00e9 cafe\u0301 \u0130 \u039f\u03a3 \u4e2d\u6587 \U0001f680", "labels": ["thing", "other"],
     "descriptions": ["A thing [E] mentioned in the text", ""], "threshold": 0., "top_k": 3},
    {"text": "", "labels": ["thing"], "threshold": 0.},
    {"text": "Email A.B@example.com or visit https://example.com", "labels": ["email", "URL"], "threshold": 0., "top_k": 2},
    {"text": "hello world ignored words", "labels": ["term"], "max_words": 1, "threshold": 0., "allow_overlap": True},
    {"text": "As of September 25, 2026, I am building a small Cloudflare Worker and need to verify two current platform limits. Consult the current official Cloudflare documentation rather than relying only on memory. (1) What is the maximum number of bound parameters in one Cloudflare D1 query, and would a single query with 120 bound parameters fit that limit? (2) What is the maximum combined number of recipients across to, cc, and bcc in one Cloudflare Email Sending message through the Workers binding, and would one message addressed to 60 total recipients fit that limit? Give both numeric limits and a yes/no for each proposed operation. Cite the official sources you used.",
     "labels": ["D1 search term", "Email Sending search term"], "threshold": 0.3, "top_k": 5,
     "descriptions": ["Words or short phrases needed to search official documentation for the Cloudflare D1 query bound parameter limit. Include the product name and the property whose limit is requested.",
                      "Words or short phrases needed to search official documentation for the Cloudflare Email Sending combined recipient limit. Include the product name and the property whose limit is requested."]},
]

RECORD_CASES = [
    {"text": "Alice works at Contoso. Bob works at Fabrikam.", "structure": "employment",
     "fields": [{"name": "person", "type": "single", "description": "Employee name"},
                {"name": "company", "type": "single", "description": "Company where the employee works"}],
     "threshold": 0., "synthetic_count": 2},
    {"text": SPAN_CASES[-1]["text"], "structure": "search_requests",
     "fields": [{"name": "terms", "type": "list", "description": "Verbatim search terms for one request needing additional context"}],
     "threshold": 0.3, "top_k": 5, "synthetic_count": 3},
    {"text": "Caf\u00e9 \u0130 \u4e2d\u6587. \U0001f680 Other words.", "structure": "items",
     "fields": [{"name": "name", "type": "single", "description": ""},
                {"name": "terms", "type": "list", "description": "Relevant [C] phrases"}],
     "threshold": 0., "top_k": 2, "synthetic_count": 2},
    {"text": "", "structure": "requests", "fields": [{"name": "terms", "type": "list"}],
     "threshold": 0., "synthetic_count": 0},
    {"text": "hello world", "structure": "items", "fields": [{"name": "name", "type": "single"}],
     "threshold": 0., "synthetic_count": 19},
    {"text": "Alpha Beta Gamma ignored", "structure": "items", "fields": [{"name": "terms", "type": "list"}],
     "max_words": 2, "allow_overlap": True, "threshold": 0., "top_k": 2, "synthetic_count": 1},
    {"text": "nothing selected", "structure": "items", "fields": [{"name": "terms", "type": "list"}],
     "threshold": 1., "synthetic_count": 3},
]

BOUNDARY_CASES = SPAN_CASES + [
    {"text": " ".join(["alpha", "beta", "gamma", "delta"] * 36), "labels": ["phrase", "term"],
     "threshold": 0., "top_k": 4, "allow_overlap": True},
    {"text": "Élodie travaille chez Airbus à Paris.", "labels": ["person", "company", "location"]},
    {"text": "张伟在北京的微软工作。", "labels": ["person", "company", "location"]},
    {"text": "تعمل ليلى لدى مايكروسوفت في دبي.", "labels": ["person", "company", "location"]},
    {"text": "María trabaja para Telefónica en Madrid.", "labels": ["person", "company", "location"]},
]

BOUNDARY_RECORD_CASES = [
    {"text": "Alice works at Contoso. Bob works at Fabrikam.", "structure": "employment",
     "fields": [{"name": "person", "type": "single", "description": "Employee name"},
                {"name": "company", "type": "single", "description": "Employer name"}]},
    {"text": "Check Cloudflare D1 parameter limits and Email Sending recipient limits.", "structure": "search_requests",
     "fields": [{"name": "terms", "type": "list", "description": "Verbatim search terms for one request needing additional context"}],
     "threshold": 0.3, "top_k": 5},
    {"text": "", "structure": "requests", "fields": [{"name": "terms", "type": "list"}]},
    {"text": "Élodie travaille chez Airbus à Paris.", "structure": "employment",
     "fields": [{"name": "person", "type": "single"}, {"name": "location", "type": "list"}], "threshold": 0.1},
    {"text": "Alice works at Contoso. Bob works at Fabrikam.", "structure": "requests",
     "fields": [{"name": "terms", "type": "list"}], "max_words": 3, "threshold": 0., "top_k": 2},
    {"text": "Cloudflare D1 parameter limit", "structure": "requests",
     "fields": [{"name": "terms", "type": "list"}], "threshold": 0., "top_k": 3, "allow_overlap": True},
    {"text": "alpha alpha beta alpha beta.", "structure": "requests",
     "fields": [{"name": "terms", "type": "list"}], "threshold": 1.},
]


def load_oracle(checkpoint):
    config = DebertaV2Config.from_pretrained(checkpoint / "encoder_config")
    model_config = json.loads((checkpoint / "config.json").read_text(encoding="utf8"))
    architecture = model_config.get("architecture")
    if architecture not in ("span", "boundary"):
        raise ValueError("Unsupported reference architecture")
    dropout = model_config["boundary_head"].get("dropout", 0.1) if architecture == "boundary" else 0
    with torch.device("meta"):
        encoder = DebertaV2Model(config)
        layers = [torch.nn.Linear(config.hidden_size, 2 * config.hidden_size), torch.nn.ReLU()]
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(2 * config.hidden_size, 1))
        classifier = torch.nn.Sequential(*layers)
    weights, head = {}, {}
    index = checkpoint / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    for shard in shards:
        with safe_open(checkpoint / shard, framework="pt") as source:
            for key in source.keys():
                if key.startswith("encoder."):
                    weights[key[len("encoder."):]] = source.get_tensor(key)
                elif key.startswith("classifier."):
                    head[key[len("classifier."):]] = source.get_tensor(key)
    encoder.load_state_dict(weights, assign=True, strict=True)
    classifier.load_state_dict(head, assign=True, strict=True)
    # This non-persistent buffer is absent from the checkpoint and stays on meta otherwise.
    encoder.embeddings.register_buffer(
        "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
    # F16 storage is evaluated with F32 activations in GGML.
    encoder.float().eval()
    classifier.float().eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(checkpoint / "tokenizer.json"),
                                        unk_token="[UNK]", pad_token="[PAD]",
                                        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    return SchemaTransformer(tokenizer=tokenizer), encoder, classifier


def load_span_oracle(checkpoint):
    from gliner2.layers import CountLSTM, SpanRepLayer
    processor, encoder, _ = load_oracle(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf8"))
    h, width = encoder.config.hidden_size, config["max_width"]
    if config["counting_layer"] != "count_lstm" or config["span_head"]["span_mode"] != "markerV0":
        raise ValueError("Unsupported span oracle configuration")
    with torch.device("meta"):
        span = SpanRepLayer(h, width, "markerV0", dropout=0)
        count_embed = CountLSTM(h)
        count_pred = torch.nn.Sequential(torch.nn.Linear(h, 2 * h), torch.nn.ReLU(), torch.nn.Linear(2 * h, 20))
    modules = {"span_rep": span, "count_embed": count_embed, "count_pred": count_pred}
    weights = {key: {} for key in modules}
    index = checkpoint / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    for shard in shards:
        with safe_open(checkpoint / shard, framework="pt") as source:
            for key in source.keys():
                prefix, _, name = key.partition(".")
                if prefix in weights:
                    weights[prefix][name] = source.get_tensor(key)
    for key, module in modules.items():
        module.load_state_dict(weights[key], assign=True, strict=True)
        module.float().eval()
    return processor, encoder, span, count_embed, count_pred, width


def load_boundary_oracle(checkpoint, records=False):
    from gliner2.models.boundary.encoding import BoundaryEncoder
    from gliner2.models.boundary.heads import BoundaryQueryHead
    from gliner2.models.boundary.pool import DocumentCandidatePool, SharedPoolScorer
    processor, encoder, _ = load_oracle(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf8"))["boundary_head"]
    if config["candidate_pool"] != "shared":
        raise ValueError("Reference expects the shared boundary pool")
    h, d = encoder.config.hidden_size, config["boundary_dim"]
    with torch.device("meta"):
        modules = {
            "boundary_encoder": BoundaryEncoder(h, d, config["dropout"], config["boundary_refinement_layers"],
                config["boundary_ffn_multiplier"], config["boundary_attention_layers"], config["boundary_attention_heads"],
                config["boundary_attention_window"]),
            "boundary_query_head": BoundaryQueryHead(h, d, h, config["dropout"]),
            "shared_pool_builder": DocumentCandidatePool(d, pool_boundary_top_k=config["pool_boundary_top_k"],
                pool_size=config["pool_size"], min_pool_per_query=config["min_pool_per_query"]),
            "shared_pool_scorer": SharedPoolScorer(d, h, config["pair_dim"], dropout=config["dropout"],
                candidate_attention_layers=config["candidate_attention_layers"],
                candidate_attention_heads=config["candidate_attention_heads"], query_attention_layers=config["query_attention_layers"],
                enable_span_content=config["enable_span_content"], content_dim=config["content_dim"],
                content_soft_max_pool=config["content_soft_max_pool"], text_hidden_size=h),
            "null_projection": torch.nn.Linear(h, 1),
        }
        if records:
            from gliner2.models.boundary.records import RecordHead
            modules["candidate_encoder"] = torch.nn.Linear(2 * d, h)
            modules["record_decoder"] = RecordHead(h, config["record_dim"], config["record_instance_queries"])
    weights = {key: {} for key in modules}
    index = checkpoint / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    for shard in shards:
        with safe_open(checkpoint / shard, framework="pt") as source:
            for key in source.keys():
                if records and key.startswith("record_decoder."):
                    weights["record_decoder"][key[len("record_decoder."):]] = source.get_tensor(key)
                if not key.startswith("boundary_head."):
                    continue
                module, _, name = key[len("boundary_head."):].partition(".")
                if module in weights:
                    weights[module][name] = source.get_tensor(key)
    for key, module in modules.items():
        module.load_state_dict(weights[key], strict=True, assign=True)
        module.float().eval()
    return processor, encoder, modules, config


def boundary_reference(oracle, case, dense=False):
    from gliner2.inference.overlap import resolve_overlaps
    processor, encoder, modules, config = oracle
    records = "fields" in case
    labels = [f["name"] for f in case["fields"]] if records else case["labels"]
    if records:
        name, fields = case["structure"], case["fields"]
        schema = {"json_structures": [{name: {field["name"]: [] for field in fields}}],
                  "json_descriptions": {name: {field["name"]: field["description"] for field in fields if "description" in field}},
                  "record_metadata": {name: {"mode": case["record_mode"]}}}
    else:
        schema = {"entities": {label: [] for label in labels},
                  "entity_descriptions": dict(zip(labels, case.get("descriptions", [])))}
    batch = processor.collate_fn_inference([(case["text"], schema)], max_len=case.get("max_words"), error_policy="raise")
    with torch.inference_mode():
        encoded = encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
        words, groups = processor.extract_embeddings_from_batch(encoded, batch.input_ids, batch)
        text_states = words[0].unsqueeze(0)
        queries = torch.stack(groups[0][0][1:]).unsqueeze(0)
        text_mask = torch.ones(text_states.shape[:2], dtype=torch.bool)
        query_mask = torch.ones(queries.shape[:2], dtype=torch.bool)
        encoding = modules["boundary_encoder"](text_states, text_mask)
        marginals = modules["boundary_query_head"](encoding.states, encoding.mask, text_states, text_mask, queries, query_mask)
        pool = modules["shared_pool_builder"](encoding.states, encoding.mask, query_mask, marginals.start_logits, marginals.end_logits)
        all_logits, _ = modules["shared_pool_scorer"](encoding.states, queries, query_mask, pool, marginals.start_logits,
            marginals.end_logits, marginals.inside_prefix, text_mask.sum(-1), text_states, text_mask, marginals.inside_prefix_mean)
        indices = pool.indices[0, pool.mask[0]]
        logits = all_logits[0, pool.mask[0]].T
        null_logits = modules["null_projection"](queries)[0, :, 0]
        probabilities = (logits / config["pair_temperature"]).sigmoid()
        if records:
            states = modules["candidate_encoder"](torch.cat(
                (encoding.states[:, pool.indices[0, :, 0]], encoding.states[:, pool.indices[0, :, 1]]), -1))
            candidates = pool.to_candidate_batch(all_logits, query_mask, states)
            return boundary_record_reference(oracle, case, batch, words, queries, candidates, indices, logits)
        dense_scores = {}
        if dense:
            # F16 execution can change the discrete pool; retain references for every valid pair.
            pairs = torch.triu_indices(encoding.states.shape[1], encoding.states.shape[1], offset=1).T
            builder = modules["shared_pool_builder"]
            starts = builder.start_projection(encoding.states)[:, pairs[:, 0]]
            ends = builder.end_projection(encoding.states)[:, pairs[:, 1]]
            compat = (starts * ends).sum(-1) / math.sqrt(starts.shape[-1])
            proposals = compat + marginals.start_logits.max(1).values[:, pairs[:, 0]] + marginals.end_logits.max(1).values[:, pairs[:, 1]]
            universe = replace(pool, indices=pairs.unsqueeze(0), mask=torch.ones((1, len(pairs)), dtype=torch.bool),
                               proposal_logits=proposals, compat_logits=compat, gold_mask=None, stats=None)
            scores, _ = modules["shared_pool_scorer"](encoding.states, queries, query_mask, universe,
                marginals.start_logits, marginals.end_logits, marginals.inside_prefix, text_mask.sum(-1),
                text_states, text_mask, marginals.inside_prefix_mean)
            dense_scores = {"all_start_words": pairs[:, 0].tolist(), "all_end_words": pairs[:, 1].tolist(),
                            "all_span_logits": scores[0].T.flatten().tolist(), "all_proposal_logits": proposals[0].tolist()}
    results = []
    text = case["text"]
    for label, name in enumerate(labels):
        candidates = []
        if float(null_logits[label].sigmoid()) <= config["abstention_threshold"]:
            for i, (s, e) in enumerate(indices.tolist()):
                first, last_start = batch.start_mappings[0][s], batch.start_mappings[0][e - 1]
                end = min(len(text), batch.end_mappings[0][e - 1])
                if first >= len(text) or last_start >= len(text) or probabilities[label, i] < case.get("threshold", 0.5):
                    continue
                candidates.append({"text": text[first:end], "start": len(text[:first].encode("utf8")),
                    "end": len(text[:end].encode("utf8")), "probability": float(probabilities[label, i]), "logit": float(logits[label, i])})
        selected = resolve_overlaps(candidates, "allow" if case.get("allow_overlap") else "flat",
            score=lambda c: c["probability"], start=lambda c: c["start"], end=lambda c: c["end"])
        if case.get("top_k", 0):
            selected = selected[:case["top_k"]]
        results.append({"label": name, "spans": selected})
    return {"case": case, "groups": results, "input_ids": batch.input_ids[0].tolist(),
        "label_positions": batch.schema_special_indices[0][0][1:],
        "word_positions": batch.text_word_indices[0, :len(words[0])].tolist(),
        "start_words": indices[:, 0].tolist(), "end_words": indices[:, 1].tolist(), "max_span_width": None,
        "span_logits": logits.flatten().tolist(), "null_logits": null_logits.tolist(),
        "start_logits": marginals.start_logits.flatten().tolist(), "end_logits": marginals.end_logits.flatten().tolist(),
        "proposal_logits": pool.proposal_logits[0, pool.mask[0]].tolist(), "pool_capacity": config["pool_size"],
        "pair_temperature": config["pair_temperature"], "abstention_threshold": config["abstention_threshold"], **dense_scores}

def boundary_record_reference(oracle, case, batch, words, queries, candidates, indices, pair_logits):
    from gliner2.models.boundary.records import decode_group
    from gliner2.processing.records import RecordSpec, RecordFieldSpec, FieldCardinality
    from gliner2.inference.candidate_decoder import finalize_spans
    _, _, modules, config = oracle
    fields = case["fields"]
    spec = RecordSpec(0, case["structure"], "json_structures", case["record_mode"], tuple(
        RecordFieldSpec(i, f["name"], i, FieldCardinality.OPTIONAL_ONE if f["type"] == "single" else FieldCardinality.ZERO_OR_MORE)
        for i, f in enumerate(fields)))
    group = modules["record_decoder"].forward_group(spec, queries[0], candidates, 0)
    threshold = case.get("threshold", 0.5)
    settings = dict(anchor_threshold=threshold, object_threshold=threshold, field_threshold=threshold,
                    temperature=config["record_temperature"])
    decoded = decode_group(group, **settings)
    # Recover the original instance slot without replacing upstream's selection/dedup logic.
    def signature(rec):
        return rec.score, tuple((k, tuple(v)) for k, v in sorted(rec.fields.items()))
    slots = {}
    object_probabilities = (group.object_logits / config["record_temperature"]).sigmoid()
    for slot in range(group.num_instances):
        single = replace(group, object_logits=group.object_logits[slot:slot + 1],
            assign_logits=[v[slot:slot + 1] for v in group.assign_logits],
            instance_seed=group.instance_seed[slot:slot + 1], instance_spans=group.instance_spans[slot:slot + 1])
        for rec in decode_group(single, **settings):
            rec.score = float(object_probabilities[slot])
            slots.setdefault(signature(rec), slot)
    pair_lookup = {tuple(span): i for i, span in enumerate(indices.tolist())}
    result = []
    text = case["text"]
    pair_probs = (pair_logits / config["pair_temperature"]).sigmoid()
    for rec in decoded:
        slot = slots[signature(rec)]
        values = {}
        for f, field in enumerate(fields):
            raw, by_offset = [], {}
            for (s, e), assignment in zip(rec.fields.get(f, []), rec.field_scores.get(f, [])):
                first, last = batch.start_mappings[0][s], batch.start_mappings[0][e - 1]
                end = min(len(text), batch.end_mappings[0][e - 1])
                c = pair_lookup[s, e]
                if first >= len(text) or last >= len(text) or pair_probs[f, c] < threshold:
                    continue
                start_byte, end_byte = len(text[:first].encode("utf8")), len(text[:end].encode("utf8"))
                probability = min(float(pair_probs[f, c]), assignment)
                raw.append((text[first:end], probability, start_byte, end_byte))
                by_offset[start_byte, end_byte] = float(group.assign_logits[f][slot, c + 1])
            selected = finalize_spans(raw, dtype="str" if field["type"] == "single" else "list",
                                      overlap_policy="allow" if case.get("allow_overlap") else "flat")
            if field["type"] != "single" and case.get("top_k", 0):
                selected = selected[:case["top_k"]]
            formatted = [{"text": t, "probability": p, "start": s, "end": e, "logit": by_offset[s, e]} for t, p, s, e in selected]
            values[field["name"]] = formatted if field["type"] != "single" else formatted[0] if formatted else None
        if any(value is not None and value != [] for value in values.values()):
            result.append({"slot_index": slot % len(indices) if case["record_mode"] == "latent" else slot, "fields": values})
    return {"case": case, "structure": case["structure"], "record_mode": case["record_mode"],
        "record_count": len(result), "records": result, "max_span_width": None,
        "n_instances": group.num_instances, "record_temperature": config["record_temperature"],
        "input_ids": batch.input_ids[0].tolist(), "field_positions": batch.schema_special_indices[0][0][1:],
        "word_positions": batch.text_word_indices[0, :len(words[0])].tolist(),
        "start_words": indices[:, 0].tolist(), "end_words": indices[:, 1].tolist(), "pair_logits": pair_logits.flatten().tolist(),
        "object_logits": group.object_logits.tolist(),
        "assignment_logits": torch.stack(group.assign_logits).flatten().tolist()}

def span_forward(oracle, case, schema):
    processor, encoder, span, count_embed, count_pred, width = oracle
    batch = processor.collate_fn_inference([(case["text"], schema)], max_len=case.get("max_words"), error_policy="raise")
    with torch.inference_mode():
        encoded = encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
        words, groups = processor.extract_embeddings_from_batch(encoded, batch.input_ids, batch)
        n_words = words[0].shape[0]
        starts, ends, dense = [], [], []
        for start in range(n_words):
            for w in range(width):
                if start + w < n_words:
                    starts.append(start)
                    ends.append(start + w + 1)
                    dense.append([start, start + w])
                else:
                    dense.append([0, 0])
        all_spans = span(words[0].unsqueeze(0), torch.tensor([dense]))[0]
        valid_spans = torch.stack([all_spans[s, e - s - 1] for s, e in zip(starts, ends)])
        queries = torch.stack(groups[0][0])
        count_logits = count_pred(queries[:1])[0]
    return batch, starts, ends, valid_spans, queries, count_logits


def span_reference(oracle, case):
    from gliner2.inference.candidate_decoder import finalize_spans
    labels = case["labels"]
    descs = {label: desc for label, desc in zip(labels, case.get("descriptions", []))}
    schema = {"entities": {label: [] for label in labels}, "entity_descriptions": descs}
    batch, starts, ends, valid_spans, queries, count_logits = span_forward(oracle, case, schema)
    n_words, width = len(batch.text_tokens[0]), oracle[-1]
    with torch.inference_mode():
        predicted = int(count_logits.argmax())
        projected = oracle[3](queries[1:], 1)[0]
        logits = torch.einsum("sh,qh->qs", valid_spans, projected)
        probs = logits.sigmoid()
    results = []
    text = case["text"]
    for label_index, label in enumerate(labels):
        candidates, by_offset = [], {}
        if predicted > 0:
            for i, (s, e) in enumerate(zip(starts, ends)):
                first = batch.start_mappings[0][s]
                last_start = batch.start_mappings[0][e - 1]
                end = min(len(text), batch.end_mappings[0][e - 1])
                if first >= len(text) or last_start >= len(text) or probs[label_index, i] < case.get("threshold", 0.5):
                    continue
                start_byte, end_byte = len(text[:first].encode("utf8")), len(text[:end].encode("utf8"))
                candidate = (text[first:end], float(probs[label_index, i]), start_byte, end_byte)
                candidates.append(candidate)
                by_offset[start_byte, end_byte] = float(logits[label_index, i])
        selected = finalize_spans(candidates, suppress=not case.get("allow_overlap", False))
        if case.get("top_k", 0):
            selected = selected[:case["top_k"]]
        results.append({"label": label, "spans": [
            {"text": surface, "probability": probability, "start": start, "end": end,
             "logit": by_offset[start, end]} for surface, probability, start, end in selected]})
    return {"case": case, "input_ids": batch.input_ids[0].tolist(),
            "label_positions": batch.schema_special_indices[0][0][1:],
            "word_positions": batch.text_word_indices[0, :n_words].tolist(),
            "start_words": starts, "end_words": ends,
            "span_logits": logits.flatten().tolist(), "count_logits": count_logits.tolist(),
            "predicted_count": predicted, "max_span_width": width, "groups": results}

def record_reference(oracle, case):
    from gliner2.inference.candidate_decoder import finalize_spans
    fields = case["fields"]
    name = case["structure"]
    schema = {"json_structures": [{name: {field["name"]: [] for field in fields}}],
              "json_descriptions": {name: {field["name"]: field["description"] for field in fields if "description" in field}}}
    batch, starts, ends, spans, queries, count_logits = span_forward(oracle, case, schema)
    predicted = int(count_logits.argmax())
    with torch.inference_mode():
        projected = oracle[3](queries[1:], predicted) if predicted else None
        logits = torch.einsum("sh,rfh->rfs", spans, projected) if predicted else torch.empty((0, len(fields), len(starts)))
        probabilities = logits.sigmoid()
    text, records = case["text"], []
    for slot in range(predicted):
        record = {}
        for f, field in enumerate(fields):
            candidates, by_offset = [], {}
            for i, (s, e) in enumerate(zip(starts, ends)):
                first, last = batch.start_mappings[0][s], batch.start_mappings[0][e - 1]
                end = min(len(text), batch.end_mappings[0][e - 1])
                if first >= len(text) or last >= len(text) or probabilities[slot, f, i] < case.get("threshold", 0.5):
                    continue
                start_byte, end_byte = len(text[:first].encode("utf8")), len(text[:end].encode("utf8"))
                candidates.append((text[first:end], float(probabilities[slot, f, i]), start_byte, end_byte))
                by_offset[start_byte, end_byte] = float(logits[slot, f, i])
            selected = finalize_spans(candidates, dtype="str" if field["type"] == "single" else "list",
                                      suppress=not case.get("allow_overlap", False))
            if field["type"] == "list" and case.get("top_k", 0):
                selected = selected[:case["top_k"]]
            values = [{"text": surface, "probability": probability, "start": start, "end": end,
                       "logit": by_offset[start, end]} for surface, probability, start, end in selected]
            record[field["name"]] = values if field["type"] == "list" else values[0] if values else None
        if any(value is not None and value != [] for value in record.values()):
            records.append({"slot_index": slot, "fields": record})
    return {"case": case, "structure": name, "records": records, "predicted_count": predicted,
            "input_ids": batch.input_ids[0].tolist(), "field_positions": batch.schema_special_indices[0][0][1:],
            "word_positions": batch.text_word_indices[0, :len(batch.text_tokens[0])].tolist(),
            "start_words": starts, "end_words": ends, "count_logits": count_logits.tolist(),
            "record_logits": logits.flatten().tolist(), "max_span_width": oracle[-1]}


def compare_spans(binary, model, expected, tolerance, backend):
    case = expected["case"]
    command = [str(binary), "--model", str(model), "--text", case["text"], "--threads", "4", "--max-tokens", "2048", "--debug"]
    for label in case["labels"]:
        command += ["--label", label]
    for label, desc in zip(case["labels"], case.get("descriptions", [])):
        command += ["--description", label + "=" + desc]
    for name, option in (("threshold", "--threshold"), ("top_k", "--top-k"), ("max_words", "--max-words")):
        if name in case:
            command += [option, str(case[name])]
    if case.get("allow_overlap"):
        command += ["--allow-overlap"]
    actual = json.loads(subprocess.check_output(command, encoding="utf8"))
    assert actual["backend"] == backend and actual["offset_unit"] == "utf8_bytes"
    boundary = "pool_capacity" in expected
    exact = ("pool_capacity",) if boundary else ("predicted_count",)
    for key in ("input_ids", "label_positions", "word_positions", "start_words", "end_words", "max_span_width", *exact):
        assert actual[key] == expected[key], key
    numerical = ("null_logits", "start_logits", "end_logits", "proposal_logits") if boundary else ("count_logits",)
    for key in ("span_logits", *numerical):
        np.testing.assert_allclose(actual[key], expected[key], atol=tolerance, rtol=tolerance, err_msg=key)
    assert len(actual["groups"]) == len(expected["groups"])
    for group, reference in zip(actual["groups"], expected["groups"]):
        assert group["label"] == reference["label"]
        assert [(x["text"], x["start"], x["end"]) for x in group["spans"]] == [
            (x["text"], x["start"], x["end"]) for x in reference["spans"]], (group, reference)
        for value, ref in zip(group["spans"], reference["spans"]):
            np.testing.assert_allclose([value["logit"], value["probability"]], [ref["logit"], ref["probability"]],
                                       atol=tolerance, rtol=tolerance)
    error = max((abs(a - b) for a, b in zip(actual["span_logits"], expected["span_logits"])), default=0)
    print(f"PASS spans {backend}: {len(actual['input_ids'])} tokens; {len(expected['span_logits'])} raw logits; max error {error:.8g}")

def compare_records(binary, model, expected, tolerance, backend):
    case = expected["case"]
    args = [str(binary), "--model", str(model), "--text", case["text"], "--structure", case["structure"],
            "--threads", "4", "--max-tokens", "2048", "--debug"]
    for field in case["fields"]:
        args += ["--single-field" if field["type"] == "single" else "--field", field["name"]]
        if "description" in field:
            args += ["--description", field["name"] + "=" + field["description"]]
    for name, option in (("threshold", "--threshold"), ("top_k", "--top-k"), ("max_words", "--max-words")):
        if name in case:
            args += [option, str(case[name])]
    if case.get("allow_overlap"):
        args += ["--allow-overlap"]
    boundary = "record_mode" in case
    if boundary:
        args += ["--record-mode", case["record_mode"], "--max-records", "4096"]
    actual = json.loads(subprocess.check_output(args, encoding="utf8"))
    assert actual["backend"] == backend and actual["offset_unit"] == "utf8_bytes"
    exact = ("record_mode", "record_count", "n_instances") if boundary else ("predicted_count",)
    for key in ("structure", "input_ids", "field_positions", "word_positions", "start_words", "end_words", "max_span_width", *exact):
        assert actual[key] == expected[key], key
    for key in (("pair_logits", "object_logits", "assignment_logits") if boundary else ("count_logits", "record_logits")):
        np.testing.assert_allclose(actual[key], expected[key], atol=tolerance, rtol=tolerance, err_msg=key)
    assert len(actual["records"]) == len(expected["records"]), (actual["records"], expected["records"])
    for record, ref in zip(actual["records"], expected["records"]):
        assert record["slot_index"] == ref["slot_index"]
        for field in case["fields"]:
            value, reference = record["fields"][field["name"]], ref["fields"][field["name"]]
            if field["type"] == "single":
                value, reference = ([] if value is None else [value]), ([] if reference is None else [reference])
            assert [(v["text"], v["start"], v["end"]) for v in value] == [(v["text"], v["start"], v["end"]) for v in reference]
            for v, r in zip(value, reference):
                np.testing.assert_allclose([v["logit"], v["probability"]], [r["logit"], r["probability"]],
                                           atol=tolerance, rtol=tolerance)
    key = "assignment_logits" if boundary else "record_logits"
    error = max((abs(a - b) for a, b in zip(actual[key], expected[key])), default=0)
    print(f"PASS records {backend}: {len(actual['input_ids'])} tokens; {actual.get('n_instances', actual.get('predicted_count'))} slots; "
          f"{len(actual['records'])} nonempty records; max logit error {error:.8g}")


def reference(oracle, case):
    processor, encoder, classifier = oracle
    tasks = case.get("tasks", [case])
    schema = []
    for spec in tasks:
        task = {"task": spec["task"], "labels": spec["labels"]}
        if spec.get("prompt"):
            task["prompt"] = spec["prompt"]
        if spec.get("descriptions"):
            task["label_descriptions"] = dict(zip(spec["labels"], spec["descriptions"]))
        schema.append(task)
    batch = processor.collate_fn_inference(
        [(case["text"], {"classifications": schema})],
        max_len=case.get("max_words"), error_policy="raise")
    with torch.inference_mode():
        outputs = encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True)
        _, schemas = processor.extract_embeddings_from_batch(outputs.last_hidden_state, batch.input_ids, batch)
        states = [torch.stack(group[1:]) for group in schemas[0]]
        logits = [classifier(group).squeeze(-1) for group in states]
    positions = [position for group in batch.schema_special_indices[0] for position in group[1:]]
    expected = {"case": case, "input_ids": batch.input_ids[0].tolist(), "label_positions": positions,
                "hidden_states": [layer[0].tolist() for layer in outputs.hidden_states],
                "label_states": torch.cat(states).tolist(), "logits": torch.cat(logits).tolist()}
    if "tasks" not in case:
        expected["probabilities"] = logits[0].softmax(-1).tolist()
        return expected
    results, probabilities, offsets = [], [], [0]
    for task, values in zip(tasks, logits):
        scaled = values.double() / task.get("temperature", 1.0)
        activation = task.get("activation", "sigmoid" if task.get("multi_label") else "softmax")
        probs = scaled.softmax(-1) if activation == "softmax" else scaled.sigmoid()
        result = {"task": task["task"], "scores": [
            {"label": label, "logit": float(value), "probability": float(prob)}
            for label, value, prob in zip(task["labels"], values, probs)]}
        if task.get("multi_label"):
            result["labels"] = [label for label, prob in zip(task["labels"], probs) if prob >= task.get("threshold", 0.5)]
        else:
            result["label"] = task["labels"][int(values.argmax())]
        results.append(result)
        probabilities.extend(probs.tolist())
        offsets.append(offsets[-1] + len(task["labels"]))
    expected.update(tasks=results, probabilities=probabilities, task_offsets=offsets)
    return expected


def cli_command(binary, model, case):
    command = [str(binary), "--model", str(model), "--text", case["text"],
               "--threads", "4", "--max-tokens", "2048", "--debug"]
    for task in case.get("tasks", [case]):
        command += ["--task", task["task"]]
        for label in task["labels"]:
            command += ["--label", label]
        if task.get("prompt"):
            command += ["--prompt", task["prompt"]]
        if task.get("descriptions"):
            for label, desc in zip(task["labels"], task["descriptions"]):
                command += ["--description", label + "=" + desc]
        if task.get("multi_label"):
            command += ["--multi-label"]
        for name in ("threshold", "activation", "temperature"):
            if name in task:
                command += ["--" + name, str(task[name])]
    if case.get("max_words") is not None:
        command += ["--max-words", str(case["max_words"])]
    return command


def compare(binary, model, expected, tolerance, directory, backend):
    dump = directory / "hidden.f32"
    if dump.exists():
        dump.unlink()
    command = cli_command(binary, model, expected["case"]) + ["--dump-hidden", str(dump)]
    actual = json.loads(subprocess.check_output(command, encoding="utf8"))
    assert actual["backend"] == backend, ("execution backend", actual["backend"], backend)
    assert actual["device"], "Missing execution device"
    assert actual["input_ids"] == expected["input_ids"], ("token IDs", actual["input_ids"], expected["input_ids"])
    assert actual["label_positions"] == expected["label_positions"], "label positions"
    states = np.asarray(expected["hidden_states"], dtype=np.float32)
    data = dump.read_bytes()
    tokens, hidden = struct.unpack_from("<II", data)
    assert (tokens, hidden) == states.shape[1:], (tokens, hidden, states.shape)
    actual_hidden = np.frombuffer(data, dtype="<f4", offset=8).reshape(states.shape)
    for i, (cpp, python) in enumerate(zip(actual_hidden, states)):
        np.testing.assert_allclose(cpp, python, atol=tolerance, rtol=tolerance, err_msg=f"encoder layer {i}")
    np.testing.assert_allclose(actual["label_states"], np.asarray(expected["label_states"]).flatten(),
                               atol=tolerance, rtol=tolerance, err_msg="label states")
    if "tasks" in expected:
        assert len(actual["tasks"]) == len(expected["tasks"]), "Task count"
        for result, golden in zip(actual["tasks"], expected["tasks"]):
            assert result["task"] == golden["task"], "Task order"
            assert [s["label"] for s in result["scores"]] == [s["label"] for s in golden["scores"]], "Label routing"
            key = "labels" if "labels" in golden else "label"
            assert result[key] == golden[key], ("Task decision", result, golden)
        assert actual["task_offsets"] == expected["task_offsets"], "Score offsets"
        scores = [score for task in actual["tasks"] for score in task["scores"]]
    else:
        scores = actual["scores"]
        assert actual["label"] == expected["case"]["labels"][int(np.argmax(expected["logits"]))], "chosen label"
    logits = [score["logit"] for score in scores]
    np.testing.assert_allclose(logits, expected["logits"], atol=tolerance, rtol=tolerance, err_msg="raw logits")
    np.testing.assert_allclose([score["probability"] for score in scores], expected["probabilities"],
                               atol=tolerance, rtol=tolerance, err_msg="softmax")
    error = float(np.max(np.abs(actual_hidden - states)))
    print(f"PASS {backend}/{actual['device']}: {len(actual['input_ids'])} tokens; "
          f"all {len(states)} hidden states; max error {error:.8g}; logits {logits}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--expect-backend", choices=("cpu", "cuda", "metal"), help="Assert the binary's build backend; does not select it")
    parser.add_argument("--write-golden", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--batch-only", action="store_true", help="Run/write only joint-schema cases")
    group.add_argument("--single-only", action="store_true", help="Run/write only single-task cases")
    group.add_argument("--kernel-only", action="store_true", help="Generate/test short and long sequences with a 32-wide synthetic encoder")
    group.add_argument("--spans-only", action="store_true", help="Generate/test grouped entity span extraction using gliner-extract")
    group.add_argument("--records-only", action="store_true", help="Generate/test count-conditioned records using gliner-extract")
    group.add_argument("--multilingual-only", action="store_true", help="Compare French, Chinese, Arabic and Spanish classification inputs")
    group.add_argument("--boundary-spans-only", action="store_true", help="Generate/test shared-pool boundary entity extraction")
    group.add_argument("--boundary-records-only", action="store_true", help="Generate/test anchorless and latent boundary records")
    parser.add_argument("--synthetic-dtype", choices=("F32", "F16"), default="F32",
                        help="Stored tensor dtype when generating a synthetic checkpoint")
    parser.add_argument("--case", type=int, action="append")
    parser.add_argument("--tolerance", type=float, default=3e-4)
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be finite and positive")
    if not args.write_golden and not (args.checkpoint and args.gguf and args.binary):
        parser.error("Provide --checkpoint, --gguf and --binary, or --write-golden")
    backend = None
    if args.binary:
        backend = json.loads(subprocess.check_output([str(args.binary), "--build-info"], encoding="utf8"))["backend"]
        assert backend in ("cpu", "cuda", "metal"), backend
        if args.expect_backend is not None:
            assert backend == args.expect_backend, (backend, args.expect_backend)
        configure_parity_environment(backend)
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = directory
            create_checkpoint(checkpoint, hidden=32 if args.kernel_only else 8, layers=2,
                              compact_tokens=args.kernel_only, spans=args.spans_only or args.records_only,
                              records=args.records_only or args.boundary_records_only,
                              boundary=args.boundary_spans_only or args.boundary_records_only, dtype=args.synthetic_dtype)
        oracle = load_boundary_oracle(checkpoint, records=args.boundary_records_only) if args.boundary_spans_only or args.boundary_records_only else load_span_oracle(checkpoint) if args.spans_only or args.records_only else load_oracle(checkpoint)
        cases = (BOUNDARY_CASES if args.boundary_spans_only else MULTILINGUAL_CASES if args.multilingual_only else RECORD_CASES if args.records_only else
                 SPAN_CASES if args.spans_only else KERNEL_CASES if args.kernel_only else
                 BATCH_CASES if args.batch_only else CASES if args.single_only else CASES + BATCH_CASES)
        if args.boundary_records_only:
            cases = BOUNDARY_RECORD_CASES[:]
            if args.checkpoint:
                cases.append({**BOUNDARY_RECORD_CASES[1], "text": SPAN_CASES[5]["text"]})
            cases = [{**case, "record_mode": mode} for case in cases for mode in ("anchorless", "latent")]
        selected = cases if args.case is None else [cases[i] for i in args.case]
        fixtures = []
        for case in selected:
            if args.records_only and args.checkpoint is None:
                with torch.no_grad():
                    oracle[4][2].bias.zero_()
                    oracle[4][2].bias[case["synthetic_count"]] = 1.
            expected = boundary_reference(oracle, case, dense=args.synthetic_dtype == "F16" and args.checkpoint is None) if args.boundary_spans_only or args.boundary_records_only else record_reference(oracle, case) if args.records_only else span_reference(oracle, case) if args.spans_only else reference(oracle, case)
            fixtures.append(expected)
            if args.binary and args.gguf:
                if args.records_only or args.boundary_records_only:
                    compare_records(args.binary, args.gguf, expected, args.tolerance, backend)
                elif args.spans_only or args.boundary_spans_only:
                    compare_spans(args.binary, args.gguf, expected, args.tolerance, backend)
                else:
                    compare(args.binary, args.gguf, expected, args.tolerance, directory, backend)
        if args.write_golden:
            args.write_golden.parent.mkdir(parents=True, exist_ok=True)
            args.write_golden.write_text(json.dumps({
                "provenance": {"gliner2_commit": "1a80c9c85272cd6809009b102b770c452e928196",
                               "transformers": transformers.__version__, "torch": torch.__version__,
                               "semantics_checked_at": "55656fbfa01d3d4a77485e1a1eeeaf682990ccdf",
                               "generator": "tests/test_parity.py " + ("--batch-only " if args.batch_only else
                                                                      "--single-only " if args.single_only else
                                                                      "--kernel-only " if args.kernel_only else
                                                                      "--spans-only " if args.spans_only else
                                                                      "--records-only " if args.records_only else
                                                                      "--boundary-records-only " if args.boundary_records_only else
                                                                      "--multilingual-only " if args.multilingual_only else
                                                                      "--boundary-spans-only " if args.boundary_spans_only else "") +
                                                    f"--synthetic-dtype {args.synthetic_dtype} --write-golden"},
                "fixtures": fixtures}, separators=(",", ":")), encoding="utf8")


if __name__ == "__main__":
    main()
