"""Optional live oracle: compare tokenizer, every DeBERTa layer, states and logits.

The regular CTest suite replays committed synthetic golden fixtures without torch.
This script can regenerate those fixtures or check a real local checkpoint.
"""

import argparse
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


def load_oracle(checkpoint):
    config = DebertaV2Config.from_pretrained(checkpoint / "encoder_config")
    with torch.device("meta"):
        encoder = DebertaV2Model(config)
        classifier = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, 2 * config.hidden_size),
                                         torch.nn.ReLU(), torch.nn.Linear(2 * config.hidden_size, 1))
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
    for key in ("input_ids", "label_positions", "word_positions", "start_words", "end_words", "predicted_count", "max_span_width"):
        assert actual[key] == expected[key], key
    for key in ("span_logits", "count_logits"):
        np.testing.assert_allclose(actual[key], expected[key], atol=tolerance, rtol=tolerance, err_msg=key)
    assert len(actual["groups"]) == len(expected["groups"])
    for group, reference in zip(actual["groups"], expected["groups"]):
        assert group["label"] == reference["label"]
        assert [(x["text"], x["start"], x["end"]) for x in group["spans"]] == [
            (x["text"], x["start"], x["end"]) for x in reference["spans"]], (group, reference)
        for value, ref in zip(group["spans"], reference["spans"]):
            np.testing.assert_allclose([value["logit"], value["probability"]], [ref["logit"], ref["probability"]],
                                       atol=tolerance, rtol=tolerance)
    error = max(abs(a - b) for a, b in zip(actual["span_logits"], expected["span_logits"]))
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
    actual = json.loads(subprocess.check_output(args, encoding="utf8"))
    assert actual["backend"] == backend and actual["offset_unit"] == "utf8_bytes"
    for key in ("structure", "input_ids", "field_positions", "word_positions", "start_words", "end_words", "predicted_count", "max_span_width"):
        assert actual[key] == expected[key], key
    for key in ("count_logits", "record_logits"):
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
    error = max((abs(a - b) for a, b in zip(actual["record_logits"], expected["record_logits"])), default=0)
    print(f"PASS records {backend}: {len(actual['input_ids'])} tokens; {actual['predicted_count']} slots; "
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
                              records=args.records_only)
        oracle = load_span_oracle(checkpoint) if args.spans_only or args.records_only else load_oracle(checkpoint)
        cases = RECORD_CASES if args.records_only else SPAN_CASES if args.spans_only else KERNEL_CASES if args.kernel_only else BATCH_CASES if args.batch_only else CASES if args.single_only else CASES + BATCH_CASES
        selected = cases if args.case is None else [cases[i] for i in args.case]
        fixtures = []
        for case in selected:
            if args.records_only and args.checkpoint is None:
                with torch.no_grad():
                    oracle[4][2].bias.zero_()
                    oracle[4][2].bias[case["synthetic_count"]] = 1.
            expected = record_reference(oracle, case) if args.records_only else span_reference(oracle, case) if args.spans_only else reference(oracle, case)
            fixtures.append(expected)
            if args.binary and args.gguf:
                if args.records_only:
                    compare_records(args.binary, args.gguf, expected, args.tolerance, backend)
                elif args.spans_only:
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
                                                                      "--records-only " if args.records_only else "") + "--write-golden"},
                "fixtures": fixtures}, separators=(",", ":")), encoding="utf8")


if __name__ == "__main__":
    main()
