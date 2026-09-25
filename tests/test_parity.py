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
    # F16 storage is evaluated with F32 activations in GGML.
    encoder.float().eval()
    classifier.float().eval()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(checkpoint / "tokenizer.json"),
                                        unk_token="[UNK]", pad_token="[PAD]",
                                        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    return SchemaTransformer(tokenizer=tokenizer), encoder, classifier


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
                              compact_tokens=args.kernel_only)
        oracle = load_oracle(checkpoint)
        cases = KERNEL_CASES if args.kernel_only else BATCH_CASES if args.batch_only else CASES if args.single_only else CASES + BATCH_CASES
        selected = cases if args.case is None else [cases[i] for i in args.case]
        fixtures = []
        for case in selected:
            expected = reference(oracle, case)
            fixtures.append(expected)
            if args.binary and args.gguf:
                compare(args.binary, args.gguf, expected, args.tolerance, directory, backend)
        if args.write_golden:
            args.write_golden.parent.mkdir(parents=True, exist_ok=True)
            args.write_golden.write_text(json.dumps({
                "provenance": {"gliner2_commit": "1a80c9c85272cd6809009b102b770c452e928196",
                               "transformers": transformers.__version__, "torch": torch.__version__,
                               "semantics_checked_at": "55656fbfa01d3d4a77485e1a1eeeaf682990ccdf",
                               "generator": "tests/test_parity.py " + ("--batch-only " if args.batch_only else
                                                                      "--single-only " if args.single_only else
                                                                      "--kernel-only " if args.kernel_only else "") + "--write-golden"},
                "fixtures": fixtures}, separators=(",", ":")), encoding="utf8")


if __name__ == "__main__":
    main()
