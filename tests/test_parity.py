"""Optional live oracle: compare tokenizer, every DeBERTa layer, states and logits.

The regular CTest suite replays committed synthetic golden fixtures without torch.
This script can regenerate those fixtures or check a real local checkpoint.
"""

import argparse
import json
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
    task = {"task": case["task"], "labels": case["labels"]}
    if case.get("prompt"):
        task["prompt"] = case["prompt"]
    if case.get("descriptions"):
        task["label_descriptions"] = dict(zip(case["labels"], case["descriptions"]))
    batch = processor.collate_fn_inference(
        [(case["text"], {"classifications": [task]})],
        max_len=case.get("max_words"), error_policy="raise")
    with torch.inference_mode():
        outputs = encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True)
        _, schemas = processor.extract_embeddings_from_batch(outputs.last_hidden_state, batch.input_ids, batch)
        states = torch.stack(schemas[0][0][1:])
        logits = classifier(states).squeeze(-1)
    positions = batch.schema_special_indices[0][0][1:]
    return {"case": case, "input_ids": batch.input_ids[0].tolist(), "label_positions": positions,
            "hidden_states": [layer[0].tolist() for layer in outputs.hidden_states],
            "label_states": states.tolist(), "logits": logits.tolist(),
            "probabilities": logits.softmax(-1).tolist()}


def cli_command(binary, model, case):
    command = [str(binary), "--model", str(model), "--task", case["task"], "--text", case["text"],
               "--threads", "4", "--max-tokens", "2048", "--debug"]
    for label in case["labels"]:
        command += ["--label", label]
    if case.get("prompt"):
        command += ["--prompt", case["prompt"]]
    if case.get("descriptions"):
        for label, desc in zip(case["labels"], case["descriptions"]):
            command += ["--description", label + "=" + desc]
    if case.get("max_words") is not None:
        command += ["--max-words", str(case["max_words"])]
    return command


def compare(binary, model, expected, tolerance, directory):
    dump = directory / "hidden.f32"
    if dump.exists():
        dump.unlink()
    command = cli_command(binary, model, expected["case"]) + ["--dump-hidden", str(dump)]
    actual = json.loads(subprocess.check_output(command, encoding="utf8"))
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
    logits = [score["logit"] for score in actual["scores"]]
    np.testing.assert_allclose(logits, expected["logits"], atol=tolerance, rtol=tolerance, err_msg="raw logits")
    np.testing.assert_allclose([score["probability"] for score in actual["scores"]], expected["probabilities"],
                               atol=tolerance, rtol=tolerance, err_msg="softmax")
    assert actual["label"] == expected["case"]["labels"][int(np.argmax(expected["logits"]))], "chosen label"
    error = float(np.max(np.abs(actual_hidden - states)))
    print(f"PASS {len(actual['input_ids'])} tokens; all {len(states)} hidden states; max error {error:.8g}; logits {logits}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--write-golden", type=Path)
    parser.add_argument("--case", type=int, action="append")
    parser.add_argument("--tolerance", type=float, default=3e-4)
    args = parser.parse_args()
    if not args.write_golden and not (args.checkpoint and args.gguf and args.binary):
        parser.error("Provide --checkpoint, --gguf and --binary, or --write-golden")
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = directory
            create_checkpoint(checkpoint, hidden=8, layers=2)
        oracle = load_oracle(checkpoint)
        selected = CASES if args.case is None else [CASES[i] for i in args.case]
        fixtures = []
        for case in selected:
            expected = reference(oracle, case)
            fixtures.append(expected)
            if args.binary and args.gguf:
                compare(args.binary, args.gguf, expected, args.tolerance, directory)
        if args.write_golden:
            args.write_golden.parent.mkdir(parents=True, exist_ok=True)
            args.write_golden.write_text(json.dumps({
                "provenance": {"gliner2_commit": "1a80c9c85272cd6809009b102b770c452e928196",
                               "transformers": transformers.__version__, "torch": torch.__version__,
                               "generator": "tests/test_parity.py --write-golden"},
                "fixtures": fixtures}, separators=(",", ":")), encoding="utf8")


if __name__ == "__main__":
    main()
