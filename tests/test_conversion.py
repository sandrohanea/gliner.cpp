#!/usr/bin/env python3
"""Standard-library-only conversion, C API and offline upstream parity tests."""

import importlib.util
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from tiny_checkpoint import create_checkpoint, write_safetensors


def run(binary, model, *args):
    return subprocess.check_output([binary, "--model", str(model), *args], encoding="utf8")


def failure(binary, model, *args):
    result = subprocess.run([binary, "--model", str(model), *args], capture_output=True, encoding="utf8")
    assert result.returncode != 0 and "error:" in result.stderr, result
    assert not result.stdout, result


def assert_close(actual, expected, tolerance, context):
    assert len(actual) == len(expected), (context, len(actual), len(expected))
    for index, (a, b) in enumerate(zip(actual, expected)):
        assert math.isfinite(a) and abs(a - b) <= tolerance * (1 + abs(b)), (context, index, a, b)


def preserved_tensors(path, tensors, dtype):
    with path.open("rb") as f:
        def integer():
            return struct.unpack("<I", f.read(4))[0]

        def size():
            return struct.unpack("<Q", f.read(8))[0]

        def string():
            return f.read(size()).decode("utf8")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                item, length = integer(), size()
                return [value(item) for _ in range(length)]
            formats = {4: "<I", 6: "<f", 12: "<d"}
            return struct.unpack(formats[kind], f.read(struct.calcsize(formats[kind])))[0]

        assert f.read(4) == b"GGUF" and integer() == 3
        count, keys = size(), size()
        metadata = {}
        for _ in range(keys):
            key = string()
            metadata[key] = value(integer())
        assert metadata["gliner.format_version"] == 2
        records = []
        for _ in range(count):
            name = string()
            dims = tuple(size() for _ in range(integer()))
            kind, offset = integer(), size()
            records.append((name, dims, kind, offset))
        start = (f.tell() + 31) // 32 * 32
        assert {r[0] for r in records} == set(tensors), "Tensor pruning or renaming"
        for name, dims, kind, offset in records:
            shape, values = tensors[name]
            assert dims == tuple(reversed(shape)) and kind == (0 if dtype == "F32" else 1)
            expected = struct.pack("<" + ("f" if dtype == "F32" else "e") * len(values), *values)
            f.seek(start + offset)
            assert f.read(len(expected)) == expected, name


def offline_parity(converter, binary, root):
    golden = json.loads((Path(__file__).parent / "fixtures" / "tiny_parity.json").read_text(encoding="utf8"))
    for dtype in ("F32", "F16"):
        tensors = create_checkpoint(root, hidden=8, layers=2, dtype=dtype)
        model = root / f"parity-{dtype}.gguf"
        converter.convert(root, model)
        preserved_tensors(model, tensors, dtype)
        tolerance = 5e-5 if dtype == "F32" else 5e-3
        for index, expected in enumerate(golden["fixtures"]):
            case = expected["case"]
            dump = root / f"hidden-{dtype}-{index}.f32"
            args = ["--task", case["task"], "--text", case["text"], "--threads", "2",
                    "--max-tokens", "2048", "--debug", "--dump-hidden", str(dump)]
            for label in case["labels"]:
                args += ["--label", label]
            if case.get("prompt"):
                args += ["--prompt", case["prompt"]]
            if case.get("descriptions"):
                for label, desc in zip(case["labels"], case["descriptions"]):
                    args += ["--description", label + "=" + desc]
            if "max_words" in case:
                args += ["--max-words", str(case["max_words"])]
            actual = json.loads(run(binary, model, *args))
            assert actual["input_ids"] == expected["input_ids"], (case, actual["input_ids"], expected["input_ids"])
            assert actual["label_positions"] == expected["label_positions"], case
            assert_close(actual["label_states"], [v for row in expected["label_states"] for v in row], tolerance, "label states")
            assert_close([s["logit"] for s in actual["scores"]], expected["logits"], tolerance, "logits")
            assert_close([s["probability"] for s in actual["scores"]], expected["probabilities"], tolerance, "probabilities")
            winner = max(range(len(case["labels"])), key=lambda i: expected["logits"][i])
            assert actual["label"] == case["labels"][winner]
            data = dump.read_bytes()
            n_tokens, hidden = struct.unpack_from("<II", data)
            assert n_tokens == len(expected["input_ids"]) and hidden == 8
            outputs = struct.unpack("<" + "f" * ((len(data) - 8) // 4), data[8:])
            assert len(outputs) == len(expected["hidden_states"]) * n_tokens * hidden
            for layer, reference in enumerate(expected["hidden_states"]):
                begin = layer * n_tokens * hidden
                assert_close(outputs[begin:begin + n_tokens * hidden], [v for row in reference for v in row],
                             tolerance, f"encoder layer {layer}, {dtype}, fixture {index}")


def negative_conversion(converter, binary, root):
    tensors = create_checkpoint(root)
    model = root / "valid.gguf"
    converter.convert(root, model)

    def rejects(message=None):
        try:
            converter.convert(root, root / "invalid.gguf")
        except (ValueError, KeyError) as error:
            if message:
                assert message in str(error), error
            return
        raise AssertionError("Invalid checkpoint accepted")

    original = tensors.pop("encoder.encoder.layer.0.attention.self.query_proj.weight")
    write_safetensors(root / "model.safetensors", tensors)
    rejects()
    tensors["encoder.encoder.layer.0.attention.self.query_proj.weight"] = original
    tensors["classifier.0.weight"] = ([2, 4], original[1] * 2)
    write_safetensors(root / "model.safetensors", tensors)
    rejects()
    create_checkpoint(root)
    config_path = root / "encoder_config" / "config.json"
    config = json.loads(config_path.read_text())
    config["position_biased_input"] = True
    config_path.write_text(json.dumps(config))
    rejects()
    create_checkpoint(root)
    tokenizer_path = root / "tokenizer.json"
    tokenizer = json.loads(tokenizer_path.read_text())
    tokenizer["normalizer"] = {"type": "NFKC"}
    tokenizer_path.write_text(json.dumps(tokenizer))
    rejects()
    create_checkpoint(root)
    source = root / "model.safetensors"
    data = source.read_bytes()
    header_size = struct.unpack_from("<Q", data)[0]
    header = json.loads(data[8:8 + header_size])
    header["encoder.embeddings.LayerNorm.bias"]["data_offsets"] = header["encoder.embeddings.LayerNorm.weight"]["data_offsets"]
    encoded = json.dumps(header).encode()
    source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + header_size:])
    rejects("Overlapping")
    damaged = root / "missing.gguf"
    damaged.write_bytes(model.read_bytes().replace(b"attention.self.query_proj.weight", b"attention.self.bogus_proj.weight"))
    failure(binary, damaged, "--inspect")
    damaged.write_bytes(model.read_bytes().replace(b"decide-v1", b"invalid-v"))
    failure(binary, damaged, "--inspect")
    legacy = root / "legacy.gguf"
    legacy.write_bytes(model.read_bytes().replace(b"gliner.format_version", b"gliner.legacy_version"))
    assert "text_inference: no" in run(binary, legacy, "--inspect")
    legacy_result = json.loads(run(binary, legacy, "--labels", "first,second", "--states", str(root / "states.txt")))
    assert_close([s["logit"] for s in legacy_result["scores"]], [5.5, 3.5], 1e-5, "legacy head")
    failure(binary, legacy, "--task", "intent", "--labels", "first,second", "--text", "hello")

    tensors = create_checkpoint(root)
    names = list(tensors)
    first, second = names[:len(names) // 2], names[len(names) // 2:]
    write_safetensors(root / "part1.safetensors", {key: tensors[key] for key in first})
    write_safetensors(root / "part2.safetensors", {key: tensors[key] for key in second})
    mapping = {key: "part1.safetensors" if key in first else "part2.safetensors" for key in names}
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": mapping}))
    sharded = root / "sharded.gguf"
    converter.convert(root, sharded)
    preserved_tensors(sharded, tensors, "F32")
    del mapping[names[0]]
    index.write_text(json.dumps({"weight_map": mapping}))
    rejects()


def main(converter_path, binary, c_test):
    spec = importlib.util.spec_from_file_location("converter", converter_path)
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        states = root / "states.txt"
        states.write_text("1 2\n-1 2\n", encoding="utf8")
        for dtype in ("F32", "F16"):
            tensors = create_checkpoint(root, dtype=dtype)
            model = root / f"model-{dtype}.gguf"
            subprocess.run([sys.executable, converter_path, str(root), str(model)], check=True)
            preserved_tensors(model, tensors, dtype)
            inspected = run(binary, model, "--inspect")
            assert "tensors: 27" in inspected and "text_inference: yes" in inspected, inspected
            result = json.loads(run(binary, model, "--labels", "first,second", "--states", str(states), "--threads", "2"))
            assert result["label"] == "first", result
            assert_close([s["logit"] for s in result["scores"]], [5.5, 3.5], 1e-5, "head")
            assert_close([s["probability"] for s in result["scores"]],
                         [1 / (1 + math.exp(-v)) for v in (5.5, 3.5)], 1e-5, "sigmoid")
            result = json.loads(run(binary, model, "--labels", "first,second", "--states", str(states),
                                    "--multi-label", "--threshold", "0.98"))
            assert result["labels"] == ["first"]
            result = json.loads(run(binary, model, "--labels", "first,second", "--states", str(states),
                                    "--activation", "softmax", "--temperature", "2"))
            assert_close([s["probability"] for s in result["scores"]],
                         [1 / (1 + math.exp(-1)), 1 / (1 + math.exp(1))], 1e-6, "temperature")
            subprocess.run([c_test, str(model)], check=True)
            base = ["--labels", "first,second", "--task", "intent", "--text", "hello world"]
            for bad in (["--max-tokens", "2"], ["--threads", "0"], ["--threads", "2bad"],
                        ["--temperature", "nan"], ["--temperature", "0"], ["--states", str(states)],
                        ["--threshold", "0.5"], ["--unknown", "option"], ["--label", "third"]):
                failure(binary, model, *base, *bad)
            failure(binary, model, "--labels", "a,a", "--task", "intent", "--text", "x")
            failure(binary, model, "--labels", "a,", "--task", "intent", "--text", "x")
            failure(binary, model, "--labels", "a,b", "--text", "x")
            failure(binary, model, "--labels", "a,[SEP_TEXT]", "--task", "intent", "--text", "x")
            failure(binary, model, *base, "--description", "first=", "--description", "first=duplicate")
            text_file = root / "input.txt"
            text_file.write_text("hello world", encoding="utf8")
            from_file = json.loads(run(binary, model, "--labels", "first,second", "--task", "intent",
                                      "--text-file", str(text_file)))
            from_text = json.loads(run(binary, model, *base))
            assert from_file == from_text
        offline_parity(converter, binary, root)
        negative_conversion(converter, binary, root)
    print("Conversion, C API, CLI and offline F32/F16 encoder parity passed")


if __name__ == "__main__":
    main(*sys.argv[1:])
