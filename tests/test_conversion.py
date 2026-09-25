#!/usr/bin/env python3
"""Standard-library-only conversion, C API and offline upstream parity tests."""

import importlib.util
import argparse
import contextlib
import io
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from tiny_checkpoint import create_checkpoint, write_safetensors
from parity_environment import configure_parity_environment


def run(binary, model, *args):
    return subprocess.check_output([*binary, "--model", str(model), *args], encoding="utf8")


def failure(binary, model, *args):
    result = subprocess.run([*binary, "--model", str(model), *args], capture_output=True, encoding="utf8")
    assert result.returncode != 0 and "error:" in result.stderr, result
    assert not result.stdout, result
    return result


def assert_close(actual, expected, tolerance, context):
    assert len(actual) == len(expected), (context, len(actual), len(expected))
    for index, (a, b) in enumerate(zip(actual, expected)):
        allowed = tolerance * (1 + abs(b))
        if not math.isfinite(a) or abs(a - b) > allowed:
            raise AssertionError(
                f"{context}: index {index}, actual={a:.9g}, expected={b:.9g}, "
                f"absolute_error={abs(a - b):.9g}, allowed={allowed:.9g}")

def parity_environment_contract():
    original = dict(os.environ)
    with patch.dict(os.environ, {"NVIDIA_TF32_OVERRIDE": "1", "GGML_CUDA_CUBLAS_COMPUTE_TYPE": "f16"}):
        before = dict(os.environ)
        for backend in ("cpu", "metal"):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                configure_parity_environment(backend)
            assert dict(os.environ) == before and not output.getvalue(), backend
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            configure_parity_environment("cuda")
        assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
        assert os.environ["GGML_CUDA_CUBLAS_COMPUTE_TYPE"] == "f32"
        assert "NVIDIA_TF32_OVERRIDE=0" in output.getvalue()
        assert "GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32" in output.getvalue()
    assert dict(os.environ) == original


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


def offline_parity(converter, binary, root, backend):
    fixtures = []
    for name in ("tiny_parity.json", "tiny_batch_parity.json"):
        fixtures += json.loads((Path(__file__).parent / "fixtures" / name).read_text(encoding="utf8"))["fixtures"]
    kernel_fixtures = json.loads((Path(__file__).parent / "fixtures" / "tiny_cuda_parity.json").read_text(encoding="utf8"))["fixtures"]
    assert 3 < len(kernel_fixtures[0]["input_ids"]) <= 16
    assert len(kernel_fixtures[1]["input_ids"]) > 16
    groups = [(8, False, fixtures), (32, True, kernel_fixtures)]
    for dtype, width, compact, group in [(dtype, *group) for group in groups for dtype in ("F32", "F16")]:
        tensors = create_checkpoint(root, hidden=width, layers=2, dtype=dtype, compact_tokens=compact)
        model = root / f"parity-{width}-{dtype}.gguf"
        converter.convert(root, model)
        preserved_tensors(model, tensors, dtype)
        tolerance = 5e-5 if dtype == "F32" else 5e-3
        for index, expected in enumerate(group):
            case = expected["case"]
            dump = root / f"hidden-{width}-{dtype}-{index}.f32"
            args = ["--text", case["text"], "--threads", "2",
                    "--max-tokens", "2048", "--debug", "--dump-hidden", str(dump)]
            for task in case.get("tasks", [case]):
                args += ["--task", task["task"]]
                for label in task["labels"]:
                    args += ["--label", label]
                if task.get("prompt"):
                    args += ["--prompt", task["prompt"]]
                if task.get("descriptions"):
                    for label, desc in zip(task["labels"], task["descriptions"]):
                        args += ["--description", label + "=" + desc]
                if task.get("multi_label"):
                    args += ["--multi-label"]
                for name in ("threshold", "activation", "temperature"):
                    if name in task:
                        args += ["--" + name, str(task[name])]
            if "max_words" in case:
                args += ["--max-words", str(case["max_words"])]
            context = f"{backend}, {dtype}, hidden_size {width}, fixture {index}"
            print(f"Checking {context}", flush=True)
            actual = json.loads(run(binary, model, *args))
            assert actual["backend"] == backend, actual["backend"]
            assert actual["device"], "Missing execution device"
            assert actual["input_ids"] == expected["input_ids"], (case, actual["input_ids"], expected["input_ids"])
            assert actual["label_positions"] == expected["label_positions"], case
            data = dump.read_bytes()
            n_tokens, hidden = struct.unpack_from("<II", data)
            assert n_tokens == len(expected["input_ids"]) and hidden == width
            outputs = struct.unpack("<" + "f" * ((len(data) - 8) // 4), data[8:])
            assert len(outputs) == len(expected["hidden_states"]) * n_tokens * hidden
            for layer, reference in enumerate(expected["hidden_states"]):
                begin = layer * n_tokens * hidden
                assert_close(outputs[begin:begin + n_tokens * hidden], [v for row in reference for v in row],
                             tolerance, f"{context}, encoder layer {layer}")
            assert_close(actual["label_states"], [v for row in expected["label_states"] for v in row],
                         tolerance, f"{context}, label states")
            if "tasks" in expected:
                assert len(actual["tasks"]) == len(expected["tasks"])
                assert actual["task_offsets"] == expected["task_offsets"]
                for result, golden in zip(actual["tasks"], expected["tasks"]):
                    assert result["task"] == golden["task"]
                    assert [s["label"] for s in result["scores"]] == [s["label"] for s in golden["scores"]]
                    key = "labels" if "labels" in golden else "label"
                    assert result[key] == golden[key], (case, result, golden)
                scores = [s for task in actual["tasks"] for s in task["scores"]]
            else:
                scores = actual["scores"]
                winner = max(range(len(case["labels"])), key=lambda i: expected["logits"][i])
                assert actual["label"] == case["labels"][winner]
                assert actual["task_offsets"] == [0, len(case["labels"])]
            assert_close([s["logit"] for s in scores], expected["logits"], tolerance, f"{context}, logits")
            assert_close([s["probability"] for s in scores], expected["probabilities"], tolerance, f"{context}, probabilities")


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


def backend_contract(binary, model, backend):
    info = json.loads(subprocess.check_output([binary, "--build-info"], encoding="utf8"))
    assert info == {"backend": backend}, info
    raw = [binary]
    inspected = run(raw, model, "--inspect")
    assert f"backend: {backend}\n" in inspected and "device: " in inspected, inspected
    for args in (["--backend", "cpu"], ["--backend", "cuda"], ["--backend", "metal"],
                 ["--device", "CPU"], ["--list-devices"]):
        error = failure(raw, model, "--inspect", *args)
        assert "build-time only" in error.stderr, error.stderr
    failure(raw, model, "--build-info")


def batch_contract(binary, model, root):
    args = ["--task", "intent", "--labels", "first,second", "--multi-label", "--threshold", "0.4",
            "--temperature", "2", "--task", "sentiment", "--labels", "first,second,third"]
    actual = json.loads(run(binary, model, "--text", "hello world", *args, "--debug"))
    assert actual["task_offsets"] == [0, 2, 5], actual
    first, second = actual["tasks"]
    assert first["task"] == "intent" and second["task"] == "sentiment", actual
    assert "labels" in first and "label" not in first
    assert "label" in second and "labels" not in second
    assert_close([s["probability"] for s in first["scores"]],
                 [1 / (1 + math.exp(-s["logit"] / 2)) for s in first["scores"]], 1e-6, "task sigmoid")
    assert first["labels"] == [s["label"] for s in first["scores"] if s["probability"] >= 0.4]
    total = sum(math.exp(s["logit"]) for s in second["scores"])
    assert_close([s["probability"] for s in second["scores"]],
                 [math.exp(s["logit"]) / total for s in second["scores"]], 1e-6, "task-local softmax")
    assert math.isclose(sum(s["probability"] for s in second["scores"]), 1, abs_tol=1e-6)
    text_file = root / "joint-text.txt"
    text_file.write_text("hello world", encoding="utf8")
    assert json.loads(run(binary, model, "--text-file", str(text_file), *args, "--debug")) == actual
    cap = str(len(actual["input_ids"]))
    assert json.loads(run(binary, model, "--text", "hello world", *args, "--debug", "--max-tokens", cap)) == actual
    failure(binary, model, "--text", "hello world", *args, "--max-tokens", str(int(cap) - 1))
    before = json.loads(run(binary, model, "--text", "hello world extra", *args, "--max-words", "1", "--debug"))
    after = json.loads(run(binary, model, "--text", "hello other", *args, "--max-words", "1", "--debug"))
    assert before == after
    for invalid in (
        ["--task", "x", "--labels", "a,b", "--task", "x", "--labels", "a,b"],
        ["--task", "x", "--labels", "a,b", "--task", "y"],
        ["--task", "x", "--task", "y", "--labels", "a,b"],
        ["--task", "", "--labels", "a", "--task", "y", "--labels", "b"],
        ["--task", "x", "--labels", "a", "--task", "y", "--labels", "b,b"],
        ["--task", "x", "--labels", "a", "--task", "y", "--labels", "b", "--description", "a=not this task"],
        ["--task", "x", "--labels", "a", "--multi-label", "--task", "y", "--labels", "b", "--threshold", "0.5"],
    ):
        failure(binary, model, "--text", "hello", *invalid)
    failure(binary, model, *args, "--states", str(root / "states.txt"))


def main(converter_path, executable, c_test, expected_backend=None, fast_metal=False):
    parity_environment_contract()
    spec = importlib.util.spec_from_file_location("converter", converter_path)
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    backend = json.loads(subprocess.check_output([executable, "--build-info"], encoding="utf8"))["backend"]
    assert backend in ("cpu", "cuda", "metal"), backend
    if expected_backend is not None:
        assert backend == expected_backend, (backend, expected_backend)
    if fast_metal and backend != "metal":
        raise ValueError("--fast-metal requires a Metal build")
    configure_parity_environment(backend)
    binary = [executable]
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
            assert f"backend: {backend}\n" in inspected, inspected
            backend_contract(executable, model, backend)
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
            subprocess.run([c_test, str(model), backend], check=True)
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
            batch_contract(binary, model, root)
        if fast_metal:
            print("Skipping strict F32/F16 encoder goldens for fast Metal; conversion and API checks continue", flush=True)
        else:
            offline_parity(converter, binary, root, backend)
        tensors = create_checkpoint(root)
        large_count = (8 * 1024 * 1024 + 32) // 4
        tensors["span_rep.unused.weight"] = ([large_count], [0.0] * large_count)
        write_safetensors(root / "model.safetensors", tensors)
        large_model = root / "multi-chunk.gguf"
        converter.convert(root, large_model)
        subprocess.run([c_test, str(large_model), backend], check=True)
        negative_conversion(converter, binary, root)
    if fast_metal:
        print("Conversion, C API and CLI checks passed on fast Metal; strict encoder parity was not run")
    else:
        print(f"Conversion, C API, CLI and offline F32/F16 encoder parity passed on {backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("converter")
    parser.add_argument("binary")
    parser.add_argument("c_test")
    parser.add_argument("--expect-backend", choices=("cpu", "cuda", "metal"), help="Assert the binary's build backend; does not select it")
    parser.add_argument("--fast-metal", action="store_true", help="Skip strict encoder goldens for the intentional fast Metal build")
    args = parser.parse_args()
    main(args.converter, args.binary, args.c_test, args.expect_backend, args.fast_metal)
