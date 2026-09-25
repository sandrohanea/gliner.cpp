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


def preserved_tensors(path, tensors, dtype, version=2):
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
        assert metadata["gliner.format_version"] == version
        original = metadata.get("gliner.tensor_name_map.original", [])
        stored = metadata.get("gliner.tensor_name_map.stored", [])
        assert len(original) == len(stored) and len(set(stored)) == len(stored)
        aliases = dict(zip(stored, original))
        records = []
        for _ in range(count):
            name = string()
            assert len(name.encode("utf8")) < 64
            dims = tuple(size() for _ in range(integer()))
            kind, offset = integer(), size()
            records.append((aliases.get(name, name), dims, kind, offset))
        start = (f.tell() + 31) // 32 * 32
        assert {r[0] for r in records} == set(tensors), "Tensor pruning or renaming"
        for name, dims, kind, offset in records:
            shape, values = tensors[name]
            assert dims == tuple(reversed(shape)) and kind == (0 if dtype == "F32" else 1)
            expected = struct.pack("<" + ("f" if dtype == "F32" else "e") * len(values), *values)
            f.seek(start + offset)
            assert f.read(len(expected)) == expected, name

def boundary_contract(converter, binary, c_test, root, extract):
    states = root / "boundary-states.txt"
    states.write_text("1 2\n-1 2\n", encoding="utf8")
    for dtype in ("F32", "F16"):
        for dropout in (0.0, 0.1):
            tensors = create_checkpoint(root, dtype=dtype, boundary=True, classifier_dropout=dropout)
            model = root / f"boundary-{dtype}-{dropout}.gguf"
            converter.convert(root, model)
            preserved_tensors(model, tensors, dtype, version=3)
            inspect = run(binary, model, "--inspect")
            assert "architecture: boundary\n" in inspect and "span_extraction: yes" in inspect, inspect
            assert "record_extraction: no" in inspect, inspect
            result = json.loads(run(binary, model, "--labels", "first,second", "--states", str(states)))
            assert_close([s["logit"] for s in result["scores"]], [5.5, 3.5], 1e-5, "boundary head")
            result = json.loads(run(binary, model, "--text", "hello", "--task", "intent", "--labels", "first,second", "--debug"))
            assert result["architecture"] == "boundary"
            assert len(result["scores"]) == 2 and result["task_offsets"] == [0, 2]
            subprocess.run([c_test, "--boundary", str(model)], check=True)
            if extract:
                result = json.loads(run([extract], model, "--text", "hello", "--labels", "person", "--threshold", "0"))
                assert result["max_span_width"] is None and result["groups"][0]["spans"]
                error = failure([extract], model, "--text", "hello", "--structure", "requests", "--field", "terms")
                assert "Boundary record metadata is missing" in error.stderr
            legacy = root / f"boundary-classification-{dtype}-{dropout}.gguf"
            converter.convert(root, legacy, classification_only=True)
            preserved_tensors(legacy, tensors, dtype, version=3)
            subprocess.run([c_test, "--boundary", str(legacy)], check=True)
    tensors.pop("classifier.3.weight")
    write_safetensors(root / "model.safetensors", tensors)
    try:
        converter.convert(root, root / "boundary-missing.gguf")
    except ValueError as error:
        assert "classifier.3.weight" in str(error)
    else:
        raise AssertionError("Boundary classifier tensor validation was skipped")
    damaged = root / "boundary-capability.gguf"
    damaged.write_bytes(model.read_bytes().replace(b"classification", b"unsupportedxxx"))
    failure(binary, damaged, "--inspect")
    tensors = create_checkpoint(root, boundary=True)
    del tensors["boundary_head.shared_pool_scorer.content_pooler.value_projection.weight"]
    write_safetensors(root / "model.safetensors", tensors)
    try:
        converter.convert(root, root / "boundary-missing-content.gguf")
    except ValueError as error:
        assert "content_pooler.value_projection.weight" in str(error)
    else:
        raise AssertionError("Missing boundary content tensor accepted")
    create_checkpoint(root, boundary=True)
    path = root / "config.json"
    config = json.loads(path.read_text())
    for key, value in (("candidate_pool", "per_query"), ("candidate_attention_layers", 1),
                       ("query_attention_layers", 1), ("content_soft_max_pool", True),
                       ("use_inside_evidence", False), ("pair_temperature", 0),
                       ("pool_boundary_top_k", 257), ("boundary_attention_heads", 3)):
        invalid = {**config, "boundary_head": {**config["boundary_head"], key: value}}
        path.write_text(json.dumps(invalid), encoding="utf8")
        try:
            converter.convert(root, root / "boundary-invalid-config.gguf")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Unsupported boundary setting accepted: {key}")
    create_checkpoint(root)


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


def span_contract(converter, extract, c_test, root, backend, fast_metal, boundary=False):
    fixture_name = "tiny_boundary_parity.json" if boundary else "tiny_spans_parity.json"
    fixtures = json.loads((Path(__file__).parent / "fixtures" / fixture_name).read_text(encoding="utf8"))["fixtures"]
    for dtype in ("F32", "F16"):
        if boundary and dtype == "F16":
            fixtures = json.loads((Path(__file__).parent / "fixtures" / "tiny_boundary_f16_parity.json").read_text(encoding="utf8"))["fixtures"]
        tensors = create_checkpoint(root, hidden=8, layers=2, dtype=dtype, spans=not boundary, boundary=boundary)
        model = root / f"{'boundary-spans' if boundary else 'spans'}-{dtype}.gguf"
        converter.convert(root, model)
        preserved_tensors(model, tensors, dtype, version=3 if boundary else 2)
        subprocess.run([c_test, "--boundary" if boundary else "--spans", str(model)], check=True)
        for expected in fixtures:
            case = expected["case"]
            args = ["--text", case["text"], "--debug", "--threads", "2", "--max-tokens", "2048"]
            for label in case["labels"]:
                args += ["--label", label]
            for label, desc in zip(case["labels"], case.get("descriptions", [])):
                args += ["--description", label + "=" + desc]
            for name, option in (("threshold", "--threshold"), ("top_k", "--top-k"), ("max_words", "--max-words")):
                if name in case:
                    args += [option, str(case[name])]
            if case.get("allow_overlap"):
                args += ["--allow-overlap"]
            actual = json.loads(run([extract], model, *args))
            assert actual["backend"] == backend and actual["offset_unit"] == "utf8_bytes"
            exact = ["input_ids", "label_positions", "word_positions", "max_span_width"]
            if not boundary or (not fast_metal and dtype == "F32"):
                exact += ["start_words", "end_words"]
            for key in exact:
                assert actual[key] == expected[key], (dtype, key)
            if not fast_metal:
                tolerance = 5e-5 if dtype == "F32" else 5e-3
                reference_scores = expected
                if boundary and dtype == "F16":
                    pairs = list(zip(expected["all_start_words"], expected["all_end_words"]))
                    lookup = {pair: i for i, pair in enumerate(pairs)}
                    selected = [lookup[pair] for pair in zip(actual["start_words"], actual["end_words"])]
                    reference_scores = {**expected,
                        "span_logits": [expected["all_span_logits"][label * len(pairs) + i]
                                        for label in range(len(case["labels"])) for i in selected],
                        "proposal_logits": [expected["all_proposal_logits"][i] for i in selected]}
                for key in (("span_logits", "null_logits", "start_logits", "end_logits", "proposal_logits") if boundary else ("span_logits", "count_logits")):
                    assert_close(actual[key], reference_scores[key], tolerance, f"{dtype} {key}")
                if boundary:
                    assert actual["pool_capacity"] == expected["pool_capacity"]
                else:
                    assert actual["predicted_count"] == expected["predicted_count"]
            assert len(actual["groups"]) == len(case["labels"])
            source = case["text"].encode("utf8")
            for group, reference in zip(actual["groups"], expected["groups"]):
                assert group["label"] == reference["label"]
                if dtype == "F32" and not fast_metal:
                    assert [(s["text"], s["start"], s["end"]) for s in group["spans"]] == [
                        (s["text"], s["start"], s["end"]) for s in reference["spans"]], (group, reference)
                for span in group["spans"]:
                    assert 0 <= span["start"] < span["end"] <= len(source)
                    assert source[span["start"]:span["end"]].decode("utf8") == span["text"]
        base = ["--text", "hello", "--labels", "a,b"]
        for bad in (["--threshold", "-0.1"], ["--threshold", "1.1"], ["--threshold", "nan"],
                    ["--top-k", "-1"], ["--threads", "0"], ["--max-tokens", "2"],
                    ["--labels", "same,same"], ["--label", "extra"], ["--task", "unsupported"]):
            failure([extract], model, *base, *bad)
        closed = json.loads(run([extract], model, *base, "--threshold", "1"))
        assert all(not group["spans"] for group in closed["groups"])
    if boundary:
        tensors["boundary_head.null_projection.bias"] = ([1], [1.])
        write_safetensors(root / "model.safetensors", tensors)
        gated = root / "boundary-abstain.gguf"
        converter.convert(root, gated)
        result = json.loads(run([extract], gated, "--text", "hello", "--labels", "a,b", "--threshold", "0", "--debug"))
        assert result["null_logits"] == [1., 1.] and all(not g["spans"] for g in result["groups"])
        for name, (shape, values) in list(tensors.items()):
            if name.startswith("boundary_head."):
                tensors[name] = (shape, [0.] * len(values))
        # Zero null logits equal the abstention cutoff and must NOT suppress a label.
        write_safetensors(root / "model.safetensors", tensors)
        tied = root / "boundary-tied.gguf"
        converter.convert(root, tied)
        result = json.loads(run([extract], tied, "--text", "alpha beta gamma", "--labels", "term", "--threshold", "0.5"))
        assert [s["text"] for s in result["groups"][0]["spans"]] == ["alpha", "beta", "gamma"], result
        config_path = root / "config.json"
        config = json.loads(config_path.read_text())
        config["boundary_head"]["pool_boundary_top_k"] = 1
        config["boundary_head"]["min_pool_per_query"] = 0
        config["boundary_head"]["boundary_attention_layers"] = 0
        config["boundary_head"]["boundary_refinement_layers"] = 0
        config_path.write_text(json.dumps(config), encoding="utf8")
        empty_pool = root / "boundary-empty-pool.gguf"
        converter.convert(root, empty_pool)
        result = json.loads(run([extract], empty_pool, "--text", "alpha beta", "--labels", "term", "--threshold", "0", "--debug"))
        assert not result["span_logits"] and not result["groups"][0]["spans"]
        config["boundary_head"]["pool_boundary_top_k"] = 4
        config["boundary_head"]["pair_temperature"] = 2.
        config_path.write_text(json.dumps(config), encoding="utf8")
        tensors["boundary_head.shared_pool_scorer.film_output.3.bias"] = ([1], [2.])
        write_safetensors(root / "model.safetensors", tensors)
        scaled = root / "boundary-temperature.gguf"
        converter.convert(root, scaled)
        result = json.loads(run([extract], scaled, "--text", "alpha beta", "--labels", "term", "--debug"))
        assert result["pair_temperature"] == 2.
        assert_close([s["probability"] for s in result["groups"][0]["spans"]], [1 / (1 + math.exp(-1))] * 2, 5e-5, "boundary temperature")
        config["boundary_head"]["pool_size"] = 1
        config_path.write_text(json.dumps(config), encoding="utf8")
        for side in ("start", "end"):
            tensors[f"boundary_head.boundary_query_head.{side}_boundary_projection.bias"] = ([8], [1.] * 8)
            tensors[f"boundary_head.boundary_query_head.{side}_query_projection.bias"] = ([8], [-10000.] * 8)
        write_safetensors(root / "model.safetensors", tensors)
        masked = root / "boundary-masked-proposals.gguf"
        converter.convert(root, masked)
        result = json.loads(run([extract], masked, "--text", "alpha beta", "--labels", "term", "--threshold", "0", "--debug"))
        assert not result["span_logits"] and not result["groups"][0]["spans"]
        return
    tensors["count_pred.2.bias"] = ([20], [2.] + [0.] * 19)
    write_safetensors(root / "model.safetensors", tensors)
    gated = root / "spans-gated.gguf"
    converter.convert(root, gated)
    result = json.loads(run([extract], gated, "--text", "hello", "--labels", "a,b", "--threshold", "0", "--debug"))
    assert result["predicted_count"] == 0 and all(not g["spans"] for g in result["groups"])
    del tensors["count_embed.gru.weight_ih_l0"]
    write_safetensors(root / "model.safetensors", tensors)
    try:
        converter.convert(root, root / "spans-invalid.gguf")
    except ValueError as error:
        assert "count_embed.gru.weight_ih_l0" in str(error)
    else:
        raise AssertionError("Missing span-conditioning tensor accepted")
    failure([extract], root / "model-F32.gguf", "--text", "hello", "--labels", "term")

def boundary_record_contract(converter, extract, c_test, root, backend, fast_metal):
    def arguments(case):
        args = ["--text", case["text"], "--structure", case["structure"], "--record-mode", case["record_mode"],
                "--max-records", "4096", "--threads", "2", "--max-tokens", "2048", "--debug"]
        for f in case["fields"]:
            args += ["--single-field" if f["type"] == "single" else "--field", f["name"]]
            if "description" in f:
                args += ["--description", f["name"] + "=" + f["description"]]
        for key, option in (("threshold", "--threshold"), ("top_k", "--top-k"), ("max_words", "--max-words")):
            if key in case:
                args += [option, str(case[key])]
        if case.get("allow_overlap"):
            args += ["--allow-overlap"]
        return args
    for dtype in ("F32", "F16"):
        name = "tiny_boundary_records_parity.json" if dtype == "F32" else "tiny_boundary_records_f16_parity.json"
        fixtures = json.loads((Path(__file__).parent / "fixtures" / name).read_text(encoding="utf8"))["fixtures"]
        tensors = create_checkpoint(root, hidden=8, layers=2, dtype=dtype, boundary=True, records=True)
        model = root / f"boundary-records-{dtype}.gguf"
        converter.convert(root, model)
        preserved_tensors(model, tensors, dtype, version=3)
        subprocess.run([c_test, "--boundary-records", str(model)], check=True)
        for expected in fixtures:
            case = expected["case"]
            actual = json.loads(run([extract], model, *arguments(case)))
            for key in ("input_ids", "field_positions", "word_positions", "record_mode", "n_instances", "max_span_width"):
                assert actual[key] == expected[key], (dtype, key)
            assert actual["backend"] == backend and actual["record_count"] == len(actual["records"])
            count, instances, fields = len(actual["start_words"]), actual["n_instances"], len(case["fields"])
            assert len(actual["assignment_logits"]) == fields * instances * (count + 1)
            if not fast_metal:
                pairs = list(zip(expected["start_words"], expected["end_words"]))
                native_pairs = list(zip(actual["start_words"], actual["end_words"]))
                if dtype == "F32":
                    assert native_pairs == pairs, case
                else:
                    assert set(native_pairs) == set(pairs), case
                lookup = {pair: i for i, pair in enumerate(pairs)}
                permutation = [lookup[pair] for pair in native_pairs]
                rows = list(range(instances)) if case["record_mode"] == "anchorless" else [
                    f * count + c for f in range(fields) for c in permutation]
                objects = [expected["object_logits"][i] for i in rows]
                assignments = [expected["assignment_logits"][(f * instances + i) * (count + 1) + c]
                               for f in range(fields) for i in rows for c in [0] + [p + 1 for p in permutation]]
                pairs = [expected["pair_logits"][f * count + c] for f in range(fields) for c in permutation]
                tolerance = 5e-5 if dtype == "F32" else 5e-3
                for key, values in (("object_logits", objects), ("assignment_logits", assignments), ("pair_logits", pairs)):
                    assert_close(actual[key], values, tolerance, f"boundary records {dtype} {case['record_mode']} {key}")
            if dtype == "F32" and not fast_metal:
                assert [r["slot_index"] for r in actual["records"]] == [r["slot_index"] for r in expected["records"]], case
            source = case["text"].encode("utf8")
            for r, record in enumerate(actual["records"]):
                for f in case["fields"]:
                    value = record["fields"][f["name"]]
                    spans = value if f["type"] == "list" else [value] if value else []
                    if dtype == "F32" and not fast_metal:
                        ref = expected["records"][r]["fields"][f["name"]]
                        ref = ref if f["type"] == "list" else [ref] if ref else []
                        assert [(v["text"], v["start"], v["end"]) for v in spans] == [
                            (v["text"], v["start"], v["end"]) for v in ref], (case, spans, ref)
                        assert_close([v["probability"] for v in spans], [v["probability"] for v in ref], 5e-5, "record confidence")
                    for span in spans:
                        assert 0 <= span["start"] < span["end"] <= len(source)
                        assert source[span["start"]:span["end"]].decode("utf8") == span["text"]
        base = ["--text", "alpha beta", "--structure", "requests", "--field", "terms"]
        for bad in (["--record-mode", "bad"], ["--max-records", "0"], ["--max-records", "4097"], ["--threshold", "nan"]):
            failure([extract], model, *base, *bad)
        default = json.loads(run([extract], model, *base, "--debug"))
        explicit = json.loads(run([extract], model, *base, "--record-mode", "anchorless", "--debug"))
        assert default == explicit
        failure([extract], model, "--text", "alpha", "--labels", "x", "--record-mode", "latent")
    # Orthogonal learned queries must produce distinct scalar assignments, not copies of slot zero.
    for name, (shape, values) in list(tensors.items()):
        if name.startswith("record_decoder."):
            tensors[name] = (shape, [0.] * len(values))
    tensors["record_decoder.object_head.bias"] = ([1], [2.])
    tensors["record_decoder.instance_embed"] = ([4, 8], [
        1, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0])
    tensors["record_decoder.inst_proj.weight"] = ([4, 8], [float(i == j) for i in range(4) for j in range(8)])
    tensors["record_decoder.cand_proj.weight"] = ([4, 8], [20. if i == j and i < 2 else 0. for i in range(4) for j in range(8)])
    tensors["record_decoder.null_embed"] = ([4], [0., 0., 0., -100.])
    write_safetensors(root / "model.safetensors", tensors)
    distinct = root / "boundary-records-distinct.gguf"
    converter.convert(root, distinct)
    distinct_args = ["--text", "Alice works at Contoso. Bob works at Fabrikam.", "--structure", "requests",
                     "--single-field", "term", "--threshold", "0", "--max-records", "4", "--debug"]
    result = json.loads(run([extract], distinct, *distinct_args))
    assert result["record_count"] >= 2 and len({r["fields"]["term"]["text"] for r in result["records"]}) >= 2
    error = failure([extract], distinct, *distinct_args, "--max-records", "1")
    assert "exceed max_records" in error.stderr
    subprocess.run([c_test, "--boundary-record-limit", str(distinct)], check=True)
    # No-object and empty-pool cases still expose raw diagnostics without fabricated records.
    for name in ("object_head", "latent_seed_head"):
        tensors[f"record_decoder.{name}.weight"] = ([1, 8], [0.] * 8)
        tensors[f"record_decoder.{name}.bias"] = ([1], [-100.])
    write_safetensors(root / "model.safetensors", tensors)
    closed = root / "boundary-records-no-object.gguf"
    converter.convert(root, closed)
    for mode in ("anchorless", "latent"):
        result = json.loads(run([extract], closed, *base, "--record-mode", mode, "--debug"))
        assert result["record_count"] == 0 and result["object_logits"] and all(v == -100 for v in result["object_logits"])
    config = json.loads((root / "config.json").read_text())
    config["boundary_head"]["pool_boundary_top_k"] = 1
    for name, (shape, values) in list(tensors.items()):
        if name.startswith("boundary_head."):
            tensors[name] = (shape, [0.] * len(values))
    (root / "config.json").write_text(json.dumps(config), encoding="utf8")
    write_safetensors(root / "model.safetensors", tensors)
    empty = root / "boundary-records-empty-pool.gguf"
    converter.convert(root, empty)
    for mode in ("anchorless", "latent"):
        result = json.loads(run([extract], empty, *base, "--record-mode", mode, "--threshold", "0", "--debug"))
        assert result["record_count"] == 0 and not result["pair_logits"]
        assert result["n_instances"] == (4 if mode == "anchorless" else 0)
    del tensors["record_decoder.inst_proj.weight"]
    write_safetensors(root / "model.safetensors", tensors)
    try:
        converter.convert(root, root / "boundary-records-invalid.gguf")
    except ValueError as error:
        assert "record_decoder.inst_proj.weight" in str(error)
    else:
        raise AssertionError("Missing record tensor accepted")
    config_path = root / "config.json"
    create_checkpoint(root, hidden=8, layers=2, boundary=True, records=True)
    config = json.loads(config_path.read_text())
    for key, value in (("record_dim", 0), ("record_instance_queries", 257), ("record_temperature", 0)):
        invalid = {**config, "boundary_head": {**config["boundary_head"], key: value}}
        config_path.write_text(json.dumps(invalid), encoding="utf8")
        try:
            converter.convert(root, root / "boundary-record-invalid-config.gguf")
        except ValueError as error:
            assert key in str(error)
        else:
            raise AssertionError(f"Invalid record setting accepted: {key}")


def record_contract(converter, extract, c_test, root, backend, fast_metal):
    fixtures = json.loads((Path(__file__).parent / "fixtures" / "tiny_records_parity.json").read_text(encoding="utf8"))["fixtures"]
    for dtype in ("F32", "F16"):
        models = {}
        for expected in fixtures:
            case = expected["case"]
            count = case["synthetic_count"]
            if count not in models:
                tensors = create_checkpoint(root, hidden=8, layers=2, dtype=dtype, spans=True, record_count=count, records=True)
                model = root / f"records-{dtype}-{count}.gguf"
                converter.convert(root, model)
                preserved_tensors(model, tensors, dtype)
                subprocess.run([c_test, "--records", str(model), str(count)], check=True)
                models[count] = model
            model = models[count]
            args = ["--text", case["text"], "--structure", case["structure"], "--threads", "2", "--max-tokens", "2048", "--debug"]
            for field in case["fields"]:
                args += ["--single-field" if field["type"] == "single" else "--field", field["name"]]
                if "description" in field:
                    args += ["--description", field["name"] + "=" + field["description"]]
            for name, option in (("threshold", "--threshold"), ("top_k", "--top-k"), ("max_words", "--max-words")):
                if name in case:
                    args += [option, str(case[name])]
            if case.get("allow_overlap"):
                args += ["--allow-overlap"]
            result = json.loads(run([extract], model, *args))
            assert result["backend"] == backend and result["offset_unit"] == "utf8_bytes"
            for key in ("structure", "input_ids", "field_positions", "word_positions", "start_words", "end_words", "predicted_count", "max_span_width"):
                assert result[key] == expected[key], (dtype, count, key)
            assert len(result["record_logits"]) == count * len(case["fields"]) * len(result["start_words"])
            if not fast_metal:
                for key in ("record_logits", "count_logits"):
                    assert_close(result[key], expected[key], 5e-5 if dtype == "F32" else 5e-3, f"{dtype} count {count} {key}")
            if dtype == "F32" and not fast_metal:
                assert len(result["records"]) == len(expected["records"])
            source = case["text"].encode("utf8")
            for r, record in enumerate(result["records"]):
                assert 0 <= record["slot_index"] < count
                assert list(record["fields"]) == [f["name"] for f in case["fields"]]
                for field in case["fields"]:
                    values = record["fields"][field["name"]]
                    if field["type"] == "single":
                        values = [] if values is None else [values]
                    for span in values:
                        assert 0 <= span["start"] < span["end"] <= len(source)
                        assert source[span["start"]:span["end"]].decode("utf8") == span["text"]
                    if dtype == "F32" and not fast_metal:
                        ref_record = expected["records"][r]
                        assert record["slot_index"] == ref_record["slot_index"]
                        ref = ref_record["fields"][field["name"]]
                        if field["type"] == "single":
                            ref = [] if ref is None else [ref]
                        assert [(v["text"], v["start"], v["end"]) for v in values] == [
                            (v["text"], v["start"], v["end"]) for v in ref], (
                                dtype, case["structure"], count, record["slot_index"], field["name"], values, ref)
        model = models[2]
        base = ["--text", "hello world", "--structure", "requests", "--field", "terms", "--threshold", "0"]
        for bad in (["--max-records", "0"], ["--max-records", "20"], ["--max-records", "1"],
                    ["--labels", "a,b"], ["--label", "x"], ["--field", "terms"],
                    ["--structure", "again"], ["--single-field", "terms"]):
            failure([extract], model, *base, *bad)
        failure([extract], model, "--text", "x", "--field", "terms")
        failure([extract], model, "--text", "x", "--structure", "requests")
        failure([extract], model, "--text", "x", "--labels", "a", "--max-records", "2")
        record = json.loads(run([extract], model, *base, "--debug"))
        cap = len(record["input_ids"])
        exact = json.loads(run([extract], model, *base, "--debug", "--max-tokens", str(cap)))
        assert exact == record
        failure([extract], model, *base, "--max-tokens", str(cap - 1))
        text_file = root / "records-input.txt"
        text_file.write_text("hello world", encoding="utf8")
        via_file = json.loads(run([extract], model, "--text-file", str(text_file), *base[2:], "--debug"))
        assert via_file == record
    failure([extract], root / "model-F32.gguf", "--text", "x", "--structure", "requests", "--field", "terms")


def main(converter_path, executable, c_test, expected_backend=None, fast_metal=False, extract=None):
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
        if extract:
            span_contract(converter, extract, c_test, root, backend, fast_metal)
            record_contract(converter, extract, c_test, root, backend, fast_metal)
        boundary_contract(converter, binary, c_test, root, extract)
        if extract:
            span_contract(converter, extract, c_test, root, backend, fast_metal, boundary=True)
            boundary_record_contract(converter, extract, c_test, root, backend, fast_metal)
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
    parser.add_argument("--extract", help="Native span extraction binary")
    args = parser.parse_args()
    main(args.converter, args.binary, args.c_test, args.expect_backend, args.fast_metal, args.extract)
