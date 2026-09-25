"""Backend-neutral benchmark contract tests; no performance thresholds or ML dependencies."""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from parity_environment import configure_parity_environment
from tiny_checkpoint import create_checkpoint


def invoke(binary, args):
    result = subprocess.run([binary, *map(str, args)], capture_output=True, encoding="utf8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def fails(binary, args, message=None):
    result = subprocess.run([binary, *map(str, args)], capture_output=True, encoding="utf8")
    assert result.returncode != 0 and "error:" in result.stderr, result
    assert not result.stdout, "Failed benchmark emitted partial/success-shaped JSON"
    if message:
        assert message in result.stderr, result.stderr


def close(a, b):
    assert math.isfinite(a) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def verify(result, modes, iterations, warmup, tasks, backend, model):
    assert result["schema_version"] == 1
    assert result["backend"] == backend and result["device"]
    assert isinstance(result["build_type"], str) and result["compiler"]
    assert result["metal_precision_mode"] in (("strict_f32", "fast") if backend == "metal" else ("not_applicable",))
    assert result["model_file_bytes"] == model.stat().st_size
    assert result["model_load_count"] == 1
    assert math.isfinite(result["model_load_ms"]) and result["model_load_ms"] > 0
    assert result["task_count"] == tasks
    assert result["warmup_requests"] == warmup
    assert result["iterations"] == iterations
    assert result["percentile_method"] == "linear interpolation at (n-1)*p"
    assert [r["mode"] for r in result["results"]] == modes
    label_count = sum(len(t["labels"]) for t in result["tasks"])
    for row in result["results"]:
        passes = 1 if row["mode"] == "joint" else tasks
        assert row["state_count"] == passes and row["encoder_passes_per_request"] == passes
        assert len(row["tokens_per_pass"]) == passes
        assert all(0 < n <= result["max_tokens"] for n in row["tokens_per_pass"])
        assert row["tokens_per_request"] == sum(row["tokens_per_pass"])
        assert math.isfinite(row["state_init_ms"]) and row["state_init_ms"] > 0
        assert math.isfinite(row["first_request_ms"]) and row["first_request_ms"] > 0
        samples = row["samples_ms"]
        assert len(samples) == iterations and all(math.isfinite(x) and x > 0 for x in samples)
        close(row["measured_total_ms"], sum(samples))
        latency = row["latency_ms"]
        close(latency["min"], min(samples))
        close(latency["max"], max(samples))
        close(latency["mean"], statistics.mean(samples))
        close(latency["median"], statistics.median(samples))
        sorted_samples = sorted(samples)
        rank = (iterations - 1) * 0.95
        lo, hi = math.floor(rank), math.ceil(rank)
        close(latency["p95"], sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * (rank - lo))
        close(row["requests_per_second"], iterations * 1000 / sum(samples))
        close(row["questions_per_second"], row["requests_per_second"] * tasks)
        close(row["encoded_tokens_per_second"], row["requests_per_second"] * row["tokens_per_request"])
        assert len(row["last_logits"]) == label_count and all(math.isfinite(v) for v in row["last_logits"])
    for key in ("NVIDIA_TF32_OVERRIDE", "GGML_CUDA_CUBLAS_COMPUTE_TYPE"):
        assert result["precision_environment"][key] == os.environ.get(key)


def main(converter, bench, classify, expected_backend):
    backend = invoke(bench, ["--build-info"])["backend"]
    assert backend == expected_backend
    configure_parity_environment(backend)
    help_result = subprocess.run([bench, "--help"], capture_output=True, encoding="utf8")
    assert help_result.returncode == 0 and "--iterations" in help_result.stdout and "--mode" in help_result.stdout
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        text = 'hello world caf\u00e9 \u4e2d\u6587 "quoted"'
        input_file = root / "text.txt"
        input_file.write_text(text, encoding="utf8")
        task_a = ["--task", "intent", "--labels", "yes,no", "--prompt", 'Choose [L] "wisely"',
                  "--description", "yes=", "--description", "no=not applicable"]
        task_b = ["--task", "topic", "--label", "first,label", "--label", "second"]
        task_args = task_a + task_b
        for dtype in ("F32", "F16"):
            create_checkpoint(root, hidden=32, layers=2, dtype=dtype, compact_tokens=True)
            model = root / f"{dtype}.gguf"
            subprocess.run([sys.executable, converter, str(root), str(model)], check=True)
            args = ["--model", model, "--text-file", input_file, *task_args,
                    "--threads", "2", "--iterations", "5", "--warmup", "1"]
            result = invoke(bench, args)
            verify(result, ["joint", "separate"], 5, 1, 2, backend, model)
            assert result["text_bytes"] == len(text.encode("utf8"))
            assert result["threads"] == 2 and result["max_words"] == 0
            assert result["tasks"][0]["label_descriptions"] == ["", "not applicable"]
            assert result["tasks"][1]["labels"] == ["first,label", "second"]
            joint = invoke(classify, ["--model", model, "--text", text, *task_args, "--debug", "--threads", "2"])
            separate = [invoke(classify, ["--model", model, "--text", text, *task, "--debug", "--threads", "2"])
                        for task in (task_a, task_b)]
            references = [
                ([s["logit"] for task in joint["tasks"] for s in task["scores"]], [len(joint["input_ids"])]),
                ([s["logit"] for task in separate for s in task["scores"]], [len(task["input_ids"]) for task in separate]),
            ]
            for row, (logits, tokens) in zip(result["results"], references):
                assert row["tokens_per_pass"] == tokens
                assert all(math.isclose(a, b, abs_tol=1e-5, rel_tol=1e-5)
                           for a, b in zip(row["last_logits"], logits)), (row["last_logits"], logits)
            for mode in ("joint", "separate"):
                single = invoke(bench, ["--model", model, "--text", "", *task_a,
                                       "--mode", mode, "--iterations", "1", "--warmup", "0"])
                verify(single, [mode], 1, 0, 1, backend, model)
                assert single["text_bytes"] == 0
                single_latency = single["results"][0]["latency_ms"]
                assert len(set(single_latency.values())) == 1
            truncated = invoke(bench, ["--model", model, "--text", text, *task_args,
                                       "--max-words", "1", "--iterations", "2", "--warmup", "0"])
            verify(truncated, ["joint", "separate"], 2, 0, 2, backend, model)
            assert truncated["max_words"] == 1
            assert truncated["results"][0]["tokens_per_request"] < result["results"][0]["tokens_per_request"]

        base = ["--model", model, "--text", text, *task_args, "--iterations", "1", "--warmup", "0"]
        for bad in (["--iterations", "0"], ["--iterations", "-1"], ["--iterations", "2bad"],
                    ["--iterations", "99999999999999999"], ["--warmup", "-1"],
                    ["--threads", "0"], ["--max-tokens", "0"], ["--max-tokens", "4097"],
                    ["--max-words", "-1"], ["--mode", "parallel"], ["--unknown", "1"],
                    ["--text-file", input_file], ["--temperature", "1"],
                    ["--device", "CPU"], ["--backend", "cpu"], ["--dump-hidden", root / "hidden.f32"]):
            fails(bench, base + bad)
        fails(bench, base + ["--max-tokens", "2"], "error:")
        fails(bench, ["--model", model, "--text", text, "--task", "x", "--labels", "a",
                      "--task", "x", "--labels", "b"], "Task names")
        fails(bench, ["--model", model, "--text", text, "--task", "x"], "requires labels")
        fails(bench, ["--model", model, "--text", text, "--labels", "a,b"], "--task")
        fails(bench, ["--model", model, "--text", text, "--task", "x", "--labels", "a,a"], "unique")
        fails(bench, ["--model", model, "--text", text, "--task", "x", "--labels", "a,b", "--label", "c"])
        fails(bench, ["--model", model, "--task", "x", "--labels", "a,b"], "exactly one")
        fails(bench, base + ["--model", root / "missing.gguf"])
        fails(bench, ["--model", model, "--text-file", root / "missing.txt", *task_a], "Cannot open text file")
        bad_text = root / "bad.txt"
        bad_text.write_bytes(b"hello\x00world")
        fails(bench, ["--model", model, "--text-file", bad_text, *task_a], "NUL")
        bad_text.write_bytes(b"\xff")
        fails(bench, ["--model", model, "--text-file", bad_text, *task_a], "UTF-8")
        legacy = root / "legacy.gguf"
        legacy.write_bytes(model.read_bytes().replace(b"gliner.format_version", b"gliner.legacy_version"))
        fails(bench, base + ["--model", legacy], "text-capable GGUF")

        # In separate mode, each task has its own token budget; a joint run may not fit.
        short = ["--model", model, "--text", "", "--task", "a", "--labels", "a",
                 "--task", "b", "--labels", "b", "--iterations", "1", "--warmup", "0"]
        separate = invoke(bench, short + ["--mode", "separate"])
        cap = max(separate["results"][0]["tokens_per_pass"])
        invoke(bench, short + ["--mode", "separate", "--max-tokens", str(cap)])
        fails(bench, short + ["--mode", "both", "--max-tokens", str(cap)], "Token limit exceeded")
    print(f"Benchmark timing/statistics, task modes, result parity and error contracts passed on {backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("converter")
    parser.add_argument("bench")
    parser.add_argument("classify")
    parser.add_argument("--expect-backend", required=True, choices=("cpu", "cuda", "metal"))
    args = parser.parse_args()
    main(args.converter, args.bench, args.classify, args.expect_backend)
