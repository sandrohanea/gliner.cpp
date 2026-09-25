"""Standard-library benchmark tests, with an optional live Python/C++ comparison."""

import argparse
import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bench_python import make_request, measure, parse_args, prepare_device


class FakeDevice:
    def __init__(self, name):
        parts = name.split(":")
        self.type = parts[0]
        self.index = int(parts[1]) if len(parts) > 1 else None


class PythonBenchmarkTests(unittest.TestCase):
    def parse(self, *args):
        return parse_args(["--checkpoint", "checkpoint", *args])

    def test_task_groups(self):
        args = self.parse("--text", "", "--labels", "yes,no", "--prompt", "Choose",
                          "--description", "yes=", "--task", "intent",
                          "--task", "topic", "--label", "first,label", "--label", "other")
        self.assertEqual(args.tasks, [
            {"task": "intent", "labels": ["yes", "no"], "prompt": "Choose", "label_descriptions": ["", None]},
            {"task": "topic", "labels": ["first,label", "other"], "prompt": "", "label_descriptions": [None, None]},
        ])
        self.assertEqual((args.threads, args.iterations, args.warmup, args.mode), (1, 20, 3, "both"))
        self.assertEqual(args.device, "cpu")
        again = self.parse("--text", "", "--task", "new", "--labels", "one")
        self.assertEqual(len(again.tasks), 1)
        for device in ("cpu", "cuda", "cuda:0", "cuda:1"):
            self.assertEqual(self.parse("--text", "", "--task", "x", "--labels", "a", "--device", device).device, device)

    def test_utf8_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "text.txt"
            path.write_bytes("Caf\u00e9\r\n\u4e2d\u6587".encode("utf8"))
            args = self.parse("--text-file", str(path), "--task", "topic", "--labels", "a,b")
            self.assertEqual(args.text, "Caf\u00e9\r\n\u4e2d\u6587")

    def test_invalid_arguments(self):
        base = ["--text", "hello", "--task", "intent", "--labels", "a,b"]
        for extra in (
            ["--iterations", "0"], ["--threads", "0"], ["--warmup", "-1"],
            ["--iterations", "2bad"], ["--iterations", "2147483648"], ["--iterations", "2_0"],
            ["--max-tokens", "4097"], ["--max-tokens", "0"],
            ["--max-words", "-1"], ["--mode", "parallel"], ["--backend", "cuda"],
            ["--device", "mps"], ["--device", "cuda:-1"], ["--device", "cuda:x"],
            ["--device", "cpu:0"], ["--text-file", "other.txt"], ["--label", "c"],
            ["--task", "intent", "--labels", "c"], ["--task", "next"],
            ["--description", "missing=desc"], ["--labels", "a,"],
            ["--labels", "a,a"], ["--labels", "[SEP_TEXT],a"],
            ["--description", "a=", "--description", "a=duplicate"],
            ["--text", "hello\0world"],
        ):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse(*base, *extra)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("--text", "", "--labels", "a,b")

    def test_measured_requests_exclude_first_and_warmup(self):
        args = SimpleNamespace(tasks=[{"labels": ["a"]}, {"labels": ["b"]}], max_tokens=512, warmup=3, iterations=4)
        calls = []

        def request():
            calls.append(1)
            return [12, 13], [float(len(calls)), 0.0], [0.5, 0.5]

        clocks = [0, 2_000_000, 10_000_000, 11_000_000, 20_000_000, 24_000_000,
                  30_000_000, 32_000_000, 40_000_000, 43_000_000]
        with patch("bench_python.time.perf_counter_ns", side_effect=clocks):
            result = measure(request, "separate", args)
        self.assertEqual(len(calls), 1 + args.warmup + args.iterations)
        self.assertEqual(result["first_request_ms"], 2)
        self.assertEqual(result["samples_ms"], [1, 4, 2, 3])
        self.assertEqual(result["measured_total_ms"], 10)
        self.assertEqual(result["latency_ms"]["median"], 2.5)
        self.assertAlmostEqual(result["latency_ms"]["p95"], 3.85)
        self.assertEqual(result["requests_per_second"], 400)
        self.assertEqual(result["questions_per_second"], 800)
        self.assertEqual(result["encoded_tokens_per_second"], 10000)
        self.assertEqual(result["last_logits"], [8.0, 0.0])
        self.assertEqual(result["tokens_per_request"], 25)
        self.assertEqual(result["state_count"], 0)
        self.assertIsNone(result["state_init_ms"])

    def test_single_sample_and_failure(self):
        args = SimpleNamespace(tasks=[{"labels": ["a"]}], max_tokens=512, warmup=0, iterations=1)
        with patch("bench_python.time.perf_counter_ns", side_effect=[0, 1_000_000, 2_000_000, 5_000_000]):
            result = measure(lambda: ([10], [0.1], [0.5]), "joint", args)
        self.assertEqual(set(result["latency_ms"].values()), {3.0})
        for output in (([10], [math.nan], [0.5]), ([0], [1], [0.5]), ([10], [], []), ([10, 20], [1], [0.5])):
            with self.subTest(output=output), self.assertRaises(ValueError):
                measure(lambda: output, "joint", args)
        with patch("bench_python.time.perf_counter_ns", return_value=0), self.assertRaises(ValueError):
            measure(lambda: ([10], [0.1], [0.5]), "joint", args)

    def test_cuda_precision_and_no_fallback(self):
        torch = MagicMock()
        torch.device.side_effect = FakeDevice
        torch.version.cuda = "12.6"
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 2
        for name, index in (("cuda", 0), ("cuda:1", 1)):
            device = prepare_device(torch, name)
            self.assertEqual((device.type, device.index), ("cuda", index))
            torch.cuda.set_device.assert_called_with(device)
            torch.set_float32_matmul_precision.assert_called_with("highest")
            self.assertIs(torch.backends.cuda.matmul.allow_tf32, False)
            self.assertIs(torch.backends.cudnn.allow_tf32, False)
        with self.assertRaisesRegex(ValueError, "out of range"):
            prepare_device(torch, "cuda:2")
        torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "cannot access a CUDA device"):
            prepare_device(torch, "cuda")
        torch.version.cuda = None
        with self.assertRaisesRegex(RuntimeError, "CPU-only"):
            prepare_device(torch, "cuda")
        torch.reset_mock()
        self.assertEqual(prepare_device(torch, "cpu").type, "cpu")
        torch.cuda.set_device.assert_not_called()
        torch.cuda.is_available.assert_not_called()
        torch.set_float32_matmul_precision.assert_not_called()

    def test_cuda_synchronization_order(self):
        args = SimpleNamespace(tasks=[{"labels": ["a"]}], max_tokens=512, warmup=1, iterations=2)
        events = []

        def clock():
            events.append("clock")
            return len(events) * 1_000_000

        def request():
            events.append("request")
            return [10], [0.1], [0.5]

        def synchronize():
            events.append("sync")

        with patch("bench_python.time.perf_counter_ns", side_effect=clock):
            result = measure(request, "joint", args, synchronize)
        timed = ["sync", "clock", "request", "sync", "clock"]
        self.assertEqual(events, timed + ["sync", "request", "sync"] + timed * 2)
        self.assertEqual(result["samples_ms"], [3, 3])
        with self.assertRaisesRegex(RuntimeError, "GPU failed"):
            measure(request, "joint", args, lambda: (_ for _ in ()).throw(RuntimeError("GPU failed")))

    def test_cuda_inputs_and_results_are_transferred(self):
        torch, processor, encoder, classifier = (MagicMock() for _ in range(4))
        args = self.parse("--text", "hello", "--task", "intent", "--label", "a")
        batch = processor.collate_fn_inference.return_value
        batch.input_ids.shape = (1, 3)
        transferred = batch.to.return_value
        processor.extract_embeddings_from_batch.return_value = ([], [[[object(), object()]]])
        values = classifier.return_value.squeeze.return_value
        values.cpu.return_value.tolist.return_value = [0.1]
        values.sigmoid.return_value.cpu.return_value.tolist.return_value = [0.5]
        device = FakeDevice("cuda:1")
        request = make_request((processor, encoder, classifier), torch, args, "joint", device)
        self.assertEqual(request(), ([3], [0.1], [0.5]))
        batch.to.assert_called_once_with(device)
        encoder.assert_called_once_with(input_ids=transferred.input_ids, attention_mask=transferred.attention_mask,
                                        output_hidden_states=False)
        processor.extract_embeddings_from_batch.assert_called_once_with(
            encoder.return_value.last_hidden_state, transferred.input_ids, transferred)
        batch.to.reset_mock()
        args.max_tokens = 2
        with self.assertRaisesRegex(ValueError, "Token limit exceeded"):
            request()
        batch.to.assert_not_called()


def compare_live(cpp_bench, converter, device):
    from tiny_checkpoint import create_checkpoint

    script = Path(__file__).with_name("bench_python.py")
    identity = json.loads(subprocess.check_output([str(cpp_bench), "--build-info"], encoding="utf8"))
    backend = "cuda" if device.startswith("cuda") else "cpu"
    if identity != {"backend": backend}:
        raise ValueError(f"The comparison test requires a {backend} gliner-bench")
    if backend == "cuda":
        from parity_environment import configure_parity_environment
        configure_parity_environment(backend)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        create_checkpoint(root, hidden=32, layers=2, compact_tokens=True)
        gguf = root / "model.gguf"
        subprocess.run([sys.executable, str(converter), str(root), str(gguf)], check=True)
        task_a = ["--task", "intent", "--labels", "yes,no", "--prompt", "Choose [L]",
                  "--description", "yes=", "--description", "no=not [P] applicable"]
        task_b = ["--task", "intent detail", "--labels", "yes,no,other"]

        def execute(command):
            result = subprocess.run(list(map(str, command)), capture_output=True, encoding="utf8")
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)

        for case in (
            ["--text", "Caf\u00e9 \u4e2d\u6587 hello world!", *task_a, *task_b, "--mode", "both"],
            ["--text", "", *task_a, "--mode", "joint"],
            ["--text", "hello world more words", *task_a, *task_b, "--mode", "separate", "--max-words", "1"],
        ):
            common = [*case, "--iterations", "2", "--warmup", "1", "--threads", "2"]
            python = execute([sys.executable, script, "--checkpoint", root, "--device", device, *common])
            cpp = execute([cpp_bench, "--model", gguf, *common])
            for key in ("schema_version", "backend", "threads", "max_tokens", "max_words",
                        "text_bytes", "task_count", "warmup_requests", "iterations", "tasks",
                        "percentile_method", "model_load_count"):
                assert python[key] == cpp[key], (key, python[key], cpp[key])
            assert python["dtype"] == "float32" and python["interop_threads"] == 1
            if backend == "cuda":
                assert python["cuda_runtime"] and python["device_name"]
                assert python["device"].startswith("cuda:")
                assert python["torch_precision"] == {
                    "float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}
                assert "torch.cuda.synchronize" in python["synchronization"]
            assert len(python["results"]) == len(cpp["results"])
            for py_row, cpp_row in zip(python["results"], cpp["results"]):
                for key in ("mode", "tokens_per_pass", "tokens_per_request", "encoder_passes_per_request"):
                    assert py_row[key] == cpp_row[key], (key, py_row[key], cpp_row[key])
                assert len(py_row["last_logits"]) == len(cpp_row["last_logits"])
                assert all(math.isclose(a, b, abs_tol=5e-5, rel_tol=5e-5)
                           for a, b in zip(py_row["last_logits"], cpp_row["last_logits"]))
                assert len(py_row["samples_ms"]) == 2 and all(v > 0 for v in py_row["samples_ms"])
        result = subprocess.run([sys.executable, str(script), "--checkpoint", str(root),
                                 "--device", device, "--text", "hello", *task_a, "--max-tokens", "2", "--iterations", "1"],
                                capture_output=True, encoding="utf8")
        assert result.returncode != 0 and not result.stdout and "Token limit exceeded" in result.stderr, result
    print("Python/C++ benchmark tasks, token counts and raw logits match")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpp-bench", type=Path, help="Also compare to this binary (requires parity dependencies and matching hardware)")
    parser.add_argument("--converter", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if bool(args.cpp_bench) != bool(args.converter):
        parser.error("--cpp-bench and --converter must be provided together")
    result = unittest.main(argv=[sys.argv[0]], exit=False).result
    if not result.wasSuccessful():
        sys.exit(1)
    if args.cpp_bench:
        compare_live(args.cpp_bench, args.converter, args.device)
