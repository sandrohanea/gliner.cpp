#!/usr/bin/env python3
"""Optional Python CPU/CUDA F32 comparison for gliner-bench (uses requirements-parity.txt).

Loads only the upstream processor, encoder and classifier through the parity loader.
No compilation, quantization, autocast, decoding or hidden-layer instrumentation is used.
"""

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path


def nonnegative(value):
    if not re.fullmatch(r"\+?[0-9]+", value):
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    result = int(value)
    if result > 2**31 - 1:
        raise argparse.ArgumentTypeError("Integer is too large")
    return result


def new_task():
    return {"task": "", "labels": [], "prompt": "", "description_args": [],
            "comma_labels": False, "repeated_labels": False}


def device_argument(value):
    if value != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", value):
        raise argparse.ArgumentTypeError("Device must be cpu, cuda or cuda:N")
    return value


class TaskArgument(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        if namespace.tasks is None:
            namespace.tasks = [new_task()]
        task = namespace.tasks[-1]
        if option_string == "--task":
            if namespace.has_task:
                task = new_task()
                namespace.tasks.append(task)
            task["task"] = value
            namespace.has_task = True
        elif option_string == "--labels":
            task["labels"] = value.split(",")
            task["comma_labels"] = True
        elif option_string == "--label":
            task["labels"].append(value)
            task["repeated_labels"] = True
        elif option_string == "--prompt":
            task["prompt"] = value
        else:
            task["description_args"].append(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local HF checkpoint corresponding to the GGUF")
    parser.add_argument("--device", type=device_argument, default="cpu", help="cpu (default), cuda or cuda:N; no fallback")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--text")
    text.add_argument("--text-file", type=Path)
    parser.add_argument("--mode", choices=("joint", "separate", "both"), default="both")
    parser.add_argument("--threads", type=nonnegative, default=1)
    parser.add_argument("--max-tokens", type=nonnegative, default=512)
    parser.add_argument("--max-words", type=nonnegative, default=0)
    parser.add_argument("--warmup", type=nonnegative, default=3)
    parser.add_argument("--iterations", type=nonnegative, default=20)
    parser.set_defaults(tasks=None, has_task=False)
    for option in ("--task", "--labels", "--label", "--prompt", "--description"):
        parser.add_argument(option, action=TaskArgument)
    args = parser.parse_args(argv)
    if args.threads == 0 or args.iterations == 0 or not 1 <= args.max_tokens <= 4096:
        parser.error("Threads and iterations must be positive; max_tokens must be 1..4096")
    if not args.has_task:
        parser.error("Every question requires --task and labels")
    names, tasks = set(), []
    for task in args.tasks:
        if not task["task"] or task["task"] in names:
            parser.error("Task names must be nonempty and unique")
        names.add(task["task"])
        if task["comma_labels"] and task["repeated_labels"]:
            parser.error("Use either --labels or --label within each task")
        labels = task["labels"]
        if not labels or any(not label for label in labels) or len(set(labels)) != len(labels):
            parser.error("Every task requires nonempty unique labels")
        if len(labels) > args.max_tokens:
            parser.error("Task labels must fit max_tokens")
        if any(label in ("[SEP_TEXT]", "[SEP_STRUCT]") for label in labels):
            parser.error("Schema separators cannot be used as labels")
        descriptions = {}
        for value in task["description_args"]:
            label, equal, desc = value.partition("=")
            if not equal or label not in labels or label in descriptions:
                parser.error("Expected a unique --description LABEL=TEXT for the current task")
            descriptions[label] = desc
        prompt = task["task"] + ": " + task["prompt"] if task["prompt"] else task["task"]
        if prompt in ("[SEP_TEXT]", "[SEP_STRUCT]") and not descriptions:
            parser.error("Schema separators cannot be used as task names")
        tasks.append({"task": task["task"], "labels": labels, "prompt": task["prompt"],
                      "label_descriptions": [descriptions.get(label) for label in labels]})
    if args.mode != "separate" and sum(len(t["labels"]) for t in tasks) > args.max_tokens:
        parser.error("Joint task labels must fit max_tokens")
    args.tasks = tasks
    if args.text_file is not None:
        args.text = args.text_file.read_bytes().decode("utf8")
    strings = [args.text]
    for task in tasks:
        strings += [task["task"], task["prompt"], *task["labels"]]
        strings += [desc for desc in task["label_descriptions"] if desc is not None]
    if any("\0" in value for value in strings):
        parser.error("Text/schema cannot contain NUL")
    return args


def prepare_device(torch, specification):
    device = torch.device(specification)
    if device.type == "cuda":
        if torch.version.cuda is None:
            raise RuntimeError("This PyTorch installation is CPU-only. Install a CUDA-enabled PyTorch wheel in this environment.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device; check the NVIDIA driver and GPU visibility.")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is out of range")
        device = torch.device(f"cuda:{index}")
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return device


def make_request(oracle, torch, args, mode, device):
    processor, encoder, classifier = oracle
    schemas = []
    groups = [args.tasks] if mode == "joint" else [[task] for task in args.tasks]
    for group in groups:
        tasks = []
        for task in group:
            tasks.append({"task": task["task"], "labels": task["labels"], "prompt": task["prompt"],
                          "label_descriptions": {
                              label: desc for label, desc in zip(task["labels"], task["label_descriptions"])
                              if desc is not None}})
        schemas.append({"classifications": tasks})

    def request():
        tokens, logits, probabilities = [], [], []
        for schema in schemas:
            batch = processor.collate_fn_inference(
                [(args.text, schema)], max_len=args.max_words or None, error_policy="raise")
            n_tokens = int(batch.input_ids.shape[1])
            if n_tokens > args.max_tokens:
                raise ValueError("Token limit exceeded; increase max_tokens or truncate with max_words")
            if device.type == "cuda":
                batch = batch.to(device)
            encoded = encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                              output_hidden_states=False).last_hidden_state
            _, schema_states = processor.extract_embeddings_from_batch(encoded, batch.input_ids, batch)
            if len(schema_states[0]) != len(schema["classifications"]):
                raise ValueError("Unexpected upstream task count")
            for task, states in zip(schema["classifications"], schema_states[0]):
                if len(states) - 1 != len(task["labels"]):
                    raise ValueError("Unexpected upstream label-state count")
                label_states = torch.stack(states[1:])
                values = classifier(label_states).squeeze(-1)
                logits.extend(values.cpu().tolist())
                probabilities.extend(values.sigmoid().cpu().tolist())
            tokens.append(n_tokens)
        return tokens, logits, probabilities

    return request


def percentile(sorted_values, p):
    rank = (len(sorted_values) - 1) * p
    lo, hi = math.floor(rank), math.ceil(rank)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (rank - lo)


def measure(request, mode, args, synchronize=None):
    def checked(result, expected_tokens=None):
        tokens, logits, probabilities = result
        expected_passes = 1 if mode == "joint" else len(args.tasks)
        expected_scores = sum(len(t["labels"]) for t in args.tasks)
        if len(tokens) != expected_passes or any(not 0 < n <= args.max_tokens for n in tokens):
            raise ValueError("Unexpected token counts")
        if expected_tokens is not None and tokens != expected_tokens:
            raise ValueError("Token count changed between repetitions")
        if len(logits) != expected_scores or len(probabilities) != expected_scores:
            raise ValueError("Unexpected score count")
        if not all(math.isfinite(v) for v in logits + probabilities):
            raise ValueError("Nonfinite inference scores")
        return tokens, logits

    def timed_request():
        if synchronize is not None:
            synchronize()
        start = time.perf_counter_ns()
        output = request()
        if synchronize is not None:
            synchronize()
        return output, (time.perf_counter_ns() - start) / 1e6

    output, first_ms = timed_request()
    tokens, last_logits = checked(output)
    for _ in range(args.warmup):
        if synchronize is not None:
            synchronize()
        output = request()
        if synchronize is not None:
            synchronize()
        _, last_logits = checked(output, tokens)
    samples = []
    for _ in range(args.iterations):
        output, ms = timed_request()
        if not math.isfinite(ms) or ms <= 0:
            raise ValueError("Nonpositive or nonfinite benchmark duration")
        samples.append(ms)
        _, last_logits = checked(output, tokens)
    sorted_samples = sorted(samples)
    total = sum(samples)
    mean = total / len(samples)
    return {
        "mode": mode, "state_count": 0, "state_init_ms": None,
        "encoder_passes_per_request": len(tokens), "tokens_per_pass": tokens,
        "tokens_per_request": sum(tokens), "first_request_ms": first_ms, "measured_total_ms": total,
        "latency_ms": {"min": sorted_samples[0], "mean": mean, "median": percentile(sorted_samples, 0.5),
                       "p95": percentile(sorted_samples, 0.95), "max": sorted_samples[-1]},
        "requests_per_second": 1000 / mean, "questions_per_second": 1000 * len(args.tasks) / mean,
        "encoded_tokens_per_second": 1000 * sum(tokens) / mean, "samples_ms": samples, "last_logits": last_logits,
    }


def main(args):
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise ValueError("--checkpoint must be a local checkpoint directory")
    for name in ("config.json", "encoder_config/config.json", "tokenizer.json"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Missing checkpoint file: {name}")
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf8"))
    if config.get("architecture") != "span" or config.get("token_pooling", "first") != "first":
        raise ValueError("Expected the span architecture with first-subword pooling")
    # Reuse exactly the upstream components and weight loading used for numerical parity.
    from test_parity import load_oracle
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    start = time.perf_counter_ns()
    device = prepare_device(torch, args.device)
    oracle = load_oracle(checkpoint)
    _, encoder, classifier = oracle
    for module in (encoder, classifier):
        module.to(device=device, dtype=torch.float32).eval()
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    if synchronize is not None:
        synchronize()
    load_ms = (time.perf_counter_ns() - start) / 1e6
    if any(p.device != device or p.dtype != torch.float32
           for module in (encoder, classifier) for p in module.parameters()):
        raise ValueError(f"Python benchmark requires float32 parameters on {device}")
    results = []
    with torch.inference_mode():
        for mode in ("joint", "separate") if args.mode == "both" else (args.mode,):
            results.append(measure(make_request(oracle, torch, args, mode, device), mode, args, synchronize))
    return {
        "schema_version": 1, "implementation": "python-upstream-components",
        "backend": device.type, "device": str(device) if device.type == "cuda" else "CPU", "dtype": "float32",
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "cuda_runtime": torch.version.cuda,
        "cuda_compute_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        "synchronization": "torch.cuda.synchronize before/after each request" if synchronize is not None else "CPU synchronous",
        "torch_precision": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "gliner2": importlib.metadata.version("gliner2"),
        "torch_build_config": torch.__config__.show(),
        "checkpoint": str(checkpoint), "model_load_count": 1, "model_load_ms": load_ms,
        "model_load_scope": "device initialization, processor, encoder, classifier and synchronized device transfer; excludes Python imports and unused span/counting modules",
        "hidden_size": encoder.config.hidden_size, "encoder_layers": encoder.config.num_hidden_layers,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "max_tokens": args.max_tokens, "max_words": args.max_words,
        "text_bytes": len(args.text.encode("utf8")), "task_count": len(args.tasks),
        "warmup_requests": args.warmup, "iterations": args.iterations, "tasks": args.tasks,
        "percentile_method": "linear interpolation at (n-1)*p",
        "timing_scope": "upstream preprocessing, input transfers, encoder, marker extraction, classifier and host logits/sigmoids; one request answers all tasks",
        "tokenizer_cache": "upstream per-segment cache retained across requests and modes",
        "precision_environment": {key: os.environ.get(key) for key in
                                  ("NVIDIA_TF32_OVERRIDE", "GGML_CUDA_CUBLAS_COMPUTE_TYPE")},
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM")},
        "results": results,
    }


if __name__ == "__main__":
    try:
        args = parse_args()
        # Upstream initialization can print banners; keep stdout machine-readable JSON.
        with contextlib.redirect_stdout(sys.stderr):
            report = main(args)
        print(json.dumps(report, allow_nan=False, ensure_ascii=True))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
