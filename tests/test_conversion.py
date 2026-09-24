#!/usr/bin/env python3
"""End-to-end GGUF conversion and GGML classifier check with a tiny checkpoint."""

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def write_safetensors(path, tensors, dtype="F32"):
    header = {}
    payload = bytearray()
    for name, (shape, values) in tensors.items():
        data = struct.pack("<" + ("f" if dtype == "F32" else "e") * len(values), *values)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def main(converter, binary, c_test):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "encoder_config").mkdir()
        (root / "config.json").write_text(json.dumps({"architecture": "span", "token_pooling": "first"}))
        (root / "encoder_config" / "config.json").write_text(json.dumps({
            "model_type": "deberta-v2", "hidden_size": 2, "num_hidden_layers": 1,
            "num_attention_heads": 1, "intermediate_size": 4,
            "max_position_embeddings": 128,
        }))
        (root / "tokenizer.json").write_text(json.dumps({
            "model": {"type": "Unigram", "vocab": [["<unk>", 0.0], ["▁hello", -1.0]]},
            "added_tokens": [{"id": 2, "content": "[L]"}],
        }))
        tensors = {
            "classifier.0.weight": ([4, 2], [1, 0, 0, 1, -1, 0, 0, -1]),
            "classifier.0.bias": ([4], [0, 0, 0, 0]),
            "classifier.2.weight": ([1, 4], [1, 2, -1, -2]),
            "classifier.2.bias": ([1], [0.5]),
            "encoder.embeddings.word_embeddings.weight": ([2, 2], [1, 2, 3, 4]),
        }
        states = root / "states.txt"
        states.write_text("1 2\n-1 2\n")
        for dtype in ("F32", "F16"):
            write_safetensors(root / "model.safetensors", tensors, dtype)
            model = root / f"model-{dtype}.gguf"
            subprocess.run([sys.executable, converter, str(root), str(model)], check=True)
            inspected = subprocess.check_output([binary, "--model", str(model), "--inspect"], text=True)
            assert "tensors: 5" in inspected, inspected
            output = subprocess.check_output([binary, "--model", str(model),
                                              "--labels", "first,second", "--states", str(states),
                                              "--threads", "2"], text=True)
            result = json.loads(output)
            assert result["label"] == "first", result
            for actual, expected in zip(result["scores"], [5.5, 3.5]):
                assert math.isclose(actual["logit"], expected, abs_tol=1e-5), result
                assert math.isclose(actual["probability"], 1 / (1 + math.exp(-expected)), abs_tol=1e-5)
            subprocess.run([c_test, str(model)], check=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
