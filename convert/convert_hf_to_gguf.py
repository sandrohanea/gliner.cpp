#!/usr/bin/env python3
"""Stream a local GLiNER2.5-Decide safetensors checkpoint into GGUF v3.

Only the Python standard library is needed. Tensor bytes stay in their original
F32/F16 format; GGUF dimensions are reversed to match ggml's layout.
"""

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path


GGUF_F32 = 0
GGUF_F16 = 1
KV_U32 = 4
KV_STRING = 8
KV_ARRAY = 9
KV_F32 = 6
ALIGNMENT = 32
CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class Tensor:
    name: str
    path: Path
    start: int
    size: int
    shape: tuple[int, ...]
    ggml_type: int


def u32(value):
    return struct.pack("<I", value)


def u64(value):
    return struct.pack("<Q", value)


def gguf_string(value):
    data = value.encode("utf-8")
    return u64(len(data)) + data


def align(value):
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def checkpoint_files(directory):
    index = directory / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(read_json(index)["weight_map"].values()))
        return [directory / name for name in names]
    single = directory / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError("Expected model.safetensors or model.safetensors.index.json")


def tensors_in(path):
    with path.open("rb") as file:
        raw = file.read(8)
        if len(raw) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size > 100_000_000:
            raise ValueError(f"Unreasonable safetensors header size: {path}")
        header = json.loads(file.read(header_size))
        data_start = 8 + header_size
        file_size = path.stat().st_size
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = spec["dtype"]
        if dtype not in ("F32", "F16"):
            raise ValueError(f"Unsupported tensor {name} dtype {dtype}; expected F32 or F16")
        shape = tuple(spec["shape"])
        begin, end = spec["data_offsets"]
        element_size = 4 if dtype == "F32" else 2
        expected = element_size
        for dim in shape:
            if dim <= 0:
                raise ValueError(f"Invalid shape for {name}: {shape}")
            expected *= dim
        if end - begin != expected or begin < 0 or data_start + end > file_size:
            raise ValueError(f"Invalid data offsets for {name}")
        yield Tensor(name, path, data_start + begin, expected, shape,
                     GGUF_F32 if dtype == "F32" else GGUF_F16)


def metadata(directory):
    model_config = read_json(directory / "config.json")
    if model_config.get("architecture") != "span":
        raise ValueError("This converter targets GLiNER2.5-Decide's serialized span checkpoint")
    encoder_path = directory / "encoder_config" / "config.json"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing encoder config: {encoder_path}")
    encoder = read_json(encoder_path)
    if encoder.get("model_type") != "deberta-v2":
        raise ValueError("Only DeBERTa-v2/v3 encoders are supported")

    tokenizer = read_json(directory / "tokenizer.json")
    if tokenizer.get("model", {}).get("type") != "Unigram":
        raise ValueError("Only SentencePiece Unigram tokenizer.json files are supported")
    vocab = tokenizer["model"]["vocab"]
    tokens = [item[0] for item in vocab]
    scores = [float(item[1]) for item in vocab]
    for item in tokenizer.get("added_tokens", []):
        token_id = int(item["id"])
        if token_id < len(tokens):
            if tokens[token_id] != item["content"]:
                raise ValueError(f"Conflicting added token ID {token_id}")
            continue
        while len(tokens) < token_id:
            tokens.append("")
            scores.append(0.0)
        tokens.append(item["content"])
        scores.append(0.0)

    kv = [
        ("general.architecture", KV_STRING, "gliner2.5-decide"),
        ("general.name", KV_STRING, directory.name),
        ("gliner.architecture", KV_STRING, "span"),
        ("gliner.token_pooling", KV_STRING, model_config.get("token_pooling", "first")),
        ("deberta.hidden_size", KV_U32, int(encoder["hidden_size"])),
        ("deberta.num_hidden_layers", KV_U32, int(encoder["num_hidden_layers"])),
        ("deberta.num_attention_heads", KV_U32, int(encoder["num_attention_heads"])),
        ("deberta.intermediate_size", KV_U32, int(encoder["intermediate_size"])),
        ("deberta.max_position_embeddings", KV_U32, int(encoder["max_position_embeddings"])),
        ("tokenizer.ggml.model", KV_STRING, "unigram"),
        ("tokenizer.ggml.tokens", KV_ARRAY, (KV_STRING, tokens)),
        ("tokenizer.ggml.scores", KV_ARRAY, (KV_F32, scores)),
    ]
    return kv


def write_kv(file, key, kind, value):
    file.write(gguf_string(key))
    file.write(u32(kind))
    if kind == KV_STRING:
        file.write(gguf_string(value))
    elif kind == KV_U32:
        file.write(u32(value))
    elif kind == KV_ARRAY:
        item_kind, items = value
        file.write(u32(item_kind))
        file.write(u64(len(items)))
        for item in items:
            if item_kind == KV_STRING:
                file.write(gguf_string(item))
            elif item_kind == KV_F32:
                file.write(struct.pack("<f", item))
            else:
                raise ValueError(f"Unsupported array kind {item_kind}")
    else:
        raise ValueError(f"Unsupported metadata kind {kind}")


def convert(directory, output):
    kv = metadata(directory)
    tensors = []
    seen = set()
    for path in checkpoint_files(directory):
        if not path.is_file():
            raise FileNotFoundError(path)
        for tensor in tensors_in(path):
            if tensor.name in seen:
                raise ValueError(f"Duplicate tensor {tensor.name}")
            seen.add(tensor.name)
            tensors.append(tensor)
    tensors.sort(key=lambda tensor: tensor.name)
    for required in ("classifier.0.weight", "classifier.0.bias",
                     "classifier.2.weight", "classifier.2.bias"):
        if required not in seen:
            raise ValueError(f"Missing required classification tensor: {required}")

    offset = 0
    with output.open("wb") as target:
        target.write(b"GGUF")
        target.write(u32(3))
        target.write(u64(len(tensors)))
        target.write(u64(len(kv)))
        for key, kind, value in kv:
            write_kv(target, key, kind, value)
        for tensor in tensors:
            target.write(gguf_string(tensor.name))
            target.write(u32(len(tensor.shape)))
            for dim in reversed(tensor.shape):
                target.write(u64(dim))
            target.write(u32(tensor.ggml_type))
            target.write(u64(offset))
            offset = align(offset + tensor.size)
        target.write(b"\0" * (align(target.tell()) - target.tell()))
        for tensor in tensors:
            with tensor.path.open("rb") as source:
                source.seek(tensor.start)
                remaining = tensor.size
                while remaining:
                    block = source.read(min(remaining, CHUNK))
                    if not block:
                        raise OSError(f"Short read from {tensor.path}")
                    target.write(block)
                    remaining -= len(block)
            target.write(b"\0" * (align(target.tell()) - target.tell()))
    return len(tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local Hugging Face checkpoint directory")
    parser.add_argument("output", type=Path, help="Output GGUF path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    count = convert(args.checkpoint, args.output)
    print(f"Wrote {count} tensors to {args.output}")


if __name__ == "__main__":
    main()
