#!/usr/bin/env python3
"""Stream a compatible GLiNER2 safetensors checkpoint into GGUF v3.

Only the Python standard library is needed. Tensor bytes stay in their original
F32/F16 format; GGUF dimensions are reversed to match ggml's layout.
Span checkpoints support classification/extraction; boundary checkpoints currently
support classification only. All tensors are preserved regardless of capability.
"""

import argparse
import hashlib
import json
import math
import struct
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


GGUF_F32 = 0
GGUF_F16 = 1
KV_U32 = 4
KV_STRING = 8
KV_ARRAY = 9
KV_F32 = 6
KV_F64 = 12
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
        mapping = read_json(index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Invalid safetensors weight_map")
        names = sorted(set(mapping.values()))
        for name in names:
            if not isinstance(name, str) or not (directory / name).resolve().is_relative_to(directory.resolve()):
                raise ValueError("Shard paths must stay inside the checkpoint directory")
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
    intervals = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        dtype = spec["dtype"]
        if dtype not in ("F32", "F16"):
            raise ValueError(f"Unsupported tensor {name} dtype {dtype}; expected F32 or F16")
        shape = tuple(spec["shape"])
        begin, end = spec["data_offsets"]
        if not 1 <= len(shape) <= 4 or any(type(d) is not int for d in shape):
            raise ValueError(f"Invalid shape for {name}: {shape}")
        if type(begin) is not int or type(end) is not int:
            raise ValueError(f"Invalid data offsets for {name}")
        element_size = 4 if dtype == "F32" else 2
        expected = element_size
        for dim in shape:
            if dim <= 0:
                raise ValueError(f"Invalid shape for {name}: {shape}")
            expected *= dim
        if end - begin != expected or begin < 0 or data_start + end > file_size:
            raise ValueError(f"Invalid data offsets for {name}")
        intervals.append((begin, end))
        yield Tensor(name, path, data_start + begin, expected, shape,
                     GGUF_F32 if dtype == "F32" else GGUF_F16)
    previous = 0
    for begin, end in sorted(intervals):
        if begin != previous:
            raise ValueError(f"Overlapping or non-contiguous tensor data in {path}")
        previous = end
    if data_start + previous != file_size:
        raise ValueError(f"Unindexed tensor data in {path}")


@lru_cache(maxsize=1)
def unicode_tables():
    """Freeze Python's word splitting and full lowercasing semantics in the GGUF."""
    lower_from, lower_to, ranges = [], [], []
    start, previous = 0, 0
    for cp in range(0x110000):
        ch = chr(cp)
        lowered = ch.lower()
        if lowered != ch:
            lower_from.append(cp)
            lower_to.append(lowered)
        cased = ch.lower() != ch.upper()
        # These contexts distinguish Case_Ignorable, including cased combining marks.
        ignorable = ("A\u03a3" + ch).lower()[1] == "\u03c2" and (
            "A\u03a3" + ch + "A").lower()[1] == "\u03c3"
        flags = int(ch.isalnum() or ch == "_") | (int(ch.isspace()) << 1)
        flags |= int(cased) << 2 | int(ignorable) << 3
        if flags != previous:
            if previous:
                ranges.extend((start, cp - 1, previous))
            start, previous = cp, flags
    if previous:
        ranges.extend((start, 0x10ffff, previous))
    return lower_from, lower_to, ranges


def validate_encoder(encoder):
    for key in ("hidden_size", "num_attention_heads", "num_hidden_layers",
                "intermediate_size", "max_position_embeddings", "vocab_size"):
        if type(encoder.get(key)) is not int or not 0 < encoder[key] < 2**31:
            raise ValueError(f"Invalid encoder {key}")
    hidden = encoder["hidden_size"]
    if hidden % encoder["num_attention_heads"]:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    required = {
        "hidden_act": "gelu", "position_biased_input": False,
        "type_vocab_size": 0, "relative_attention": True, "share_att_key": True,
        "norm_rel_ebd": "layer_norm",
    }
    for key, value in required.items():
        if encoder.get(key) != value:
            raise ValueError(f"Unsupported encoder {key}: expected {value!r}")
    if sorted(encoder.get("pos_att_type", [])) != ["c2p", "p2c"]:
        raise ValueError("Expected c2p and p2c position attention")
    if encoder.get("conv_kernel_size", 0) or encoder.get("embedding_size", hidden) != hidden:
        raise ValueError("Convolution and projected embeddings are not supported")
    if encoder.get("attention_head_size", hidden // encoder["num_attention_heads"]) != (
            hidden // encoder["num_attention_heads"]):
        raise ValueError("Unsupported attention_head_size")
    buckets = encoder.get("position_buckets", -1)
    relative = encoder.get("max_relative_positions", -1)
    relative = relative if relative > 0 else encoder["max_position_embeddings"]
    if type(buckets) is not int or buckets < 4 or buckets % 2 or relative <= buckets // 2 + 1:
        raise ValueError("Invalid relative position buckets")
    eps = encoder.get("layer_norm_eps", 1e-7)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid layer_norm_eps")
    return relative


def encoder_shapes(encoder):
    h, f = encoder["hidden_size"], encoder["intermediate_size"]
    shapes = {
        "encoder.embeddings.word_embeddings.weight": (encoder["vocab_size"], h),
        "encoder.encoder.rel_embeddings.weight": (2 * encoder["position_buckets"], h),
    }
    for prefix in ("encoder.embeddings.LayerNorm", "encoder.encoder.LayerNorm"):
        shapes[prefix + ".weight"] = (h,)
        shapes[prefix + ".bias"] = (h,)
    for i in range(encoder["num_hidden_layers"]):
        prefix = f"encoder.encoder.layer.{i}."
        for name, width, height in (
            ("attention.self.query_proj", h, h), ("attention.self.key_proj", h, h),
            ("attention.self.value_proj", h, h), ("attention.output.dense", h, h),
            ("intermediate.dense", h, f), ("output.dense", f, h),
        ):
            shapes[prefix + name + ".weight"] = (height, width)
            shapes[prefix + name + ".bias"] = (height,)
        for name in ("attention.output.LayerNorm", "output.LayerNorm"):
            shapes[prefix + name + ".weight"] = (h,)
            shapes[prefix + name + ".bias"] = (h,)
    return shapes


def span_shapes(hidden):
    shapes = {}

    def linear(name, width, height):
        shapes[name + ".weight"] = (height, width)
        shapes[name + ".bias"] = (height,)

    for name in ("project_start", "project_end", "out_project"):
        prefix = "span_rep.span_rep_layer." + name
        linear(prefix + ".0", hidden * (2 if name == "out_project" else 1), 4 * hidden)
        linear(prefix + ".3", 4 * hidden, hidden)
    linear("count_pred.0", hidden, 2 * hidden)
    linear("count_pred.2", 2 * hidden, 20)
    shapes["count_embed.pos_embedding.weight"] = (20, hidden)
    for side in ("ih", "hh"):
        shapes[f"count_embed.gru.weight_{side}_l0"] = (3 * hidden, hidden)
        shapes[f"count_embed.gru.bias_{side}_l0"] = (3 * hidden,)
    linear("count_embed.projector.0", 2 * hidden, 4 * hidden)
    linear("count_embed.projector.2", 4 * hidden, hidden)
    return shapes


def metadata(directory):
    model_config = read_json(directory / "config.json")
    architecture = model_config.get("architecture")
    if architecture not in ("span", "boundary"):
        raise ValueError("Only GLiNER2 span and boundary checkpoints are supported")
    output_index = 2
    if architecture == "boundary":
        if model_config.get("architecture_version") != 1:
            raise ValueError("Only boundary architecture version 1 is supported for classification")
        boundary_head = model_config.get("boundary_head")
        if not isinstance(boundary_head, dict):
            raise ValueError("Missing boundary_head configuration")
        dropout = boundary_head.get("dropout", 0.1)
        if not isinstance(dropout, (float, int)) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("Invalid boundary classifier dropout")
        # Dropout is inactive at inference but changes the saved Sequential indices.
        output_index = 3 if dropout > 0 else 2
    encoder_path = directory / "encoder_config" / "config.json"
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing encoder config: {encoder_path}")
    encoder = read_json(encoder_path)
    if encoder.get("model_type") != "deberta-v2":
        raise ValueError("Only DeBERTa-v2/v3 encoders are supported")
    relative = validate_encoder(encoder)
    if model_config.get("token_pooling", "first") != "first":
        raise ValueError("Only first-subword pooling is supported")

    tokenizer = read_json(directory / "tokenizer.json")
    if tokenizer.get("model", {}).get("type") != "Unigram":
        raise ValueError("Only SentencePiece Unigram tokenizer.json files are supported")
    normalizer = {"type": "Sequence", "normalizers": [
        {"type": "Replace", "pattern": {"Regex": r"\s{2,}|[\n\r\t]"}, "content": " "},
        {"type": "NFC"}, {"type": "Strip", "strip_left": False, "strip_right": True}]}
    pretokenizer = {"type": "Sequence", "pretokenizers": [
        {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "always", "split": True}]}
    if tokenizer.get("normalizer") != normalizer or tokenizer.get("pre_tokenizer") != pretokenizer:
        raise ValueError("Unsupported tokenizer normalizer or pretokenizer; expected Decide NFC/Metaspace")
    if tokenizer["model"].get("byte_fallback", False):
        raise ValueError("Unigram byte fallback is not supported")
    vocab = tokenizer["model"]["vocab"]
    tokens = [item[0] for item in vocab]
    scores = [float(item[1]) for item in vocab]
    if not tokens or len(set(tokens)) != len(tokens) or any(not t for t in tokens):
        raise ValueError("Unigram vocabulary must be nonempty and unique")
    if any(not math.isfinite(s) for s in scores):
        raise ValueError("Unigram scores must be finite")
    unk_id = tokenizer["model"].get("unk_id")
    if type(unk_id) is not int or not 0 <= unk_id < len(tokens):
        raise ValueError("Invalid Unigram unk_id")
    added_ids, added_tokens = [], []
    for item in tokenizer.get("added_tokens", []):
        token_id = int(item["id"])
        if any(item.get(k, False) for k in ("single_word", "lstrip", "rstrip")):
            raise ValueError("Unsupported added-token matching flags")
        if item.get("normalized", False) and item["content"] != tokens[unk_id]:
            raise ValueError("Only the unknown token may have normalized matching")
        if token_id < 0 or token_id > len(tokens) or token_id in added_ids:
            raise ValueError(f"Invalid added token ID {token_id}")
        added_ids.append(token_id)
        added_tokens.append(item["content"])
        if token_id < len(tokens):
            if tokens[token_id] != item["content"]:
                raise ValueError(f"Conflicting added token ID {token_id}")
            continue
        tokens.append(item["content"])
        scores.append(0.0)
    if len(tokens) != encoder["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match encoder vocab_size")
    if len(set(tokens)) != len(tokens):
        raise ValueError("Duplicate added token content")
    for marker in ("[P]", "[L]", "[SEP_TEXT]", "[DESCRIPTION]"):
        if marker not in added_tokens:
            raise ValueError(f"Missing structural token {marker}")
    lower_from, lower_to, ranges = unicode_tables()

    kv = [
        ("general.architecture", KV_STRING, "gliner2" if architecture == "boundary" else "gliner2.5-decide"),
        ("general.name", KV_STRING, directory.name),
        ("gliner.architecture", KV_STRING, architecture),
        ("gliner.token_pooling", KV_STRING, model_config.get("token_pooling", "first")),
        ("gliner.format_version", KV_U32, 3 if architecture == "boundary" else 2),
        ("gliner.classifier_output_index", KV_U32, output_index),
        ("deberta.hidden_size", KV_U32, int(encoder["hidden_size"])),
        ("deberta.num_hidden_layers", KV_U32, int(encoder["num_hidden_layers"])),
        ("deberta.num_attention_heads", KV_U32, int(encoder["num_attention_heads"])),
        ("deberta.intermediate_size", KV_U32, int(encoder["intermediate_size"])),
        ("deberta.max_position_embeddings", KV_U32, int(encoder["max_position_embeddings"])),
        ("deberta.vocab_size", KV_U32, encoder["vocab_size"]),
        ("deberta.position_buckets", KV_U32, encoder["position_buckets"]),
        ("deberta.max_relative_positions", KV_U32, relative),
        ("deberta.layer_norm_eps", KV_F32, float(encoder.get("layer_norm_eps", 1e-7))),
        ("deberta.variant", KV_STRING, "deberta-v3" if architecture == "boundary" else "decide-v1"),
        ("tokenizer.ggml.model", KV_STRING, "unigram"),
        ("tokenizer.ggml.tokens", KV_ARRAY, (KV_STRING, tokens)),
        ("tokenizer.ggml.scores", KV_ARRAY, (KV_F32, scores)),
        ("tokenizer.gliner.scores", KV_ARRAY, (KV_F64, [float(v[1]) for v in vocab])),
        ("tokenizer.gliner.unknown_id", KV_U32, unk_id),
        ("tokenizer.gliner.added_ids", KV_ARRAY, (KV_U32, added_ids)),
        ("tokenizer.gliner.added_tokens", KV_ARRAY, (KV_STRING, added_tokens)),
        ("tokenizer.gliner.pipeline", KV_STRING, "decide-nfc-v1"),
        ("tokenizer.gliner.unicode_version", KV_STRING, unicodedata.unidata_version),
        ("tokenizer.gliner.lower_from", KV_ARRAY, (KV_U32, lower_from)),
        ("tokenizer.gliner.lower_to", KV_ARRAY, (KV_STRING, lower_to)),
        ("tokenizer.gliner.unicode_ranges", KV_ARRAY, (KV_U32, ranges)),
    ]
    h = encoder["hidden_size"]
    shapes = encoder_shapes(encoder)
    shapes.update({"classifier.0.weight": (2 * h, h), "classifier.0.bias": (2 * h,),
                   f"classifier.{output_index}.weight": (1, 2 * h), f"classifier.{output_index}.bias": (1,)})
    if architecture == "boundary":
        kv.append(("gliner.capabilities", KV_STRING, "classification"))
    elif "span_head" in model_config:
        head = model_config["span_head"]
        width = head.get("max_width")
        if head.get("span_mode") != "markerV0" or model_config.get("counting_layer") != "count_lstm":
            raise ValueError("Span extraction supports markerV0 with count_lstm only")
        if type(width) is not int or not 1 <= width <= 128 or model_config.get("max_width") != width:
            raise ValueError("Invalid or inconsistent span max_width")
        if "[E]" not in added_tokens:
            raise ValueError("Missing structural token [E]")
        kv += [("gliner.span_mode", KV_STRING, "markerV0"),
               ("gliner.max_width", KV_U32, width),
               ("gliner.counting_layer", KV_STRING, "count_lstm")]
        shapes.update(span_shapes(h))
    return kv, shapes


def write_kv(file, key, kind, value):
    file.write(gguf_string(key))
    file.write(u32(kind))
    if kind == KV_STRING:
        file.write(gguf_string(value))
    elif kind == KV_U32:
        file.write(u32(value))
    elif kind == KV_F32:
        file.write(struct.pack("<f", value))
    elif kind == KV_ARRAY:
        item_kind, items = value
        file.write(u32(item_kind))
        file.write(u64(len(items)))
        for item in items:
            if item_kind == KV_STRING:
                file.write(gguf_string(item))
            elif item_kind == KV_F32:
                file.write(struct.pack("<f", item))
            elif item_kind == KV_F64:
                file.write(struct.pack("<d", item))
            elif item_kind == KV_U32:
                file.write(u32(item))
            else:
                raise ValueError(f"Unsupported array kind {item_kind}")
    else:
        raise ValueError(f"Unsupported metadata kind {kind}")


def convert(directory, output):
    kv, required_shapes = metadata(directory)
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
    by_name = {tensor.name: tensor for tensor in tensors}
    index = directory / "model.safetensors.index.json"
    if index.exists():
        mapping = read_json(index)["weight_map"]
        if set(mapping) != seen or any(
            by_name[name].path.resolve() != (directory / shard).resolve() for name, shard in mapping.items()
        ):
            raise ValueError("Safetensors index does not match shard contents")
    for required, shape in required_shapes.items():
        if required not in by_name:
            raise ValueError(f"Missing required tensor: {required}")
        if by_name[required].shape != shape:
            raise ValueError(f"Invalid shape for {required}: expected {shape}")

    stored_names = {tensor.name: tensor.name for tensor in tensors}
    long_names = [tensor.name for tensor in tensors if len(tensor.name.encode("utf8")) >= 64]
    if long_names:
        if dict((key, value) for key, _, value in kv)["gliner.format_version"] < 3:
            raise ValueError("Tensor names of 64 bytes or more require application format version 3")
        used = set(stored_names.values())
        aliases = []
        for name in long_names:
            alias = "gliner.t." + hashlib.sha256(name.encode("utf8")).hexdigest()[:48]
            if alias in used:
                raise ValueError(f"Tensor name alias collision for {name}")
            used.add(alias)
            stored_names[name] = alias
            aliases.append(alias)
        kv += [("gliner.tensor_name_map.original", KV_ARRAY, (KV_STRING, long_names)),
               ("gliner.tensor_name_map.stored", KV_ARRAY, (KV_STRING, aliases))]

    offset = 0
    with output.open("wb") as target:
        target.write(b"GGUF")
        target.write(u32(3))
        target.write(u64(len(tensors)))
        target.write(u64(len(kv)))
        for key, kind, value in kv:
            write_kv(target, key, kind, value)
        for tensor in tensors:
            target.write(gguf_string(stored_names[tensor.name]))
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
