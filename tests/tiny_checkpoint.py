"""Deterministic complete Decide-shaped checkpoint; standard library only."""

import json
import math
import random
import struct


def write_safetensors(path, tensors, dtype="F32"):
    header, payload = {}, bytearray()
    for name, (shape, values) in tensors.items():
        data = struct.pack("<" + ("f" if dtype == "F32" else "e") * len(values), *values)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def create_checkpoint(root, hidden=2, layers=1, dtype="F32", compact_tokens=False, spans=False, record_count=1,
                      records=False, boundary=False, classifier_dropout=0.1):
    (root / "encoder_config").mkdir(exist_ok=True)
    config = {
        "model_type": "deberta-v2", "hidden_size": hidden, "num_hidden_layers": layers,
        "num_attention_heads": 2 if hidden > 2 else 1, "intermediate_size": hidden * 2,
        "max_position_embeddings": 128, "position_buckets": 8, "max_relative_positions": 128,
        "relative_attention": True, "share_att_key": True, "norm_rel_ebd": "layer_norm",
        "pos_att_type": ["p2c", "c2p"], "position_biased_input": False, "type_vocab_size": 0,
        "hidden_act": "gelu", "layer_norm_eps": 1e-7, "pad_token_id": 0,
        "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
    }
    pieces = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "\u2581"]
    pieces += list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.!?,:()/@+%")
    pieces += ["\u2581hello", "\u2581world", "hello", "ll", "\u00e9", "\u03c3", "\u03c2", "\u4e2d", "\u6587"]
    if compact_tokens:
        pieces += ["\u2581(", "\u2581)", "\u2581.", "\u2581a"]
    markers = ["[MASK]", "[SEP_STRUCT]", "[SEP_TEXT]", "[P]", "[C]", "[E]", "[R]",
               "[L]", "[EXAMPLE]", "[OUTPUT]", "[DESCRIPTION]"]
    vocab = [[piece, 0.0 if i < 4 else -1.0 - (i % 7) * 0.125] for i, piece in enumerate(pieces)]
    added = [{"id": i, "content": piece, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": i == 3, "special": True} for i, piece in enumerate(pieces[:4])]
    added += [{"id": len(vocab) + i, "content": piece, "single_word": False, "lstrip": False,
               "rstrip": False, "normalized": False, "special": True} for i, piece in enumerate(markers)]
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Replace", "pattern": {"Regex": r"\s{2,}|[\n\r\t]"}, "content": " "},
            {"type": "NFC"}, {"type": "Strip", "strip_left": False, "strip_right": True}]},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Metaspace", "replacement": "\u2581", "prepend_scheme": "always", "split": True}]},
        "post_processor": None, "decoder": None,
        "model": {"type": "Unigram", "unk_id": 3, "vocab": vocab, "byte_fallback": False},
    }
    config["vocab_size"] = len(vocab) + len(markers)
    model_config = {"architecture": "span", "token_pooling": "first"}
    if boundary:
        if spans:
            raise ValueError("Synthetic boundary classification cannot declare a span head")
        model_config.update(architecture="boundary", architecture_version=1,
                            boundary_head={"dropout": classifier_dropout})
    if spans:
        model_config.update(max_width=8, counting_layer="count_lstm",
                            span_head={"span_mode": "markerV0", "max_width": 8})
    (root / "config.json").write_text(json.dumps(model_config), encoding="utf8")
    (root / "encoder_config" / "config.json").write_text(json.dumps(config), encoding="utf8")
    (root / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf8")
    tensors = {}

    def tensor(name, shape, norm=False):
        seed = sum((i + 1) * ord(c) for i, c in enumerate(name)) % 997
        if records:
            # Uncorrelated record fixtures avoid near-tied span scores from sinusoidal weights.
            generator = random.Random(seed)
            values = [(1.0 if norm else 0.0) + generator.uniform(-0.3, 0.3) for _ in range(math.prod(shape))]
        else:
            values = [(1.0 if norm else 0.0) + 0.13 * math.sin(seed + i * 0.71)
                      for i in range(math.prod(shape))]
        tensors[name] = (shape, values)

    tensor("encoder.embeddings.word_embeddings.weight", [config["vocab_size"], hidden])
    tensor("encoder.encoder.rel_embeddings.weight", [2 * config["position_buckets"], hidden])
    for prefix in ("encoder.embeddings.LayerNorm", "encoder.encoder.LayerNorm"):
        tensor(prefix + ".weight", [hidden], norm=True)
        tensor(prefix + ".bias", [hidden])
    for i in range(layers):
        prefix = f"encoder.encoder.layer.{i}."
        for name, width, height in (
            ("attention.self.query_proj", hidden, hidden), ("attention.self.key_proj", hidden, hidden),
            ("attention.self.value_proj", hidden, hidden), ("attention.output.dense", hidden, hidden),
            ("intermediate.dense", hidden, 2 * hidden), ("output.dense", 2 * hidden, hidden),
        ):
            tensor(prefix + name + ".weight", [height, width])
            tensor(prefix + name + ".bias", [height])
        for name in ("attention.output.LayerNorm", "output.LayerNorm"):
            tensor(prefix + name + ".weight", [hidden], norm=True)
            tensor(prefix + name + ".bias", [hidden])
    tensor("classifier.0.weight", [2 * hidden, hidden])
    tensor("classifier.0.bias", [2 * hidden])
    tensor("classifier.2.weight", [1, 2 * hidden])
    tensor("classifier.2.bias", [1])
    if hidden == 2:
        tensors.update({
            "classifier.0.weight": ([4, 2], [1, 0, 0, 1, -1, 0, 0, -1]),
            "classifier.0.bias": ([4], [0, 0, 0, 0]),
            "classifier.2.weight": ([1, 4], [1, 2, -1, -2]),
            "classifier.2.bias": ([1], [0.5]),
        })
    if spans:
        for name in ("project_start", "project_end", "out_project"):
            prefix = "span_rep.span_rep_layer." + name
            tensor(prefix + ".0.weight", [4 * hidden, hidden * (2 if name == "out_project" else 1)])
            tensor(prefix + ".0.bias", [4 * hidden])
            tensor(prefix + ".3.weight", [hidden, 4 * hidden])
            tensor(prefix + ".3.bias", [hidden])
        tensor("count_pred.0.weight", [2 * hidden, hidden])
        tensor("count_pred.0.bias", [2 * hidden])
        tensors["count_pred.2.weight"] = ([20, 2 * hidden], [0.] * (40 * hidden))
        if not 0 <= record_count <= 19:
            raise ValueError("Synthetic record count must be 0..19")
        tensors["count_pred.2.bias"] = ([20], [1. if i == record_count else 0. for i in range(20)])
        tensor("count_embed.pos_embedding.weight", [20, hidden])
        for side in ("ih", "hh"):
            tensor(f"count_embed.gru.weight_{side}_l0", [3 * hidden, hidden])
            tensor(f"count_embed.gru.bias_{side}_l0", [3 * hidden])
        tensor("count_embed.projector.0.weight", [4 * hidden, 2 * hidden])
        tensor("count_embed.projector.0.bias", [4 * hidden])
        tensor("count_embed.projector.2.weight", [hidden, 4 * hidden])
        tensor("count_embed.projector.2.bias", [hidden])
    # Retained but never executed by the classification runtime.
    tensors["unused.span.weight" if spans else "span_rep.unused.weight"] = ([2, 3, 4], list(range(24)))
    if boundary:
        if classifier_dropout > 0:
            for suffix in ("weight", "bias"):
                tensors[f"classifier.3.{suffix}"] = tensors.pop(f"classifier.2.{suffix}")
        tensors["boundary_head." + "long_component_name_" * 4 + ".weight"] = ([2, 3], list(range(6)))
    write_safetensors(root / "model.safetensors", tensors, dtype)
    return tensors
