#!/usr/bin/env python3
"""Export upstream contextual [L] states to compare with the GGML classifier.

Requires `pip install gliner2` and a local GLiNER2.5-Decide checkpoint.
This is a parity tool, not part of the C++ runtime.
"""

import argparse
import sys
from pathlib import Path

import torch
from gliner2 import AutoExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    if args.threads <= 0:
        parser.error("--threads must be positive")
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(args.threads)
    model = AutoExtractor.from_pretrained(args.checkpoint, use_flashdeberta=False)
    model.eval()
    schema = {"classifications": [{"task": args.task, "labels": args.labels}]}
    batch = model.processor.collate_fn_inference([(args.text, schema)], error_policy="raise")
    device = next(model.parameters()).device
    batch = batch.to(device)
    with torch.inference_mode():
        encoded = model.encoder(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask
        ).last_hidden_state
        _, schema_states = model.processor.extract_embeddings_from_batch(
            encoded, batch.input_ids, batch
        )
        labels = [batch.schema_tokens_list[0][0][i + 1]
                  for i, token in enumerate(batch.schema_tokens_list[0][0][:-1])
                  if token == "[L]"]
        states = torch.stack(schema_states[0][0][1:]).float().cpu()
        reference = model.classifier(states.to(device)).squeeze(-1).float().cpu()
    if labels != args.labels or len(states) != len(args.labels):
        raise RuntimeError("Upstream label order differs from requested labels")
    with args.output.open("w", encoding="utf-8") as file:
        for row in states:
            file.write(" ".join(f"{float(value):.9g}" for value in row) + "\n")
    for label, logit in zip(labels, reference):
        print(f"{label}: {float(logit):.7g}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
