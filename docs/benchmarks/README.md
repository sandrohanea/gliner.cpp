# Benchmark results

Reports contain measurements supplied from the named device, not performance guarantees or cross-device rankings. Benchmark usage and timing definitions are in the [main README](../../README.md#performance-benchmark).

| Date | Platform | Comparisons |
|---|---|---|
| 2026-09-25 | [Windows / Core i7-13800H / RTX 4060 Laptop](2026-09-25-windows-rtx4060.md) | C++ and Python, CPU and CUDA, joint and separate questions |

## Adding a device report

Add a separate dated file such as `YYYY-MM-DD-macos-<device>.md` for Metal, then add its link to this index. Keep device reports independent so they can be published from different machines without editing another device's results.

Include:

- Hardware, OS, driver/runtime/compiler versions, model revision and precision; mark unrecorded details explicitly.
- Exact text, ordered tasks/labels, prompts/descriptions, limits, thread count, warmups and measured iterations.
- Backend synchronization and precision settings, including any TF32 or mixed-precision changes.
- Mean, median and p95 latency, request/question throughput, actual token counts, load time and first-request time.
- Correctness evidence for the measured workload and any separate parity suite; do not treat matching final logits as proof of every intermediate layer.
- Reproduction commands, source revisions when captured, measurement caveats and raw JSON logs when available.

One request answers all questions. Compare the same mode and workload across implementations. Joint and separate questions are different contextual inputs, so a timing ratio does not establish equivalent classification behavior. Do not compare model-load timings across different loading strategies as if they measured identical startup work.
