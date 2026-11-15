# Flash-Next test inputs

Operator-supplied inputs that cases read, never results. Cases find them with
`context.input(name)`.

| File | Used by | Contents |
|---|---|---|
| `chat-workload.txt` | `chat-workload`, `chat-settings`, `chat-pin-budget`, `chat-pin-memory` | One everyday prompt, UTF-8 |
| `stage4-shape-corpus.json` | the kernel lab (`bench_kernel_lab.py`, `kernel_corpus.py`) | Decode kernel shapes and per-step call counts extracted from a recorded production run, JSON |
| `stage4-route-windows.json` | the kernel lab (`bench_kernel_lab.py`, block MoE section) | Expert-overlap windows of 2, 4 and 8 consecutive decode tokens from a recorded route trace, ids relabelled per window, JSON |
