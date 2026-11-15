# Stage 10 production integration

The default HTTP server uses one inference thread and one live conversation. At admission it drops expired or disconnected tickets before invoking the model and releases the queued payload. The bounded queue continues to reject excess requests with HTTP 503. Streaming drains on the handler thread. A shared, rate-limited disconnect probe serves both the decode callback and model cancellation checks.

Flash-Next now checks cancellation at streamed MoE layer boundaries during speculative verification. An interrupted speculative turn closes its generator, clears the route observer, and discards target and draft caches so a subsequent request can replay the committed transcript. Nemotron closes its speculative generator and clears draft history along with target state. Singleton Nemotron decode finishes the forward for the last emitted token before observing cancellation. The server resets the session on a failed response.

The universal cache retains separate namespaces for Flash-Next and Nemotron state. The prefix compatibility records now identify the speculative implementation, settings and target-only snapshot format. Existing prefix entries under older keys will be cold once. Named session compatibility is kept separate. No draft state is read from a target-only snapshot. The content-addressed artifact tier still holds opaque immutable files. Token IDs are keyed by tokenizer content and encoding semantics. A replaced tokenizer or an invalid token blob causes a cache miss.

A guarded real Nemotron BF16 checkpoint run restored 487 of 512 prompt IDs. Its 128-token digest matched the cold run. TTFT was 9.36 s cold and 2.59 s with the prefix. The two decode rates were 6.50 and 6.45 tok/s. See [Stage 11](stage-11.md) for the raw artifacts.

## Decode-first server

Set `VEYSTRIX_CONTINUOUS_BATCHING=1` when launching the server with greedy sampling. Flash-Next also supports `--enable prefill-16gb --pool-gb 0`. The scheduler admits up to four active conversations. Each owns a cache and sampler while sharing the loaded model and expert storage. The model thread advances a conversation by at most 128 prefill tokens at a time. It chooses a ready decoder first and ages queued prefills so they eventually progress. It refuses another conversation when the combined state reservation would cross the host headroom or the 4 GiB live-state cap.

Nemotron groups ready decode rows from the same model in a single layer-major pass. Each row keeps its own recurrent state, KV and masks. The pass keeps routed experts alive across rows of the same layer. Flash-Next co-schedules independent singleton decode steps. Both paths preserve the system-prefix boundary used by their own chunk plan and scope its snapshots to that plan. The server uses the dedicated path for sampled and speculative configurations.

Checkpoint-free tests exercise HTTP state isolation, decode-first order, grouped cancellation and memory refusal. The Nemotron tiny-checkpoint test matches singleton output tokens and all cache tensors at two different offsets. Real concurrency 1 overhead and throughput at concurrency 2 and 4 remain unqualified. Qwen layer-local decode reuse is still open.
