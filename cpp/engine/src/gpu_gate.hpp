#pragma once
// CPU-signalled GPU wait (one MTLSharedEvent) on MLX's default GPU stream.
//
// Decode of a streamed MoE layer is serial: route sync -> read the routed expert
// rows -> encode the expert compute and the next layer's mixer and router -> wait.
// With the gate, the expert compute is built and encoded while the rows are still
// being read: arm() puts a GPU-side wait into the stream's open command buffer, so
// every later kernel on that stream starts only after signal() (called once the
// rows have landed). Same ops on the same bytes, so results are identical; only
// the CPU encode time leaves the critical path.
#include <cstdint>

namespace veystrix::engine::gpu_gate {

// VEYSTRIX_GENERIC_GATE=1 (default off until measured).
bool enabled();

// Encode a wait for the next event value on the default GPU stream. Returns the
// value to signal, or 0 when no command buffer is open (the caller then waits for
// its reads on the CPU as before). Every armed value must be signalled.
uint64_t arm();

// Release the GPU wait for `value` (thread-safe; values are signalled in order).
void signal(uint64_t value);

// VEYSTRIX_CPP_GATE_TAIL=N (pipelined decode, both Qwen runtimes; default 0 = off): once at most
// N experts of a layer are still being read, build their compute behind a gate and return, so
// the next layer's mixer and router are encoded while the last rows land.
int tail();

// A gated read failed: the next layer throws (the GPU was released with bad rows).
void mark_failed();
bool failed();

}  // namespace veystrix::engine::gpu_gate
