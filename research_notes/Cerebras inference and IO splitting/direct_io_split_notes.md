# Direct search notes — 'split I/O time' methods, 2026-11-17

## The literal 'split I/O' systems (2025-2026) — most relevant to our question
- **CoPilotIO (OSDI'26, usenix.org/system/files/osdi26-chen-guanyi.pdf):** GPUs
  initiate I/O while CPU cores act as I/O co-pilots for completion polling; split
  SQ/CQ, hardware barriers, lock-free barrier table, adaptive CPU-GPU co-polling.
  Reduces I/O-induced stalls up to 55.5%, 50% fewer SMs to saturate PCIe BW,
  MoE 1.44x. Key insight: in MoE, some warps do expert GEMMs while others fetch
  next expert weights; under synchronous model I/O warps STALL and occupy GPU
  resources during NVMe polling, delaying compute warps. OFFLOAD POLLING to CPU
  so I/O warps yield immediately. Relevance to us: our decode overlap was rejected
  (tied/-8.2%), but we never tested moving the WAIT/poll off the critical path
  to a dedicated CPU thread with event signaling. Bit-exact (just scheduling).
  CAVEAT: on M4 the GIL and unified memory differ from CUDA SM polling; our E0
  already showed GPU+SSD overlap with only 10% inflation, so the stall CoPilotIO
  removes may be small here. But the split-poll architecture is a new idea.
- **Weave (arxiv 2609.21483):** first MoE overlap with fine-grained DYNAMIC SM
  scheduling — after routing, layer's comm/compute volumes known; cost model
  inside persistent megakernel splits SMs into comm vs compute workers per layer,
  plus temporal scheduling to hide bubbles. 2.89x MoE-layer, 1.33x end-to-end on
  4xH100. Transferable idea: split compute units between read-servicing and
  expert-GEMM dynamically per layer. On M4 no SMs to split, but the CPU SME/ANE
  could be the 'comm workers' — except they share the memory bus (ANE research).
- **COMET (2609.07536):** launch-time analytical resource manager for MoE
  compute-comm overlap, no profiling. Same family.
- **SwiftEP (NSDI'26):** TMA-offloaded token multicast, overlap large hidden-state
  loads with small scale/index/weight loads via async copy. TMA = hardware async
  copy; M4 has no TMA. Buffer-fusion ideas not applicable to single M4.
- **CoX-MoE (2605.17889) / TriMoE (2603.01058) / DAOP (2501.10375):** CPU-GPU
  co-execution using Intel AMX for some experts; hot/warm/cold placement across
  GPU/CPU/NDP. 1.7-2.4x, 2.1-2.8x, up to 8.2x. Key: use CPU matrix units for
  some experts while GPU does others. On M4 the CPU has SME/AMX-like units but
  they share memory bandwidth with GPU (ANE research: no extra bandwidth), so
  splitting compute across them doesn't split the SSD byte wall — but CoX's
  'coalesced expert execution' (process whole batch per expert, not micro-batch)
  is a bit-exact scheduling idea.

## Meta-conclusion forming
Every real 'split I/O time' method either (a) overlaps I/O with compute on a
separate unit (we proved this works but caps at +25% in E1 because bytes unchanged),
or (b) splits weight storage across a faster tier (Cerebras SRAM / Groq — needs
hardware we cannot buy), or (c) amortizes reads across tokens via batching/
speculation (our K=2 sharing loses on Metal; E1 caps it). None creates new SSD
bandwidth or removes bytes on a single 16 GB M4. The research streams will confirm.
