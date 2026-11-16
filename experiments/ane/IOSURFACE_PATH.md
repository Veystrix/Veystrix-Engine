# Avoiding the second allocation: research notes (2026-11-17)

## Q1: why can't Core ML / ANE use INT8 straight from the GPU?

INT8 in Core ML is a **weight-storage format**, not the ANE's activation format.
The ANE multiplies in FP16; INT8 weights are dequantized to FP16 before the
MAC. Core ML's public model inputs/outputs are FP32/INT32 at lower deployment
targets and FP16 only at higher opsets (`iOS16/macOS13+`) — the coremltools
converter confirmed this locally: an FP16 program input was rewritten to FP32
unless the deployment target was raised. There is no public
"INT8-in → INT8-out" ANE matmul entry point for activations. MLX also computes
in BF16, so quantizing activations to INT8 would be a third lossy change with no
matching kernel. Result: INT8 reduces the *weight* copy by 2x (already used in
`ane_nemotron_hook_int8.py`), but the activation crossing is still FP16.

## Q2: alternatives to the Core ML-owned allocation

Apple provides three real single-buffer paths, none reachable from
`coremltools`' Python `predict()`:

1. **IOSurface-backed `MLMultiArray`** — `init(pixelBuffer:shape:)`. Apple's docs:
   "creates an IOSurface backed MLMultiArray that reduces the inference latency
   by avoiding the buffer copy to and from some compute units." Pixel format
   must be `kCVPixelFormatType_OneComponent16Half` (FP16) or
   `kCVPixelFormatType_OneComponent8` (INT8). A Metal `MTLBuffer` can be created
   over the same IOSurface (`newBufferWithBytesNoCopy`), so GPU and ANE can
   share one physical buffer. Verified locally: the pixel buffer was created
   (2688x2048, 5376 bytes/row) and `initWithPixelBuffer_shape_` produced an
   FP16 `MLMultiArray`. Filling the surface needs Swift/ctypes rather than
   PyObjC's base-address mapping (which segfaults in Python).

2. **`MLMultiArray(dataPointer:shape:dataType:strides:deallocator:)` with a real
   deallocator and page-aligned memory.** Earlier segfaults used a non-aligned
   NumPy pointer and `deallocator=None`. Apple's `MLPredictionOptions` docs say
   "use a page-aligned address in MLMultiArray" and show
   `aligned_alloc(vm_page_size, ...)`, with a deallocator block that frees it.
   This keeps caller ownership and avoids a Core ML allocation.

3. **`MLTensor.init(bytesNoCopy:shape:scalarType:deallocator:)`** (newer Core AI
   API) — explicitly "without copying the bytes", Swift-only, pairs with
   `MLComputeEngine` and explicit ANE placement on macOS 27.

Additional lever: **`MLPredictionOptions.outputBackings`** lets the caller
supply the output buffer too (CVPixelBuffer or MLMultiArray), avoiding the
output-side allocation. Verify it was used by comparing object identity.

### Caveats from external reports
- An IOSurface-backed `MLMultiArray` consumed as model input becomes **locked**
  and cannot later be reused as output backing (CoreML-LLM); plan separate
  surfaces per direction.
- Core ML multifunction models wire a **separate weight copy per function**;
  Core AI keeps one. Matters for multiple prefill shapes.
- Cross-device work needs explicit completion-handler ownership: never mutate
  a buffer while the other device may still be reading it.

### Verdict
The single-allocation path exists (IOSurface / `bytesNoCopy` / `outputBackings`)
but requires a **native Swift bridge** in the app — `coremltools` cannot express
it, and MLX's Python buffer is not a safe Core ML backing. This is the concrete
engineering route that would remove the ~21 ms marshalling cost.
