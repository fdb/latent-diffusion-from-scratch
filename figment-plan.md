# Figment: WebGL to WebGPU Migration Plan

## Current Architecture

Figment is a visual programming environment for creative coding. The rendering stack currently works as follows:

- **WebGL 1.0 via TWGL** is used for all rendering. The `Framebuffer` class wraps a TWGL framebuffer object (`twgl.createFramebufferInfo`) with RGBA attachments.
- **Nodes pass `Framebuffer` objects** between each other via image ports. A `Framebuffer` exposes a `.texture` getter (the WebGL texture attachment) and `.bind()` / `.unbind()` methods for rendering into it.
- **Helper functions** like `drawQuad()` render fullscreen quads with shader programs, and `framebufferToImageData()` reads pixels back to CPU via `gl.readPixels`.
- **WebGPU is only used inside ONNX Runtime (ORT)** for ML inference nodes. After creating an ORT session with the `webgpu` execution provider, the node obtains a `GPUDevice` from `ort.env.webgpu.device`.

## The Core Bottleneck

ML nodes (e.g., `onnxImageModel.js`) must shuttle pixel data between WebGL and WebGPU through the CPU:

1. **WebGL -> CPU**: `gl.readPixels()` copies the input framebuffer's texture into a `Uint8Array`.
2. **CPU -> WebGPU**: `device.queue.writeBuffer()` uploads that array to a GPU storage buffer.
3. ORT runs inference entirely on the GPU.
4. **WebGPU -> CPU**: `stagingBuffer.mapAsync()` + `getMappedRange()` copies the result back to a `Uint8Array`.
5. **CPU -> WebGL**: `gl.texImage2D()` uploads the array into the output framebuffer's WebGL texture.

Steps 1-2 and 4-5 are pure waste -- the data is already on the GPU but must round-trip through main memory because WebGL and WebGPU cannot share textures. This adds latency and blocks the main thread on synchronous readback.

## Migration Strategy

The migration should be incremental. The goal is not to rewrite everything at once, but to introduce WebGPU as the primary GPU API while keeping WebGL functional for nodes that have not yet been ported.

Key principles:

- **WebGPU becomes the primary rendering path.** New nodes use WebGPU compute or render pipelines.
- **WebGL compatibility layer remains** for existing nodes until they are ported. The `Framebuffer` class supports both backends transparently.
- **A single shared `GPUDevice`** is used across all of Figment, including ORT.
- **Zero-copy data flow** between WebGPU-native nodes (ML, compute shaders, canvas output) without CPU staging.

## Framebuffer Dual Representation

The `Framebuffer` class gains a `.gpuTexture` property alongside the existing `.texture` (WebGL):

- **`framebuffer.texture`** -- the existing WebGL texture (TWGL attachment). Created lazily or on demand.
- **`framebuffer.gpuTexture`** -- a `GPUTexture` with `TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC | COPY_DST | RENDER_ATTACHMENT` usage. Created lazily on first access.
- **Dirty flags** track which representation is current:
  - `_webglDirty` -- set when a WebGPU node writes to `gpuTexture`, meaning the WebGL texture is stale.
  - `_webgpuDirty` -- set when a WebGL node renders into the framebuffer, meaning the `gpuTexture` is stale.
- **Lazy sync** happens on access. When a WebGL node reads `.texture` and `_webglDirty` is set, the system copies `gpuTexture` -> WebGL texture (via `readPixels` from a WebGPU staging buffer, or eventually via `copyExternalImageToTexture` if both share a canvas). The reverse applies for `.gpuTexture`.
- **When both consumer and producer are WebGPU nodes**, no sync is needed. The `gpuTexture` is passed directly. This is the zero-copy fast path.

```
Producer (WebGPU) -> Framebuffer.gpuTexture -> Consumer (WebGPU)   // zero-copy
Producer (WebGL)  -> Framebuffer.texture    -> Consumer (WebGL)    // zero-copy
Producer (WebGL)  -> Framebuffer.texture    -> Consumer (WebGPU)   // sync on .gpuTexture access
Producer (WebGPU) -> Framebuffer.gpuTexture -> Consumer (WebGL)    // sync on .texture access
```

## Shared GPUDevice

All WebGPU usage in Figment must go through a single `GPUDevice`:

- **`figment.gpu.device`** -- the shared device, requested at application startup with `navigator.gpu.requestAdapter()` / `adapter.requestDevice()`.
- **ORT integration**: ORT supports receiving a pre-existing device via `ort.env.webgpu.device = figment.gpu.device` before session creation. This must be set before any `InferenceSession.create()` call. This ensures ORT's internal buffers live on the same device as Figment's textures.
- **`figment.gpu.queue`** -- shorthand for `figment.gpu.device.queue`.
- **Nodes access the device** through the `figment` global (already available in the node scripting environment). No node should call `requestAdapter()` or `requestDevice()` on its own.
- **Fallback**: If WebGPU is unavailable (older browsers), `figment.gpu` is `null` and the system falls back to WebGL-only mode. ML nodes will not function in this case (they already require WebGPU via ORT).

## Phase Plan

### Phase 1: Shared GPUDevice + Framebuffer.gpuTexture

**Goal**: Eliminate the CPU roundtrip for ML nodes.

- Request a `GPUDevice` at Figment startup and expose it as `figment.gpu.device`.
- Set `ort.env.webgpu.device = figment.gpu.device` before any ORT session creation.
- Add `gpuTexture` property and dirty-flag sync to the `Framebuffer` class.
- Update `onnxImageModel.js` and similar ML nodes to:
  - Read input from `framebuffer.gpuTexture` directly (via `copyTextureToBuffer` or bind as texture in compute shader) instead of `gl.readPixels`.
  - Write output to `framebuffer.gpuTexture` directly (via `copyBufferToTexture`) instead of `gl.texImage2D`.
- Non-ML nodes continue using WebGL unchanged. Cross-API sync happens automatically via dirty flags.

**Outcome**: ML inference pipeline is fully zero-copy on the GPU. All other nodes unaffected.

### Phase 2: WebGPU canvas context for final composite

**Goal**: Eliminate the last WebGL-to-canvas blit.

- Create a WebGPU canvas context (`canvas.getContext('webgpu')`) for the main output.
- The final composite node (viewer/output) renders the output `gpuTexture` to the WebGPU canvas using a simple fullscreen-quad render pipeline.
- The WebGL canvas becomes a secondary/hidden context used only for nodes that still need it.
- Implement `drawQuad()` equivalent for WebGPU (render pipeline with vertex/fragment shaders for textured quad).

**Outcome**: If the entire pipeline is WebGPU-native, no data ever touches the CPU.

### Phase 3: Port core nodes to compute shaders

**Goal**: Move the most-used nodes off WebGL.

- Port `blend`, `transform`, `color_adjust`, `blur`, and other high-frequency image processing nodes to WebGPU compute shaders.
- Each ported node reads/writes `framebuffer.gpuTexture` directly.
- Establish patterns and utilities for common operations:
  - A `figment.gpu.createComputePipeline()` helper that handles boilerplate.
  - Standard bind group layouts for single-texture-in / single-texture-out operations.
  - A WGSL snippet library for common color math.
- Nodes that use custom GLSL fragment shaders (user-authored) get a compatibility wrapper that translates simple fragment shaders to WGSL, or run through a WebGL-emulation path.

**Outcome**: The majority of common pipelines run entirely on WebGPU.

### Phase 4: Deprecate WebGL path

**Goal**: Remove the WebGL dependency.

- Remove TWGL dependency.
- Remove the WebGL canvas context and `window.gl` global.
- The `Framebuffer` class drops `.texture` and the dual-representation sync logic; `.gpuTexture` becomes the sole backing store.
- Any remaining nodes that relied on WebGL are either ported or removed.
- Simplify the `Framebuffer` class (no more dirty flags, no lazy sync).

**Outcome**: Single rendering backend. Simpler codebase. No cross-API overhead anywhere.
