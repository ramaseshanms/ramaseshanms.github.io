# What I Learned Building a Vulkan LLM Runtime for Android From Scratch

*M S Ramaseshan | March 2026*

---

I spent several weeks building a from-scratch Vulkan compute LLM inference engine for Android. I wrote 9 quantization shaders, a thermal governor, a paged KV cache, a full Android SDK — and ended up 10x slower than llama.cpp running on the CPU of the same phone.

This is what I learned.

---

## The Hypothesis

The idea was simple: mobile GPUs have massive parallel compute. A Snapdragon 7s Gen 3's Adreno 810 has 51 GB/s of memory bandwidth and hundreds of ALUs. LLM decode is a memory-bandwidth-bound GEMV (matrix-vector multiply) — one token at a time, reading the entire weight matrix. GPU should crush this.

llama.cpp runs on CPU. It uses ARM NEON SIMD intrinsics. On a mid-range phone, it gets about 6-20 tok/s depending on the model and quantization format. Respectable, but CPU cores top out at maybe 15 GB/s of practical bandwidth across 4-8 threads. The GPU has more bandwidth, more parallelism, and a wider memory bus.

So I built a Vulkan 1.1 compute pipeline to run LLMs entirely on the mobile GPU.

Why Vulkan 1.1? Because 80%+ of Android phones in the mid-range segment only support Vulkan 1.1. llama.cpp's Vulkan backend requires 1.2. If you want GPU acceleration on a $200 Android phone, Vulkan 1.1 is what you get.

## What I Built

The project (Viking) is a C++20 Vulkan compute runtime with a Kotlin Android SDK. The core components:

**Inference engine.** A Llama-family forward pass (`LlamaForward`) that dispatches quantized GEMV operations to the GPU via Vulkan compute shaders. RMSNorm, RoPE, SiLU, GQA attention, and residual connections all have GPU shader implementations. CPU fallback for any operation the GPU can't handle.

**9 quantization format shaders.** Each format needed a dedicated GLSL compute shader that understands the bit-level block layout:

| Format | Block Size | Bits/Weight | Technique |
|--------|-----------|-------------|-----------|
| Q4_0   | 18B / 32 elem  | 4-bit symmetric  | Nibble extraction + fp16 scale |
| Q4_1   | 20B / 32 elem  | 4-bit asymmetric | + fp16 minimum offset |
| Q8_0   | 34B / 32 elem  | 8-bit symmetric  | Direct int8 + fp16 scale |
| Q4_K   | 144B / 256 elem | 4-bit K-quant   | 6-bit packed scales, sub-block mins |
| Q5_K   | 176B / 256 elem | 5-bit K-quant   | Q4_K + 5th bit from qh array |
| Q6_K   | 210B / 256 elem | 6-bit K-quant   | Split ql/qh reconstruction |
| IQ3_S  | 110B / 256 elem | 3-bit importance | 512-entry codebook + sign bits |
| IQ4_XS | 136B / 256 elem | 4-bit importance | 16-entry non-linear lookup |

These are GGUF-compatible — the same model files that llama.cpp uses work directly. Getting the bit manipulation right for K-quant formats is non-trivial. Q4_K has sub-blocks 0-3 and 4-7 that use different scale extraction formulas. Q5_K has a separate `qh[32]` byte array where each bit is the 5th bit of a weight value. IQ3_S uses a 512-entry grid codebook where the 9th index bit comes from a separate high-bit array. You don't get these right by guessing — you read the ggml source, trace through the bit operations, and verify with golden-value tests.

**Safety stack.** A thermal governor that polls device temperature every 500ms and throttles inference across 6 performance tiers. A memory pressure handler that integrates with Android's `onTrimMemory`. A battery monitor. A GPU utilization cap at 80%. This is the part that llama.cpp doesn't have, and it matters — but I'll come back to that.

**Android SDK.** Kotlin API with coroutine-based streaming (`Flow<GenerationEvent>`), lifecycle-aware sessions, and a JNI bridge.

## The Wall: 0.50 Tokens Per Second

First benchmark. POCO M2 Pro — a Snapdragon 665 with an Adreno 610 GPU. TinyLlama 1.1B, Q4_0 quantization.

```
Viking GPU:   0.50 tok/s    TTFT: 18,530ms
llama.cpp CPU: 6.00 tok/s    TTFT: 305ms
```

Twelve times slower. Time to first token: one minute versus a third of a second.

My immediate reaction was that the shaders were slow. The Q4_0 shader was reading weights byte-by-byte — each 4-bit nibble required two shift operations and a mask. Surely vectorizing to uint32 reads would fix this.

But before optimizing blindly, I instrumented the pipeline. Added per-dispatch and per-fence timing counters to the submit loop.

## Bottleneck Decomposition: The 6ms Fence

The profiling data told a different story:

```
dispatches: 2161    fences: 1237    tokens: 14
dispatches/token: 154    fences/token: 88
avg fence time: 19.98ms
total fence time: 24,715ms
```

Each Vulkan dispatch follows this sequence:

```
vkResetCommandBuffer()      ~0.01ms
vkBeginCommandBuffer()      ~0.01ms
vkCmdBindPipeline()         ~0.01ms
vkCmdBindDescriptorSets()   ~0.01ms
vkCmdPushConstants()        ~0.01ms
vkCmdDispatch()             ~0.01ms
vkEndCommandBuffer()        ~0.01ms
vkQueueSubmit()             ~1-2ms   ← driver overhead
vkWaitForFences()           ~10-12ms ← GPU pipeline drain
```

Total per-dispatch overhead: **~13ms**, dominated by the submit-and-wait roundtrip.

Each transformer layer needs 7 GEMV operations (Q, K, V, O projections + gate, up, down FFN). TinyLlama has 22 layers. Plus one final lm_head projection.

**22 layers x 7 GEMVs + 1 lm_head = 155 dispatches per token.**

**155 dispatches x 13ms = 2,015ms per token.**

**Predicted: 0.50 tok/s. Measured: 0.49 tok/s.**

The prediction matched the measurement exactly. The GPU wasn't slow — it was barely doing any work. The bottleneck was the Vulkan driver's command submission overhead. Each time you submit a command buffer and wait for a fence, Qualcomm's Vulkan driver takes roughly 6ms just for the roundtrip, regardless of how trivial the GPU work is. Then you add the actual GPU compute on top.

This isn't a hardware limitation. It's a driver design choice. Vulkan on mobile is optimized for graphics workloads: a few large draw calls per frame, 60 times per second. Not 155 tiny compute dispatches per token, hundreds of times per second.

## The Optimization Journey

What followed was a systematic attempt to close the gap by attacking every bottleneck I could find.

### Attempt 1: Batch Command Buffers (Result: +2%)

Hypothesis: record multiple GEMV dispatches into a single command buffer, share one fence across them. QKV projections are independent (same input, different weight matrices, different outputs), so batch them 3-into-1. Similarly batch gate+up 2-into-1.

```
                 Dispatches/tok  Fences/tok  Avg Fence  Decode
Baseline         154            154         ~13ms      0.50 tok/s
Batched QKV+gate 154            88          ~20ms      0.51 tok/s
```

Fences dropped from 154 to 88. But the average fence time increased from 13ms to 20ms. Three dispatches in one fence means the GPU runs all three sequentially — the fence wait includes all the GPU compute. Net improvement: 2%.

The bottleneck traded one form for another.

### Attempt 2: Fused Weight Tensors (Result: -2%)

Hypothesis: concatenate QKV weight matrices at init time, dispatch one large GEMV instead of three. Reduces dispatch count from 155 to 89.

```
                 Dispatches/tok  Fences/tok  Avg Fence  Decode
Baseline         154            154         ~13ms      0.50 tok/s
Fused weights    88             88          ~21ms      0.49 tok/s
```

Same story. Fewer dispatches, but each dispatch does 2-3x more work, so each fence wait is longer. The byte-level shader is bandwidth-limited — doubling the output dimension doubles the memory reads. No net gain.

### Attempt 3: Remove Pipeline Barriers (Result: -15%)

Hypothesis: barriers between independent dispatches in a batch are unnecessary overhead.

```
                  Avg Fence  Decode
With barriers     20.0ms     0.51 tok/s
Without barriers  23.6ms     0.44 tok/s
```

Performance got *worse*. On Adreno 610, pipeline barriers between dispatches help the GPU flush L1 caches. Without them, concurrent dispatches thrash each other's cache lines. This is Adreno-specific — the tiled architecture needs explicit synchronization points.

### Attempt 4: Shader Vectorization (Result: +3.6x)

This was the first real win. Replaced byte-by-byte weight reads with uint32 bulk reads across all shaders:

```glsl
// Before: 1 nibble per read
uint byteVal = wReadByte(nibBase + (li >> 1u));
uint nib = (li & 1u) == 0u ? (byteVal & 0xFu) : (byteVal >> 4u);

// After: 8 nibbles per read
uint word = weights.data[wordIdx];
// Extract 8 nibbles from registers, no memory ops
```

Results:

| Shader | Before | After | Speedup |
|--------|:------:|:-----:|:-------:|
| Q4_0 GEMV  | 0.49 tok/s | **1.78 tok/s** | **3.6x** |
| Q8_0 GEMV  | 0.49 tok/s | **1.24 tok/s** | **2.5x** |
| Q4_K GEMV  | (new)      | **1.85 tok/s** | — |
| Q6_K GEMV  | CPU 0.13   | **1.85 tok/s** | **14x** |

Average fence time dropped from 27ms to 6.4ms. This confirmed that on the vectorized shaders, GPU compute had been the dominant bottleneck — not dispatch overhead. Both bottlenecks are real; which one dominates depends on the shader's efficiency.

### Attempt 5: New Device — Nothing Phone 3a Pro / Adreno 810

Moved to a newer phone. Snapdragon 7s Gen 3, Adreno 810, Vulkan 1.3, 51 GB/s bandwidth.

```
llama.cpp Q4_0:   19.60 tok/s (target to beat)
Viking Q4_0:       1.97 tok/s
Viking Q4_K_M:     2.09 tok/s
```

Still 10x behind. And here's the critical discovery:

**The Adreno 810's Vulkan fence roundtrip is 6.27ms — identical to the Adreno 610.**

Same ~6ms fixed overhead. Same Qualcomm Vulkan driver. The GPU has 3.5x more bandwidth and far more compute, but the driver overhead is the same. This is a software bottleneck, not a hardware one.

### Attempt 6: Full GPU Pipeline — 1 Fence Per Token

The nuclear option. Record ALL 22 transformer layers into a single Vulkan command buffer. 330 dispatches, 1 fence wait per token. Every operation on GPU — RMSNorm, RoPE, SiLU, attention, residuals. No CPU round-trips between layers.

Theoretical: 12.5ms GPU compute (bandwidth limit) + 6ms fence = 18.5ms = **54 tok/s**.

Reality:

```
GPU compute:     50ms  (tiled GEMV, 2.3x faster with shared-memory reduction)
CPU recording:  400ms  (1,320 Vulkan API calls x 0.3ms each)
Total:          450ms  = 2.2 tok/s
```

The GPU was fast. The CPU was slow. Recording 1,320 Vulkan API calls (`vkCmdBindPipeline`, `vkCmdBindDescriptorSets`, `vkCmdPushConstants`, `vkCmdDispatch`, `vkCmdPipelineBarrier` — 6 calls per dispatch x ~220 dispatches) took 400ms because *each Vulkan API call on Qualcomm's host-side driver costs 0.3ms*. That's 300x slower than the same calls on desktop Vulkan drivers.

I had eliminated the fence bottleneck only to reveal the next one: host-side API call overhead.

### Attempt 7: OpenCL Dispatch Test

At this point I tested OpenCL as an alternative compute API on the same Adreno 810:

```
OpenCL individual dispatch+finish:  0.281ms  (Vulkan: 6.27ms = 22x slower)
OpenCL batch 10 dispatches:         0.055ms/dispatch
OpenCL batch 100 dispatches:        0.007ms/dispatch
OpenCL batch 330 (full token):      3.18ms total
```

OpenCL dispatch overhead on Qualcomm is **22x faster than Vulkan**. The same hardware, the same driver team, but OpenCL's command queue model has fundamentally lower submission overhead than Vulkan's explicit fence model.

Projected with optimized OpenCL GEMV kernels: **18-35 tok/s** — within striking distance of or exceeding llama.cpp.

But the OpenCL GEMV kernel itself was only achieving 1.4 GB/s out of 51 GB/s theoretical (2.7% utilization). The dispatch overhead problem was solved; now the kernel throughput was the bottleneck. Getting efficient memory access patterns on Adreno's tiled architecture requires weight layout transformations and cooperative loading via local memory — real GPU kernel optimization work.

### The Pivot: NEON SDOT

While wrestling with Vulkan and OpenCL, I tried the obvious alternative: ARM NEON intrinsics on CPU. The same approach llama.cpp uses.

**Day 1: Naive NEON.** Scalar nibble extraction with NEON `vfmaq_f32` accumulators. 0.8 GB/s. Terrible — the scalar extraction defeats the SIMD pipeline.

**Day 2: NEON SDOT.** The key insight from ggml: quantize the activation vector to Q8_0, then use `vdotq_s32` (ARMv8.2 SDOT instruction) for the dot product. This does 16 multiply-accumulates per instruction. Unpack Q4_0 nibbles with vectorized bit ops using `vzip1q_s8`/`vzip2q_s8`.

Result: **10.0 tok/s single-threaded.**

**Day 3: Multi-threading.** Row-striped N-dimension parallelism across CPU cores.

Result: **11.2 tok/s with 2 threads.** 4+ threads degraded performance — memory bandwidth saturated at 2 threads on this SoC.

**Day 4: Cache blocking and i8mm.** Tried K-tiling, N-tiling, and ARMv8.6 i8mm compiler flags.

Result: **No improvement.** The kernel is bandwidth-bound, not compute-bound. Tiling within already-cached data adds overhead without reducing memory traffic.

Four days of CPU work got me to 11.2 tok/s. Weeks of GPU work got me to 2.6 tok/s.

The CPU won because it has zero submission overhead. `vdotq_s32` operates directly on memory. No driver, no command buffer, no fence, no API call overhead. For a bandwidth-bound operation like GEMV (one row of output per token, entire weight matrix read), the CPU's direct memory access is a structural advantage.

## The Full Scorecard

Here's where everything landed after all optimizations, tested on Llama 3.2 1B Instruct across all 7 quantization formats:

| Quant | Viking GPU | llama.cpp CPU | Gap |
|-------|:---------:|:------------:|:---:|
| IQ3_M  | 2.45 tok/s | 6.99 tok/s  | 2.9x |
| IQ4_XS | 2.61 tok/s | 11.89 tok/s | 4.6x |
| Q4_K_M | 2.25 tok/s | 11.68 tok/s | 5.2x |
| Q5_K_M | 2.51 tok/s | 10.11 tok/s | 4.0x |
| Q6_K   | 2.46 tok/s | 9.13 tok/s  | 3.7x |

The closest gap is IQ3_M (2.9x) because llama.cpp's NEON codebook lookup for importance-quantized formats is less optimized than its K-quant NEON kernels. The GPU's codebook lookup via shared memory is comparatively more competitive.

The NEON path: 11.2 tok/s on Q4_0, competitive with llama.cpp's 14.35 tok/s on the same device.

## What Production On-Device AI Actually Requires

Benchmarks measure peak throughput. Production measures sustained throughput.

Run llama.cpp at full speed on a phone for 60 seconds. The SoC heats up. The OS thermal daemon kicks in. CPU frequency drops. What started at 14 tok/s drops to 8, then 5, then throttles to 3 tok/s while the phone gets hot enough that users notice.

This is the problem I built Viking's safety stack to solve:

- **Thermal governor**: polls temperature every 500ms, manages 6 performance tiers with hysteresis (3s delay before upgrading tier, immediate downgrade). At the `Throttled` tier, it inserts 10ms delays between tokens. At `Critical`, 20ms. This keeps the SoC temperature under 40C instead of letting it spike to 45C+ and crash.

- **Adaptive token rate**: dynamically adjusts inter-token delays based on thermal trend. If temperature is rising, slow down before the OS forces a harder throttle.

- **GPU utilization cap**: never exceeds 80% GPU utilization at any performance tier. This leaves headroom for the Android compositor and prevents the "my phone is frozen" experience during inference.

- **Memory pressure handler**: integrates with Android's `onTrimMemory` to release GPU buffers progressively instead of getting OOM-killed.

- **Battery monitor**: at 15% battery, drops to Medium tier. At 5%, drops to Minimal. If the phone is charging and GPU temperature exceeds 38C, drops to Cooldown to prevent battery degradation.

None of these are in llama.cpp's Android story. And they matter more than the difference between 12 and 15 tok/s. A user who gets consistent 10 tok/s for a 5-minute conversation has a better experience than a user who gets 15 tok/s for 30 seconds before the phone throttles to 3 tok/s and gets too hot to hold.

I don't have sustained-throughput comparison data to prove this claim with numbers (that's future work), but the infrastructure is real and the physics is straightforward: unmanaged inference on mobile SoCs WILL thermal-throttle, and a governor that prevents the spike-and-crash pattern WILL deliver better sustained performance.

## What I Learned Building With AI

I used AI extensively throughout this project — for code generation, architecture decisions, code review, security auditing. Here's what I learned the hard way.

### You will not catch AI hallucinations until it's too late

At one point, my CHANGELOG claimed "~24 tok/s decode throughput" on a Snapdragon 8 Gen 3. This number appeared in a section written during an AI-assisted session. It was never measured. No benchmark on a Snapdragon 8 Gen 3 exists anywhere in the repository. No hardware branch for that SoC was ever created.

I didn't catch it for weeks. It looked plausible — it was in the right ballpark for what that hardware could theoretically achieve. It had the right format, the right specificity, the right confidence. It was a perfect hallucination.

The lesson: AI-generated content that *looks* like measured data is the most dangerous kind of output. You can't distinguish "I computed this from the benchmark logs" from "I generated a plausible number" without going back to the raw evidence. Every quantitative claim in AI-assisted work needs a traceable source, or it's suspect.

### Current LLM architectures will still fail on novel engineering

The parts of this project that worked best with AI were the parts closest to existing open-source implementations: GGUF parsing (llama.cpp's format, well-documented), quantization block layouts (ggml source code exists), thermal management patterns (Android documentation).

The parts where AI struggled were the parts with no existing dataset to draw from: optimizing Vulkan compute dispatch patterns for Qualcomm's mobile driver, understanding Adreno's tiled memory architecture, predicting that pipeline barriers would *improve* performance due to cache flushing behavior. These required empirical measurement and hardware-specific reasoning that doesn't exist in training data.

When you're building something that doesn't exist yet, AI will generate confident, syntactically correct, architecturally reasonable code that doesn't work. It has no way to know it doesn't work because the feedback loop (run it on this specific GPU, measure this specific metric, observe this specific driver behavior) wasn't in its training set.

### Write tests and measurement infrastructure first

The single biggest process mistake was writing the runtime before writing the benchmark and accuracy infrastructure. I had thousands of lines of shader code before I had a way to measure whether any of it was fast or correct.

When I finally instrumented the pipeline, I discovered in 10 minutes that the bottleneck was fence overhead, not shader performance. If I'd had that instrumentation from day one, I would have skipped a week of shader optimization that didn't matter.

The right order: (1) golden-value correctness tests, (2) performance benchmark harness, (3) accuracy regression tracking, (4) then write the actual code. Every new shader should be born with a test that proves it produces correct output and a benchmark that proves it's faster than what it replaces.

### AI code review is necessary but not sufficient

I ran an AI-driven adversarial security audit on the codebase. It found 69 findings, 6 critical — including a GPU shared memory overflow that would corrupt inference output on any context length over 1024 tokens. This was valuable work that a human might have missed.

But when AI generated the fixes for those 69 findings, the fix batch for 19 MEDIUM findings didn't compile. The AI changed integer counters to `std::atomic` but forgot to use `.load()` and `.store()` at the usage sites. It was committed without being compiled first. A follow-up fix was needed immediately.

More subtly: the path traversal fix (C-03) used `string.find("..")` instead of `realpath()` with a sandbox prefix check. The AI chose the simpler approach that looks correct but fails against symlinks, encoded slashes, and absolute paths. A human reviewer with security experience would have caught this. The AI reviewer that originally *found* the vulnerability didn't catch that its own fix was inadequate.

The pattern: AI is good at finding problems (wide coverage, no fatigue, consistent application of rules). AI is adequate at generating fixes for well-understood patterns. AI is poor at evaluating whether its own fixes are sufficient — it lacks the adversarial mindset to attack its own work.

First round of review: let AI do it. It's faster and more thorough than a human for mechanical checks. Final round of review: must be human. Ideally multiple humans, because each person catches different classes of issues.

### A stray thought on the future of AI-assisted programming

Every programming language lets you write the same thing multiple ways. Python has at least 5 ways to iterate over a list, 3 ways to format strings, and dozens of patterns for the same control flow. Each alternative is a fork in the probability space that an LLM must navigate.

For production-grade code, you don't need 5 ways to iterate. You need one. If a programming language (or a strict subset of one) had fewer valid ways to express the same logic, the embedding space for that language would be smaller. Fewer valid token sequences means less room for the model to wander into plausible-but-wrong territory.

In other words: if you halve the number of logits a programming LLM needs to consider, you reduce the probability mass allocated to "correct-looking but subtly wrong" alternatives. The model becomes more deterministic not by being smarter, but by having fewer ways to be wrong.

This is a half-formed thought, not a rigorous claim. But it points at something worth exploring: languages designed for AI-assisted development, where the syntax is minimal and unambiguous, might get dramatically better AI code generation than today's sprawling, multi-paradigm languages.

## Where This Goes

Viking taught me what I set out to learn: where the real bottlenecks are in mobile GPU LLM inference, and why CPU still wins today.

The answer isn't "GPUs are slow." The Adreno 810's compute and bandwidth are more than sufficient. The answer is that Qualcomm's Vulkan driver has 6ms of fixed overhead per fence and 0.3ms per API call, and LLM inference requires hundreds of dispatches per token. The same hardware with OpenCL dispatch shows 22x lower overhead. The same hardware class with a desktop Vulkan driver would have <0.1ms per fence.

The path forward for mobile GPU LLM inference is one of:
1. **OpenCL on Qualcomm** — 22x lower dispatch overhead, needs kernel throughput optimization
2. **Vulkan with pre-recorded command buffers** — amortize the 0.3ms/call recording cost by reusing command buffers across tokens (positions passed via UBO instead of push constants)
3. **Wait for Qualcomm to fix their Vulkan driver** — they know about this; it's an active area of work

Or: accept that CPU NEON is the right answer for single-token decode on current mobile hardware, and use GPU for batch prefill and speculative decoding where parallelism actually helps.

The thermal and safety infrastructure is the real differentiator for production mobile AI. Everyone is racing on tok/s benchmarks. Nobody is measuring what happens after 60 seconds of sustained inference. That's the gap worth closing.

---

*The full codebase, including all benchmark data and commit history, is at [github.com/MadrasAI/Viking](https://github.com/MadrasAI/Viking).*

*All performance numbers in this post are from on-device measurements with raw output preserved in the repository. Nothing is projected or estimated unless explicitly stated.*
