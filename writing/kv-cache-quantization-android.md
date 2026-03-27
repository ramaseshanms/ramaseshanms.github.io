# Running LLMs on Phones: What Happens When You Compress the Memory That Models Think With

*M S Ramaseshan | March 2026*

---

Large language models have a memory problem. Not the kind where they forget what you said — the kind where they literally run out of RAM.

When a model processes your conversation, it builds a data structure called the KV cache. Every token the model has ever seen in the conversation gets stored as a pair of vectors — one Key, one Value — across every layer of the network. For a 7-billion-parameter model with a 32K context window, this cache alone consumes over 2 GB of memory. The model weights themselves might fit comfortably on your phone. The memory of your conversation will not.

I spent the last few weeks integrating two research-grade quantization methods — HQQ and Google's TurboQuant — into llama.cpp, rebuilding for Android, and running them on a phone. The goal was simple: compress the KV cache aggressively enough that a 1.7B model can hold a long conversation on a device with 3.4 GB of available RAM, and measure what that compression costs in speed and quality.

This is what I found.

---

## Why the KV Cache Is the Real Bottleneck

If you've followed the discourse around running LLMs on edge devices, you've heard a lot about weight quantization — shrinking the model from 16-bit floats to 4-bit integers. This is a solved problem. Tools like GGUF, GPTQ, and AWQ can compress a 7B model from 14 GB to under 4 GB with minimal quality loss. Your phone can load these models.

But weight quantization is only half the story. Here is where the memory actually goes during a conversation:

| Component | Memory (7B model, 8K context) | Memory (7B model, 32K context) |
|-----------|-------------------------------|--------------------------------|
| Model weights (Q4_0) | 3.8 GB | 3.8 GB |
| KV cache (FP16) | 512 MB | **2,048 MB** |
| Activations + overhead | ~200 MB | ~200 MB |
| **Total** | **4.5 GB** | **6.0 GB** |

The weights are fixed. The KV cache grows linearly with every token in your conversation. At FP16 precision, each token costs about 64 KB of KV memory per layer for a model with 32 KV heads and 128-dimensional head embeddings. Multiply that across 32 layers and a 32K context, and you have consumed 2 GB of RAM that your phone does not have.

This is why your "on-device AI" conversation starts getting choppy or crashes after a few thousand tokens. The model fits. The conversation doesn't.

KV cache quantization attacks this directly: instead of storing each cached Key and Value vector in 16-bit floating point, compress them to 4, 3, or even 2 bits per element. The model weights stay untouched — you're only compressing the memory the model uses to remember your conversation.

The question is how much accuracy you lose, and how much speed you gain or sacrifice.

---

## Two Approaches to Compressing What the Model Remembers

I chose two methods that represent different points on the compression-quality spectrum, and a third technique that improves both.

### HQQ: The Calibration-Free Optimizer

HQQ — Half-Quadratic Quantization — was developed by Hicham Badri and Appu Shaji. The core idea is elegant: instead of using simple min-max scaling to map floating point values into integer buckets (which is what standard quantization does), HQQ frames quantization as an optimization problem and solves it using a proximal operator from convex optimization theory.

In practice, this means HQQ finds better scale and zero-point parameters for each block of weights without ever looking at calibration data. Standard quantization computes `scale = (max - min) / 15` for 4-bit and calls it a day. HQQ iteratively refines the scale and zero-point to minimize the actual reconstruction error across the block.

The block structure is simple:

```
Q4_HQQ block (group size 32):
  - fp16 scale    [2 bytes]
  - fp16 zero     [2 bytes]
  - 32 x 4-bit    [16 bytes packed]
  Total: 20 bytes per 32 elements = 5.0 bits per weight
```

This is the same memory footprint as Q4_1, llama.cpp's existing asymmetric 4-bit format. Same block size, same byte count. The difference is purely algorithmic — HQQ's proximal solver finds quantization parameters that produce lower reconstruction error than Q4_1's simple min/max approach.

Why does this matter for KV caches specifically? Because KV cache values have different statistical properties than model weights. Weights are trained and relatively well-behaved. KV cache values are computed at runtime from arbitrary user input — they can have outliers, skewed distributions, and patterns that simple min-max quantization handles poorly. HQQ's optimization-based approach adapts to these distributions without needing calibration data.

I implemented two variants: Q4_HQQ with a group size of 32 (5.0 bits per weight, optimized for speed) and Q4_HQQ_128 with a group size of 128 (4.25 bits per weight, optimized for compression). The larger group size means less overhead from scale/zero-point metadata, but coarser approximation of the distribution within each group.

### TurboQuant: Google's Rotational Trick

TurboQuant, published by Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni at ICLR 2026, takes a fundamentally different approach. Instead of building a better optimizer for the same quantization problem, it transforms the data so that a simpler quantizer works better.

The core insight comes from a technique called PolarQuant (presented separately at AISTATS 2026). When you apply a Randomized Hadamard Transform — essentially a structured random rotation — to a vector before quantizing it, something useful happens to the distribution of values. The rotation spreads the information across all coordinates, eliminating outliers and concentrating the distribution into a predictable shape. After rotation, the values follow a Beta distribution that is nearly the same regardless of what the original input looked like.

This is important because it means you can pre-compute the optimal quantization codebook once, at compile time, and use it for all inputs. There's no per-block scale or zero-point to store. The rotation makes all blocks look statistically similar.

TurboQuant uses this rotational preprocessing followed by Lloyd-Max optimal scalar quantization — an algorithm from 1960s information theory that finds the quantizer centroids that minimize mean squared error for a given distribution. Because PolarQuant makes the distribution predictable, Lloyd-Max can pre-compute 8 optimal centroids (for 3-bit) that work for any input.

The block layout:

```
TQ3_0 block (32 elements):
  - fp16 scale     [2 bytes]   — per-block norm
  - 32 x 3-bit     [12 bytes packed indices into Lloyd-Max codebook]
  Total: 14 bytes per 32 elements = 3.5 bits per weight
```

At 3.5 bits per weight, TQ3_0 achieves 4.6x compression over FP16. Compare this to HQQ's 3.2x at 5.0 bits. You're storing 30% less data per cached token.

The original TurboQuant paper also describes a second stage called QJL (Quantized Johnson-Lindenstrauss) that adds error correction using a random projection. Multiple independent implementers in the llama.cpp community found that this second stage is unnecessary for KV cache quantization in practice — allocating all bits to Lloyd-Max centroids gives better speed, simpler code, and equivalent perplexity. This is a community finding, not a claim from the original paper, and it may not generalize to all use cases.

### The Hadamard Rotation: Improving Everything Else

The third piece is a draft contribution from Georgi Gerganov, the creator of llama.cpp. His approach takes the rotational insight from TurboQuant and applies it differently: instead of creating a new quantization format, he applies a Hadamard rotation to the Q, K, and V activations before they enter the KV cache, and an inverse rotation when they come out.

This means every existing quantization type — Q4_0, Q5_0, Q8_0 — benefits from the rotation. The distribution of values going into the cache is more uniform, so even a simple symmetric quantizer produces lower error. On Qwen3 0.6B, this rotation alone dropped the perplexity of Q5_1 KV cache from 61.70 to 14.15. Same quantization format, same bits per weight, dramatically better quality — just by rotating the data first.

I integrated all three techniques into a single branch: HQQ for high-quality 4-bit KV caching, TQ3_0 for aggressive 3.5-bit compression, and the Hadamard rotation to improve quality across the board.

---

## The Integration: Rebasing 184 Commits and Resolving Type ID Collisions

The HQQ implementation existed as a set of patches I had developed against an earlier version of llama.cpp. The codebase had moved forward by 184 commits since those patches were created. TurboQuant existed as community forks that had never been combined with HQQ. Merging all of this required understanding how llama.cpp's type system works at a low level.

Every quantization format in llama.cpp has a numeric type ID — an integer in the `GGML_TYPE` enum. When you add a new type, you pick the next available number. The problem is that between when I wrote the HQQ patches and when I rebased them, the upstream project added NVFP4 (NVIDIA's FP4 format) at position 40 — the same position my HQQ patch was using.

This isn't a trivial conflict. Type IDs are baked into serialized model files, runtime dispatch tables, and backend kernel selection logic across CPU, CUDA, Metal, Vulkan, and OpenCL. Changing an ID isn't just editing one line — it's ensuring every `switch` statement, every type-traits array, and every backend that routes operations based on type ID is consistent.

The final layout after resolution:

| Type ID | Format | Origin |
|---------|--------|--------|
| 39 | MXFP4 | upstream |
| 40 | NVFP4 | upstream (new) |
| 41 | Q4_HQQ | HQQ patches |
| 42 | Q4_HQQ_128 | HQQ patches |
| 43 | TQ3_0 | TurboQuant community |

The rebase touched 8 files with merge conflicts across the type system, the quantization dispatch tables, CPU SIMD kernel registration, CUDA operation routing, and the command-line argument parser. One particularly subtle conflict was in `llama-context.cpp`, where the upstream had changed `n_embd_head_v` from a simple member variable to a per-layer function `n_embd_head_v(il)` — a refactor to support models with variable head dimensions across layers. The HQQ patch had the old API. Keeping the wrong version would have compiled fine but silently broken models with non-uniform head sizes.

Another conflict was in `llama-quant.cpp`, where the upstream had refactored a 40-line `switch` statement into a single function call `llama_ftype_get_default_type()`. The HQQ patch added two new cases to the old `switch`. The correct resolution was to delete the old `switch` entirely and add the HQQ cases to the new function instead. Getting this wrong would have caused runtime crashes when trying to quantize a model to HQQ format.

Conflict resolution in a project this large isn't about picking "ours" or "theirs." It's about understanding why each side changed what it changed, and producing a result that preserves both intents. Every one of these conflicts could have been resolved mechanically in a way that compiles but doesn't work. The only way to get it right is to read the surrounding code and understand the architectural intent of each change.

---

## Building for Android: The NDK, ADB, and Path Mangling

Cross-compiling llama.cpp for Android is straightforward with CMake and the Android NDK. The configuration:

```
Target:     arm64-v8a (AArch64)
API level:  28 (Android 9.0)
NDK:        r27d (Clang 18.0.4)
OpenMP:     disabled (static linking issues with NDK)
Build type: Release
```

The resulting binary is a statically linked ELF executable that runs directly on the device via ADB shell — no APK, no Java, no Android framework. Push the binary and the model file to `/data/local/tmp/`, `chmod +x`, and run.

One unexpected obstacle: ADB on Windows under Git Bash (MSYS2) silently rewrites destination paths. When you run `adb push file.bin /data/local/tmp/model.gguf`, MSYS2's path conversion intercepts `/data/local/tmp/model.gguf` and converts it to a Windows path like `C:/Users/you/data/local/tmp/model.gguf`. The push command reports "1 file pushed" but the file ends up on your local filesystem, not the device. The fix is `MSYS_NO_PATHCONV=1` as an environment prefix. This is a known MSYS2 behavior, but when `adb push` prints a success message while silently failing to push to the device, it can cost an hour of debugging before you realize the file never left your machine.

---

## On-Device Results: HQQ vs TurboQuant

Test device: a Qualcomm Snapdragon-based Android phone with an AArch64 CPU, approximately 5.7 GB total RAM and 3.4 GB available at test time.

Model: Qwen3-1.7B. Two weight quantizations were tested: Q8_0 (1.83 GB, 8.5 bpw) as a high-precision baseline to isolate KV cache effects, and Q4_HQQ (1.0 GB, 5.28 bpw) requantized on-device to test the full double-quantization stack.

### Inference 1: HQQ KV Cache (5.0 bits per weight)

```
llama-cli -m Qwen3-1.7B-Q8_0.gguf \
  --cache-type-k q4_hqq --cache-type-v q4_hqq \
  --flash-attn on -c 2048 -n 128
```

| Metric | Value |
|--------|-------|
| KV cache type | Q4_HQQ (K and V) |
| Bits per weight | 5.0 |
| KV compression vs FP16 | 3.2x |
| Context window | 2,048 tokens |
| Prompt processing | **9.8 tokens/sec** |
| Token generation | **1.6 tokens/sec** |
| Output quality | Coherent, structured reasoning |

The model produced well-organized output with clear thought structure. It correctly explained quantum computing concepts with appropriate analogies, maintained logical flow across multiple sentences, and showed no signs of quantization-induced degradation.

### Inference 2: TurboQuant TQ3_0 KV Cache (3.5 bits per weight)

```
llama-cli -m Qwen3-1.7B-Q8_0.gguf \
  --cache-type-k tq3_0 --cache-type-v tq3_0 \
  --flash-attn on -c 512 -n 64
```

| Metric | Value |
|--------|-------|
| KV cache type | TQ3_0 (K and V) |
| Bits per weight | 3.5 |
| KV compression vs FP16 | 4.6x |
| Context window | 512 tokens |
| Prompt processing | **24.3 tokens/sec** |
| Token generation | **4.9 tokens/sec** |
| Output quality | Functional, some reasoning repetition |

But those first two runs only tested KV cache compression with full-precision Q8_0 weights. The real question is: what happens when you compress *both* the weights and the KV cache?

### Inference 3: HQQ Weights + TurboQuant KV Cache (Maximum Compression)

I requantized the Q8_0 model to Q4_HQQ on-device using `llama-quantize`, producing a 1.0 GB model file — down from 1.83 GB. Then ran it with TQ3_0 KV cache, the most aggressive combination possible.

```
llama-quantize --allow-requantize Qwen3-1.7B-Q8_0.gguf Qwen3-1.7B-Q4_HQQ.gguf Q4_HQQ
# 1743 MiB -> 1084 MiB (5.28 BPW), 44.7 seconds on-device

llama-cli -m Qwen3-1.7B-Q4_HQQ.gguf \
  --cache-type-k tq3_0 --cache-type-v tq3_0 \
  --flash-attn on -c 512 -n 64
```

| Metric | Value |
|--------|-------|
| Model weights | Q4_HQQ (5.28 bpw, 1.0 GB) |
| KV cache type | TQ3_0 (3.5 bpw) |
| Prompt processing | **9.5 tokens/sec** |
| Token generation | **6.9 tokens/sec** |
| Output quality | **Gibberish** |

The output collapsed entirely. Instead of English, the model produced incoherent text — random characters, fragments, nothing resembling the prompt. The speed was fine (6.9 t/s generation is actually the fastest of all runs), but there was no signal in the output.

This is the double-quantization wall. HQQ-quantized weights at 5.28 bits already introduce reconstruction error in every matrix multiply. TQ3_0's 3.5-bit KV cache adds a second layer of error on every attention computation. On a 1.7B model, there simply aren't enough parameters to absorb the compounding noise. Each layer amplifies the error from the previous one, and by the time you reach the output projection, the signal has been destroyed.

### Inference 4: HQQ Weights + HQQ KV Cache (The Winner)

```
llama-cli -m Qwen3-1.7B-Q4_HQQ.gguf \
  --cache-type-k q4_hqq --cache-type-v q4_hqq \
  --flash-attn on -c 512 -n 64
```

| Metric | Value |
|--------|-------|
| Model weights | Q4_HQQ (5.28 bpw, 1.0 GB) |
| KV cache type | Q4_HQQ (5.0 bpw) |
| Prompt processing | **13.0 tokens/sec** |
| Token generation | **6.8 tokens/sec** |
| Output quality | **High — coherent reasoning, correct content** |

This was the surprise. HQQ weights + HQQ KV cache produced the best overall result: the smallest model (1.0 GB), the fastest generation speed (6.8 t/s), and fully coherent output. The model explained quantum computing correctly, with structured reasoning about qubits, superposition, and entanglement.

Why did this work when HQQ + TQ3_0 failed? Because using the same quantization method for both weights and KV cache keeps the error characteristics consistent. HQQ's proximal solver optimizes for reconstruction accuracy at 5 bits — the weights and the cached attention states both have similar error profiles. The model can tolerate this consistent distortion pattern. When you mix HQQ weights with TQ3_0's completely different codebook-based quantization, the error patterns are uncorrelated and compound unpredictably.

### The Full Picture

| Run | Weights | KV Cache | Model Size | Prompt t/s | Gen t/s | Quality |
|-----|---------|----------|-----------|-----------|---------|---------|
| 1 | Q8_0 (1.83 GB) | Q4_HQQ (5.0 bpw) | 1.83 GB | 9.8 | 1.6 | High |
| 2 | Q8_0 (1.83 GB) | TQ3_0 (3.5 bpw) | 1.83 GB | 24.3 | 4.9 | Moderate |
| 3 | Q4_HQQ (1.0 GB) | TQ3_0 (3.5 bpw) | 1.0 GB | 9.5 | 6.9 | Gibberish |
| **4** | **Q4_HQQ (1.0 GB)** | **Q4_HQQ (5.0 bpw)** | **1.0 GB** | **13.0** | **6.8** | **High** |

Three things stand out:

**Smaller model = faster generation.** Runs 3 and 4 (1.0 GB weights) achieved 6.8-6.9 t/s generation versus 1.6-4.9 t/s for Runs 1-2 (1.83 GB weights). On a bandwidth-constrained mobile SoC, reading 45% less weight data per token generation step makes a dramatic difference.

**TQ3_0 KV is only safe with high-precision weights.** Run 2 (Q8_0 + TQ3_0) produced acceptable output because 8-bit weights have enough precision to compensate for aggressive 3.5-bit KV compression. Run 3 (Q4_HQQ + TQ3_0) failed because there was no precision margin left.

**Uniform quantization beats mixed quantization on small models.** Run 4's consistent HQQ approach outperformed every mixed combination in the quality-speed-size tradeoff. This may not hold for 7B+ models, where larger parameter counts provide more error absorption capacity — but for the 1-3B models that actually run on phones, consistency wins.

---

## What This Means for Practical Context Windows

Here is the part that matters for anyone building products on edge AI. The table below shows how much conversation an LLM can hold in a fixed amount of KV cache memory:

| KV Cache Type | Bits | Memory per 1K Tokens (7B, 32 heads) | Context in 2 GB |
|---------------|------|-------------------------------------|-----------------|
| FP16 | 16.0 | 16.8 MB | ~119K tokens |
| Q8_0 | 8.5 | 8.9 MB | ~225K tokens |
| Q4_HQQ | 5.0 | 5.2 MB | ~385K tokens |
| TQ3_0 | 3.5 | 3.6 MB | **~556K tokens** |

On a phone with 2 GB of available KV memory, TurboQuant lets you hold a conversation 4.7x longer than FP16 before running out of space. That is the difference between a chatbot that forgets what you said 5 minutes ago and one that maintains context for an entire working session.

For smaller models — the 1-3B parameter range that actually runs well on phones — the improvement is even more dramatic because the KV cache constitutes a larger fraction of total memory usage. A 1.7B model in Q4_0 quantization uses about 1 GB for weights. With FP16 KV cache, a 4K context adds 67 MB. With TQ3_0, the same context adds 15 MB. The model itself dominates memory, and the KV cache becomes almost negligible.

---

## The Ecosystem: Where TurboQuant Stands Today

TurboQuant has generated more community activity than any KV cache technique since flash attention. Within 48 hours of the Google Research blog post, the llama.cpp discussion thread had multiple independent implementations:

- **CPU implementation** with Lloyd-Max codebook and WHT rotation (functional, benchmarked)
- **Metal implementation** for Apple Silicon, achieving 102% of Q8_0 prefill speed on M5 Max
- **CUDA implementation** enabling 700K context on an RTX 5090 with a 27B model
- **Standalone C reference** with 18/18 correctness tests passing

Several pull requests were submitted to the main llama.cpp repository. All were closed — some for not following the project's contribution guidelines (which require human-authored code with AI used only in an assistive capacity), others for merge issues or build failures.

The approach most likely to be merged upstream is Gerganov's Hadamard rotation, which captures the core insight of TurboQuant (rotation reduces outliers) without introducing new quantization types. It is backend-agnostic, works with all existing formats, and requires no new GPU kernels.

This is a pattern worth noting: the research community publishes a technique with a complete, novel architecture (new types, new codebooks, new kernels). The open-source community discovers that 80% of the benefit comes from one key insight (the rotation), and the most practical implementation applies that insight to existing infrastructure rather than building new infrastructure around the full technique.

---

## Engineering Observations

### On-device initialization matters more than you think

TQ3_0's Lloyd-Max codebook is computed iteratively — 178 iterations of convergence to find optimal centroids for the target distribution. On a desktop CPU, this takes milliseconds. On a mobile ARM core, with a 2048-token context, the initialization took long enough to be noticeable. Pre-computing the codebook as compile-time constants would eliminate this entirely, and the values are deterministic — they depend only on the target distribution (which is fixed by the Hadamard rotation) and the bit width.

### Flash attention isn't optional for quantized V caches

If you quantize the Value cache but don't enable flash attention, llama.cpp materializes the full attention matrix in memory — an O(n^2) allocation that defeats the purpose of cache compression. The `--flash-attn on` flag is mandatory when using quantized V types. The TurboQuant Metal implementation added auto-detection: if the cache type is quantized, flash attention is silently enabled. This is the kind of UX decision that prevents users from getting silently terrible results.

### The MSYS2 path mangling problem is real

I lost time to an issue where `adb push` appeared to succeed (printing "1 file pushed") but the file never reached the device. Git Bash on Windows translates Unix-style paths in command arguments to Windows paths, and ADB interprets the rewritten path as a local destination instead of a device path. The symptom is maddening: the command succeeds, the file vanishes, and the device directory is empty. `MSYS_NO_PATHCONV=1` fixes it, but you have to know it exists. This is the kind of environment-specific friction that eats engineering hours and has nothing to do with the actual problem you're solving.

### Merge conflict resolution is an underappreciated skill

Rebasing a 39-file patch across 184 upstream commits produced conflicts in type ID enums, backend dispatch tables, API signatures that had been refactored, and switch statements that had been replaced by function calls. Each conflict had a mechanically obvious resolution (keep both sides) and a semantically correct resolution (understand why each side changed and produce a result that preserves both intents). These are not the same thing. The mechanically obvious resolution compiled in every case. It would have produced incorrect runtime behavior in three of them.

---

## What I Would Do Differently

**Start with the rotation, not the new type.** Gerganov's Hadamard rotation improves every existing quantization type with no new infrastructure. If I were building a product today, I would ship rotation + Q4_0 KV cache before investing in a custom TQ3_0 type. The complexity-to-benefit ratio is dramatically better.

**Pre-compute everything possible.** Lloyd-Max centroids, Hadamard matrices, sign flip patterns — these are all deterministic and can be compile-time constants. Computing them at runtime adds initialization latency on exactly the devices where latency matters most.

**Don't mix quantization methods on small models.** The most surprising result of this entire experiment was that uniform HQQ quantization (weights + KV cache) produced better results than any mixed combination. On a 1.7B model, consistency of the error profile matters more than the theoretical advantage of a more advanced method. If you're deploying to phones, pick one quantization approach and use it everywhere.

**Test the full pipeline on-device before optimizing.** I knew from my previous Vulkan project that on-device behavior is different from desktop behavior. But I still spent time optimizing the rebase and integration before confirming that the binary ran correctly on the phone. The right order is: build, push, verify one inference, then optimize.

---

## Where This Goes

KV cache quantization is moving fast. Within months we've gone from "FP16 is the only option" to "3.5 bits with near-baseline quality" — a 4.6x compression that fundamentally changes what's possible on memory-constrained devices.

The remaining gaps are clear:

1. **Rotation + TQ3_0 integration.** The Hadamard rotation and the Lloyd-Max codebook are complementary — rotation improves the input distribution, codebook-based quantization exploits the improved distribution. Wiring them together end-to-end should close the quality gap between TQ3_0 and Q8_0.

2. **GPU kernels for TQ3_0.** The current CPU-only implementation means TQ3_0 can't benefit from GPU acceleration. CUDA and Metal implementations exist in community forks but haven't been upstreamed.

3. **Double quantization at scale.** HQQ weights + TQ3_0 KV cache collapsed on a 1.7B model, but larger models (7B+) have more redundant parameters to absorb compounding error. Testing this combination on Llama-3.1-8B or Qwen3-8B would determine whether the double-quantization wall is a small-model phenomenon or a fundamental limit.

4. **Mixed K/V quantization strategies.** Using different precision for K and V caches (e.g., TQ3_0 for Keys, HQQ for Values) could find a better quality-compression trade-off than using the same type for both. The Key cache primarily affects attention routing, while the Value cache directly affects output content — they may tolerate different levels of quantization error.

5. **Sustained performance measurement.** Every benchmark I've reported is peak throughput. On mobile, thermal throttling degrades performance within 30-60 seconds of sustained inference. The real question isn't "how fast is the first token" but "how fast is the thousandth token, five minutes into a conversation, on a phone that's warming up in someone's hand."

The memory wall for on-device LLMs is real, but it's not fixed. Compressing the KV cache from 16 bits to 3.5 bits is not a theoretical exercise — it runs today, on a phone, generating coherent text. The gap between "runs on a server" and "runs in your pocket" is narrowing, and KV cache quantization is one of the key techniques closing it.

---

*All benchmarks were measured on-device via ADB shell. Models: Qwen3-1.7B-Q8_0 and Qwen3-1.7B-Q4_HQQ (requantized on-device). Implementation built on llama.cpp (ggml-org/llama.cpp). HQQ reference: Badri & Shaji, "Half-Quadratic Quantization of Large Machine Learning Models" (2023). TurboQuant: Zandieh, Daliri, Hadian & Mirrokni, "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," ICLR 2026 (arXiv:2504.19874). PolarQuant: Han, Kacham, Karbasi, Mirrokni & Zandieh, AISTATS 2026 (arXiv:2502.02617).*
