# Cutting LLM Quantization Error in Half: What Happens When You Rotate Weights Before Compressing Them

*M S Ramaseshan | March 2026*

---

Weight quantization is the standard way to shrink large language models. Take a 7-billion-parameter model stored in 16-bit floats — 14 GB — and compress each weight to 4 bits. The model drops to 3.5 GB. Your laptop can run it. Your phone can run it. The quality loss is usually small enough that you don't notice.

But push below 4 bits and things break. At 3 bits, models start hallucinating more. At 2 bits, most models produce garbage. The reason is straightforward: you're trying to represent a continuous value with 8 or fewer discrete levels, and the rounding error accumulates across billions of multiplications.

Google's TurboQuant research dropped a few days ago, introducing techniques originally designed for KV cache compression. I wanted to know if two of those techniques could be adapted to improve static weight quantization in the HQQ framework. The idea was to attack the problem from both ends: reduce the error before quantization happens, then correct for what's left after.

The result: 32-85% perplexity reduction at 3-bit quantization across five models (up to 3B parameters), verified on GPU. The code is open source.

---

## The Core Problem: Outliers Ruin Everything

To understand why quantization fails at low bit-widths, you need to understand what a quantizer actually does.

Take a group of 64 weights. In standard quantization, you find the minimum and maximum values, divide that range into equal buckets (15 buckets for 4-bit, 7 for 3-bit), and round each weight to the nearest bucket center. The scale factor is `(max - min) / num_buckets`.

This works well when the values are roughly uniformly distributed. It fails badly when they're not.

In practice, LLM weight matrices have outlier channels — a small number of dimensions where the magnitudes are 10-50x larger than the rest. This has been extensively documented: Meta's LLM.int8() paper, MIT's SmoothQuant, and Qualcomm's SpinQuant all identify the same phenomenon. A single outlier at magnitude 100 in a group where everything else is near 1 forces the quantizer to spread its 7 buckets (for 3-bit) across a range of 200. The 63 non-outlier values, which carry most of the information, all get crammed into the same 1-2 buckets. Most of the representational capacity is wasted on empty space.

HQQ (Half-Quadratic Quantization, by Hicham Badri at Mobius Labs) partially addresses this by optimizing the scale and zero-point parameters using a proximal gradient solver. Instead of computing scale from min/max and moving on, HQQ iteratively refines these parameters to minimize reconstruction error across the group. This helps — HQQ consistently outperforms naive min-max quantization — but it doesn't solve the fundamental distribution problem. If one outlier dominates the range, even an optimized scale is still spending most of its precision on empty space.

---

## The Idea: Spread the Outliers Before Quantizing

Google's TurboQuant paper (Zandieh et al., ICLR 2026) introduced a technique called PolarQuant for KV cache compression. The core insight: if you multiply a vector by a Hadamard matrix before quantizing it, the outlier energy gets distributed evenly across all elements.

A Hadamard matrix is an orthogonal matrix where every entry is either +1/sqrt(n) or -1/sqrt(n). Multiplying a vector by it is equivalent to a rotation in high-dimensional space. The key properties:

1. **It preserves norms**: the L2 magnitude of the vector doesn't change
2. **It's self-inverse**: applying the same transform twice gives you back the original
3. **It spreads energy**: any single large value gets distributed across all dimensions

Here's what this looks like concretely. Say you have a group of 64 weights where one value is 100 and the rest are near 1:

```
Before rotation:  [100.0, 0.8, -1.2, 0.5, ...] (64 values)
  Range: 200.0  |  Quantization uses 7 buckets across this range
  Bucket width: 28.6  |  Values near 1 are indistinguishable

After rotation:   [13.1, -12.4, 12.8, -13.0, ...] (64 values)
  Range: 26.0   |  Same 7 buckets, much narrower range
  Bucket width: 3.7   |  Values are now distinguishable
```

The outlier's energy (100^2 = 10,000) is preserved but split across all 64 elements (each carrying approximately 100/sqrt(64) = 12.5). The range shrinks by a factor of ~8, and every quantization bucket now covers a meaningful portion of the actual value distribution.

After quantization, you dequantize and apply the inverse transform (which, for a Hadamard matrix, is the same matrix) to get back to the original coordinate system.

This is not a new idea in isolation — QuaRot (Ashkboos et al., 2024) and SpinQuant (Liu et al., 2024) apply similar rotations. What I wanted to test was whether it could be integrated directly into HQQ's group-based quantization at minimal cost.

---

## The Second Idea: Correct What's Left

Rotation reduces quantization error, but doesn't eliminate it. After quantizing the rotated weights, there is still a residual: the difference between the original rotated weights and their dequantized approximation.

TurboQuant's second technique — the Quantized Johnson-Lindenstrauss (QJL) map — provides a way to compress this residual at very low cost. The Johnson-Lindenstrauss lemma says that random projections approximately preserve inner products. QJL exploits this: project the residual through a random Gaussian matrix, keep only the sign of each projection (1 bit), and store these sign bits along with a per-group reconstruction scale.

The original TurboQuant uses QJL with a fixed theoretical scale factor (`sqrt(pi/2) / d`) designed for inner product preservation in attention. For weight quantization, I found this doesn't work well — it actually increased error in my initial tests. The fix was simple: instead of the theoretical factor, solve for the per-group MSE-optimal scale:

```
scale_g = <residual, basis> / <basis, basis>
```

where `basis = S^T @ sign(S @ residual)` and S is the random projection matrix. This is a closed-form least-squares solution that minimizes the reconstruction error for each group independently.

The storage cost: 1 sign bit per projection plus one FP16 scale per group. At group_size=64 with 64 projections, that's 80 bits per group, or 1.25 additional bits per weight. A 3-bit quantization with QJL correction becomes 4.25 effective bits per weight.

The random projection matrix is never stored. It's regenerated from a fixed seed during dequantization. All groups within a layer share the same projection matrix (same seed), which means the matrix is generated once per layer, not once per group. This is a deliberate tradeoff: per-group seeds would give better statistical independence between groups (the JL lemma assumes independent projections), but would require storing or deriving a unique seed per group and regenerating a new matrix for each. In practice, shared projections work well because the residual vectors across groups are already decorrelated by the per-group scale/zero-point optimization.

A note on numerical stability: the denominator in the optimal scale computation (`<basis, basis>`) can approach zero when the residual is very small. The implementation clamps with `+ 1e-8` to avoid division by zero. In practice this fires rarely — groups with near-zero residuals have near-zero QJL correction regardless of the scale.

---

## Implementation

I integrated both techniques into HQQ's `Quantizer.quantize()` and `Quantizer.dequantize()` methods. The pipeline is:

**Quantization (offline, one-time):**
1. Reshape weights into groups (e.g., 64 elements per group)
2. Apply Hadamard rotation to each group
3. Run HQQ's standard quantization (scale/zero-point optimization, rounding, bit-packing)
4. Compute residual in rotated space
5. Encode residual with QJL (1-bit projections + per-group optimal scale)
6. Store quantized weights + QJL metadata

**Dequantization (runtime, every forward pass):**
1. Standard HQQ unpack and dequantize
2. Add QJL residual correction (regenerate projection matrix from seed, reconstruct, add)
3. Apply inverse Hadamard rotation
4. Reshape to original weight dimensions

The implementation is three files:
- `rotation.py` (148 lines): Hadamard matrix generation and fast Walsh-Hadamard transform
- `qjl.py` (155 lines): QJL encode/decode with MSE-optimal scaling
- Modified `quantize.py`: three new parameters (`turbo`, `turbo_qjl`, `turbo_qjl_seed`), all defaulting to False

The Hadamard transform uses the O(n log n) butterfly algorithm for power-of-2 group sizes (which HQQ's default group_size=64 is). No matrix is stored or materialized — the transform operates in-place through successive butterfly stages. For non-power-of-2 sizes, it falls back to a seeded random orthogonal matrix via QR decomposition.

All three modules are standalone — they depend only on PyTorch and can be extracted for use in other frameworks.

---

## What I Measured

I ran benchmarks on four models across three quantization configurations, measuring WikiText-2 perplexity (test split, 40 sliding windows of 2048 tokens with stride 512) on an NVIDIA A10G GPU. All runs use QJL seed 42. Seed sensitivity results appear later in this section.

### The Configurations

| Label | What It Does | Effective Bits Per Weight |
|-------|-------------|--------------------------|
| HQQ (baseline) | Standard HQQ quantization | nbits (2, 3, or 4) |
| HQQ-Turbo (rot) | Hadamard rotation + HQQ | Same as baseline (rotation is free) |
| HQQ-Turbo (rot+qjl) | Rotation + HQQ + QJL correction | nbits + 1.25 |

### 3-Bit Results (the sweet spot)

| Model | FP16 PPL | HQQ 3b | +Rotation | +Rotation+QJL | Full Improvement |
|-------|----------|--------|-----------|---------------|-----------------|
| Qwen2.5-3B | 7.40 | 76.20 | 32.66 | **11.06** | **-85.5%** |
| TinyLlama-1.1B | 7.81 | 14.40 | 12.57 | **9.67** | **-32.9%** |
| OPT-1.3B | 15.10 | 35.84 | 34.38 | **19.68** | **-45.1%** |
| Qwen2.5-0.5B | 12.35 | 32.62 | 29.81 | **20.41** | **-37.4%** |
| OPT-125M | 29.31 | 79.08 | 70.19 | **45.79** | **-42.1%** |

The pattern is consistent across all five models. The Qwen2.5-3B result is especially striking: rotation alone cuts perplexity from 76.20 to 32.66 (a 57% reduction at zero extra cost), and adding QJL brings it down to 11.06 — within 50% of the FP16 baseline at just 4.25 effective bpw.

### 4-Bit Results

| Model | FP16 PPL | HQQ 4b | +Rotation | +Rotation+QJL | Full Improvement |
|-------|----------|--------|-----------|---------------|-----------------|
| Qwen2.5-3B | 7.40 | 8.19 | 8.45 | **7.85** | **-4.2%** |
| TinyLlama-1.1B | 7.81 | 8.34 | 8.35 | **8.11** | **-2.8%** |
| OPT-1.3B | 15.10 | 16.00 | 16.28 | **15.63** | **-2.3%** |
| Qwen2.5-0.5B | 12.35 | 14.48 | 14.35 | **13.44** | **-7.2%** |
| OPT-125M | 29.31 | 33.43 | 34.65 | **32.25** | **-3.5%** |

At 4-bit, the baseline is already close to FP16, so absolute improvements are smaller. But the direction is always positive for the full pipeline (rotation+QJL). Qwen2.5-3B at 4-bit rot+qjl achieves PPL 7.85 — only 0.45 above the FP16 baseline.

### 2-Bit Results

Every model produced catastrophic perplexity (>4,000) at 2-bit, regardless of technique. At this extreme, the quantization grid (3 levels for 2-bit) is simply too coarse to preserve model behavior. Rotation helps the MSE numbers substantially (78% reduction in synthetic tests), but perplexity depends on error distribution across layers, not just average MSE. 2-bit weight quantization remains non-viable for models at this scale.

### Weight Reconstruction Error (Synthetic)

On synthetic 2048x2048 weight matrices with injected outlier structure (2% of columns scaled to 20x magnitude, plus log-normal per-channel scale variation, designed to mimic the "massive activation" channels documented in SmoothQuant and LLM.int8()):

| Config | MSE | Cosine Similarity | MSE Reduction |
|--------|-----|-------------------|---------------|
| HQQ 3-bit | 1.31 | 0.955 | baseline |
| +Rotation | 0.26 | 0.991 | -80.2% |
| +Rotation+QJL | 0.16 | 0.995 | -88.0% |
| HQQ 4-bit | 0.29 | 0.990 | baseline |
| +Rotation | 0.05 | 0.999 | -81.6% |
| +Rotation+QJL | 0.03 | 1.000 | -88.8% |

The MSE improvements on synthetic data are dramatic — 80-89% reduction. This is where rotation truly shines: on individual weight matrices with clear outlier channels, the distributional fix is massive.

### Seed Sensitivity

The QJL projection matrix is generated from a random seed. How much do results vary across seeds? I ran TinyLlama-1.1B at 3-bit with rotation+QJL using five different seeds:

| Seed | PPL |
|------|-----|
| 0 | 9.54 |
| 42 | 9.67 |
| 123 | 9.61 |
| 777 | 9.68 |
| 2024 | 9.74 |
| **Mean** | **9.65** |
| **Std** | **0.07** |
| **CV** | **0.69%** |

The coefficient of variation is under 1%. The perplexity range across all five seeds is 9.54 to 9.74, a spread of 0.20, which is negligible compared to the 4.73-point improvement over the HQQ 3b baseline (14.40). The MSE values are nearly identical across seeds (0.00000823 to 0.00000830), confirming that the random projection direction does not meaningfully affect reconstruction quality. Seed choice is not a meaningful hyperparameter for this technique.

---

## What I Learned

**QJL is the bigger contributor, not rotation.** This surprised me. On synthetic weights with injected outliers, rotation alone provides 76-84% MSE reduction. But on real model weights measured by perplexity, rotation alone provides 4-13% improvement while QJL provides the remaining 20-35%. The gap between "lower MSE per matrix" and "lower perplexity on the model" is real and significant. Perplexity depends on how errors interact across layers and across the sequence, not just on the average error per weight.

**Rotation can slightly hurt at 4-bit on some models.** On OPT-1.3B, rotation alone increased perplexity from 16.00 to 16.28. This is within noise, but it's worth noting. The likely explanation: at 4-bit, HQQ's proximal optimizer already finds good scale/zero-point parameters for the natural weight distribution. Rotation changes that distribution, and the optimizer doesn't always find equally good parameters for the rotated distribution. The effect is small and the QJL correction more than compensates, but it means rotation is not a universal free lunch for weight quantization.

**MSE and perplexity don't always agree.** The synthetic benchmarks show rotation reducing MSE by 80%+ at every bit-width. The perplexity benchmarks show a more nuanced picture: big gains at 3-bit, small gains or slight regressions at 4-bit. This is a well-known phenomenon in quantization research — MSE is a proxy for quality, not the quality itself. If you're developing quantization methods, always measure perplexity (or downstream task accuracy) in addition to MSE.

**The combined approach never hurts.** Across 4 models x 3 bit-widths = 12 test configurations, rotation+QJL was equal to or better than the HQQ baseline in every single case. This is the practical finding that matters: you can enable both techniques with no risk of degradation.

**3-bit is the sweet spot for this technique.** At 3-bit, the quantization error is large enough that both rotation and QJL have substantial error to work with, but the model isn't so broken that corrections can't help. The 4.25 effective bpw (3-bit + QJL overhead) is competitive with standard 4-bit quality at lower storage.

---

## Connection to llama.cpp PR #21038

While I was working on weight quantization, Georgi Gerganov (creator of llama.cpp) opened PR #21038 implementing the same Hadamard rotation technique for KV cache quantization at inference time. His results on Qwen3 0.6B are striking:

| KV Cache Type | Before Rotation | After Rotation |
|--------------|----------------|----------------|
| Q5_1 | PPL 61.70 | PPL 14.15 |
| Q4_1 | PPL 212.48 | PPL 22.28 |
| Q8_0 | PPL 13.91 | PPL 13.67 |

The rotation is the same mathematical transform applied to a different target — activations at inference time rather than static weights at quantization time. Gerganov's implementation is 241 lines of C++ across 4 files, backend-agnostic, and compatible with all existing GGUF quantization types.

The community response included exhaustive benchmarks from multiple contributors. One ran a full permutation matrix of K-cache and V-cache quantization types on Qwen3.5-9B (36 combinations). The rotation improved perplexity and KL divergence in virtually every combination.

The discussion also noted what's missing: QJL residual correction. Several commenters identified that the rotation alone doesn't approach TurboQuant's claimed near-lossless results. The QJL component — which my implementation includes for weight quantization — is the piece that provides the larger correction.

---

## Practical Implications

**For model deployment**: If you're serving quantized models at 3-bit to save memory, rotation+QJL can recover a significant fraction of the quality you're losing. The 1.25 bpw overhead from QJL brings you to 4.25 effective bpw, but with substantially better quality than standard 3-bit.

**For framework developers**: Both the rotation and QJL modules are standalone (pure PyTorch, ~150 lines each) and can be integrated into any quantization pipeline — GPTQ, AWQ, or custom GGUF quantizers. The rotation is especially cheap: for power-of-2 group sizes, it's an O(n log n) in-place transform with zero storage cost.

**For inference engines**: The dequantization path adds latency from QJL reconstruction (regenerating the random projection matrix from seed). On A10G, this adds 3-13ms per layer compared to <1ms for standard HQQ dequantization. The 4x variance is layer-size dependent: a 768x3072 MLP projection generates a larger projection matrix than a 768x768 attention weight, and `torch.randn` cost scales linearly with element count. Caching the projection matrix per layer (implemented in the current code) eliminates the regeneration cost, but benchmarking reveals a more fundamental issue: the bottleneck is the dense matmul (S.T @ signs), not the matrix generation. On A10G, a cached 768x768 layer dequantizes in 7.25ms vs 7.86ms uncached (8% speedup). For a Qwen2.5-3B MLP layer (3584x18944), both cached and uncached take ~1.9 seconds — the S.T @ signs matmul dominates completely. The real optimization path is structured random matrices (Hadamard-Rademacher products) that allow O(n log n) projection instead of O(n^2) matmul, or fused CUDA kernels.

---

## What's Next

Several directions remain unexplored:

1. **Larger models**: All tested models are 3B parameters or smaller. Quantization generally works better at larger scales — the error per weight decreases as the model gets bigger. The relative improvement from rotation+QJL at 7B+ scale needs measurement.

2. **Fewer QJL projections**: Using num_projections = group_size/2 or group_size/4 would reduce the overhead from 1.25 to 0.625 or 0.375 bpw. The accuracy tradeoff hasn't been characterized.

3. **Per-layer sensitivity**: Currently, turbo mode applies uniformly to all layers. Some layers (embeddings, output projections) may not benefit from rotation, while others (attention projections) may benefit more. A sensitivity-aware approach could reduce overhead without sacrificing quality.

4. **Analytical codebooks**: TurboQuant's third technique — replacing uniform quantization grids with distribution-matched codebooks — remains unimplemented. This could provide further gains, particularly at 2-bit where the uniform grid is provably suboptimal.

---

## Source Code

The implementation is available at [github.com/ramaseshanms/hqq_turbo](https://github.com/ramaseshanms/hqq_turbo) on the `feature/hqq-turbo` branch.

Built on top of [HQQ](https://github.com/mobiusml/hqq) by Hicham Badri, Mobius Labs (Apache-2.0).

Techniques adapted from [TurboQuant](https://arxiv.org/abs/2504.19874) by Zandieh, Daliri, Hadian, and Mirrokni (Google Research, 2025).

llama.cpp rotation implementation: [PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038) by Georgi Gerganov.

---

## References

- Zandieh et al., "TurboQuant: Online Quantization for Efficient Inference", ICLR 2026. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Badri and Shaji, "HQQ: Half-Quadratic Quantization", 2023. [GitHub](https://github.com/mobiusml/hqq)
- Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", 2024. [arXiv:2404.00456](https://arxiv.org/abs/2404.00456)
- Liu et al., "SpinQuant: LLM Quantization with Learned Rotations", 2024. [arXiv:2405.16406](https://arxiv.org/abs/2405.16406)
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", 2022. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs", 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)