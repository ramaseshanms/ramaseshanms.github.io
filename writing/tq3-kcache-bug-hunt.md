# How I Found a Bug in TurboQuant by Asking a Language Model What 2+2 Is

*M S Ramaseshan | March 2026*

---

I ran a quantized language model on a phone. It produced gibberish. My first explanation was wrong. My second explanation was wrong. The actual cause took six targeted experiments and about forty minutes to find.

This is a short piece about debugging a quantization bug in an open-source LLM runtime, and why the obvious theory is sometimes the worst one to chase.

---

## The Setup

I had just integrated two KV cache quantization methods into llama.cpp for Android deployment:

- **HQQ** (Half-Quadratic Quantization) — a calibration-free 4-bit method at 5.0 bits per weight
- **TurboQuant TQ3_0** — Google Research's 3-bit method using Hadamard rotation and Lloyd-Max codebooks, at 3.5 bits per weight

Both are designed to compress the KV cache — the memory an LLM uses to remember your conversation. On a phone with 3.4 GB free RAM, the KV cache is often the bottleneck, not the model weights.

The model was Qwen3-1.7B, running on a Qualcomm Snapdragon AArch64 CPU. I had four configurations to test:

| Run | Model Weights | KV Cache | Goal |
|-----|--------------|----------|------|
| 1 | Q8_0 (1.83 GB) | HQQ | Baseline quality |
| 2 | Q8_0 (1.83 GB) | TQ3_0 | Test TurboQuant alone |
| 3 | Q4_HQQ (1.0 GB) | TQ3_0 | Maximum compression |
| 4 | Q4_HQQ (1.0 GB) | HQQ | Uniform HQQ |

Runs 1, 2, and 4 worked. Run 3 produced gibberish — random characters, fragments of non-English scripts, no semantic content at all.

---

## Theory 1: Double Quantization Error (Wrong)

The obvious explanation: compressing both the weights (Q4_HQQ at 5.28 bpw) and the KV cache (TQ3_0 at 3.5 bpw) creates compounding quantization error. Each transformer layer amplifies the reconstruction error from the previous one. On a 1.7B model with only 28 layers and limited parameter redundancy, the signal-to-noise ratio degrades until the output is meaningless.

This theory is clean, intuitive, and explains the data: higher-precision weights (Q8_0) give TQ3_0 enough headroom to work (Run 2), while lower-precision weights (Q4_HQQ) don't (Run 3).

I believed this for about an hour. Then I tried three strategies to fix it, and all of them failed in ways the theory couldn't explain.

---

## Strategy 1: Mixed K/V Cache

**Hypothesis:** The Value cache directly determines output token content. The Key cache only affects attention routing. If quantization error is the problem, the V cache should be more sensitive. Use TQ3_0 for Keys (where precision matters less) and HQQ for Values (where it matters more).

```
llama-cli -m Qwen3-1.7B-Q4_HQQ.gguf \
  --cache-type-k tq3_0 --cache-type-v q4_hqq \
  --flash-attn on -c 512 -n 64
```

**Result:** Gibberish. Output: `ástéľțș`

This was unexpected. The V cache was now at 5.0 bpw (HQQ) — the same precision that produced coherent output in Run 4 when used for both K and V. The only 3.5-bit component was the K cache, which I'd hypothesized was less sensitive.

If double-quantization error were the cause, giving the V cache more precision should have helped. It didn't.

## Strategy 2: Constrained Sampling

**Hypothesis:** Maybe the logits aren't completely destroyed — they're just noisy enough that the sampler picks wrong tokens. Lowering temperature to 0.1 and increasing repetition penalty to 1.5 should force the sampler to pick the highest-probability tokens and avoid degenerate loops.

```
llama-cli -m Qwen3-1.7B-Q4_HQQ.gguf \
  --cache-type-k tq3_0 --cache-type-v tq3_0 \
  --flash-attn on -c 512 -n 64 \
  --temp 0.1 --repeat-penalty 1.5
```

**Result:** Gibberish. Output: `ástéľțș`

If the problem were noisy logits, aggressive temperature scaling would have cleaned it up. The output was identical regardless of sampling parameters. The corruption is happening before the logits, not after.

## Strategy 3: Higher Precision Weights

**Hypothesis:** Q4_HQQ at 5.28 bpw is too aggressive. Use Q6_K at 6.56 bpw — higher precision, more headroom for TQ3_0.

I requantized on-device to Q6_K (1.35 GB) and ran with TQ3_0 KV cache.

```
llama-cli -m Qwen3-1.7B-Q6_K.gguf \
  --cache-type-k tq3_0 --cache-type-v tq3_0 \
  --flash-attn on -c 512 -n 64
```

**Result:** Gibberish. Output: a single Arabic character.

This killed Theory 1. Q6_K is only 1.94 bpw below Q8_0 (6.56 vs 8.50). If double quantization were the problem, Q6_K should have given TQ3_0 plenty of headroom — it's less than 23% less precise than the Q8_0 that supposedly "worked" in Run 2.

At this point, I also retested Run 2 (Q8_0 + TQ3_0) with a controlled seed. It also produced gibberish: `ätig iguiente Okay Okay Oh Oh!`. The earlier "moderate quality" result was either a lucky seed or I'd been too generous in my assessment. TQ3_0 was broken across the board, regardless of weight precision.

---

## The Right Question

All three strategies treated weight precision as the variable. None of them worked because weight precision was never the problem.

I stepped back and asked a different question: what exactly does TQ3_0 do during inference, and which specific operation could produce this failure mode?

The KV cache has two components, and they participate in attention differently:

1. **Key cache** — used in the QK dot product. This dot product computes attention scores: how much each previous token should influence the current one. The Key cache values go through a `vec_dot` function — a quantized dot product that dequantizes and accumulates across 128 dimensions per head.

2. **Value cache** — multiplied by the attention weights (the softmax of the QK scores). The Values are dequantized and used in a weighted sum. No dot product — just multiply-accumulate with known-correct attention weights.

If TQ3_0's `vec_dot` is broken, the attention scores are garbage. The model looks at the wrong tokens. Everything downstream is noise, regardless of how precise the weights are.

If TQ3_0's dequantization is broken, both K and V paths fail.

If only `vec_dot` is broken, V-only should work (it doesn't use `vec_dot`), and K-only should fail.

---

## The Isolation Test

I ran the simplest possible prompt — `What is 2+2? Answer with just the number.` — with a fixed seed, varying only which cache uses TQ3_0:

| K Cache | V Cache | Output | Verdict |
|---------|---------|--------|---------|
| TQ3_0 | TQ3_0 | `ätig iguiente Okay Okay Oh Oh!` | Broken |
| TQ3_0 | Q8_0 | Arabic script repetition | **K is broken** |
| Q8_0 | TQ3_0 | `Let me think... 2 plus 2 is 4` | **V works** |
| Q4_HQQ | TQ3_0 | Coherent quantum computing explanation | **V works** |

Four runs. Forty seconds each. The pattern was unambiguous:

- Every run with TQ3_0 as K-cache: gibberish.
- Every run with TQ3_0 as V-cache only: perfect.

The bug is in TQ3_0's `vec_dot` function — the quantized dot product used for QK attention scoring. The dequantization path (used for V-cache) works correctly.

---

## What the Bug Probably Is

TQ3_0 packs 3-bit indices into 12 bytes per block of 32 elements. Each 3-bit index maps to one of 8 Lloyd-Max centroids. The `vec_dot` function extracts these indices, looks up centroids, and accumulates `centroid * query_element` across 128 dimensions.

The likely culprits:

1. **Packed index extraction error.** 3 bits don't align to byte boundaries. Extracting 32 three-bit indices from 12 bytes requires careful bit shifting and masking. An off-by-one in the extraction would map some elements to wrong centroids — close enough to pass single-element tests, wrong enough to corrupt a 128-dimensional dot product.

2. **Scale application order.** The per-block FP16 scale might be applied at the wrong point in the accumulation — before centroid lookup instead of after, or multiplied when it should divide.

3. **Sign handling.** The Lloyd-Max codebook for a zero-mean distribution has both positive and negative centroids. If the sign bit or the centroid index mapping is wrong for some bit patterns, the dot product accumulates in the wrong direction.

The dequantization path doesn't have this bug because it operates element-by-element: extract index, lookup centroid, multiply by scale, write to output buffer. The dot product path does the same extraction but accumulates instead of writing — and the accumulation magnifies any per-element error by 128x.

---

## The Fix and the Workaround

The workaround is immediate: use TQ3_0 for V-cache only.

```
llama-cli -m model.gguf \
  --cache-type-k q4_hqq --cache-type-v tq3_0 \
  --flash-attn on
```

This gets TurboQuant's 4.6x compression on the V-cache (which is typically the same size or larger than the K-cache) while using a reliable type for the K-cache. On my test device, this configuration produced 13.3 tok/s prompt processing and 7.0 tok/s generation — the fastest coherent result of all runs.

The proper fix requires debugging the `vec_dot` implementation in the TQ3_0 CPU backend. A targeted test would be: generate a random Q vector and a known K vector, compute the dot product via `vec_dot`, and compare against an FP32 reference. If they diverge, bisect across the 128 dimensions to find which packed byte boundary produces the wrong index.

---

## What I Learned

**The obvious theory is seductive and expensive.** "Double quantization error exceeds model capacity" is a satisfying explanation. It's mathematically plausible, it matches the observed failure, and it implies the problem is fundamental rather than fixable. I spent an hour trying to work around it before discovering the actual cause was a bug in a specific function.

**Systematic isolation beats clever hypotheses.** Three strategies based on Theory 1 all failed. One four-run isolation test based on "which component is actually broken" found the answer in minutes. The strategies were creative. The isolation test was boring. The boring approach worked.

**The cost of a wrong theory isn't the time spent testing it — it's the things you don't try.** While I was varying weight precision across Q4_HQQ, Q6_K, and Q8_0, I wasn't varying K vs V cache independently. The wrong theory frames the wrong search space. If I'd started with isolation instead of optimization, I would have found the bug in the first 10 minutes.

**Test at the boundaries, not the center.** `What is 2+2?` is a better diagnostic prompt than `Explain quantum computing in 3 bullet points.` Short prompts with unambiguous correct answers make failure modes obvious. Long prompts with subjective quality create ambiguity that lets you talk yourself into "moderate quality" when the output is actually broken.

---

*Bug reported to the llama.cpp TurboQuant discussion (ggml-org/llama.cpp#20969). The TQ3_0 CPU implementation is from Aaryan-Kapoor's turboquant-tq3_0 branch. Benchmarks on Qwen3-1.7B (Qwen/Qwen3-1.7B-GGUF), Qualcomm Snapdragon AArch64, Android.*
