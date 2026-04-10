# I Built an AI That Remembers Its Own Mistakes. Here Is What Broke Anyway.

*M S Ramaseshan | April 2026*

---

There is a specific failure mode that nobody talks about when they describe using AI to write production code.

It is not hallucination. It is not bad syntax. It is not the model refusing to help. It is this: you ask an AI to help you solve a hard problem. It tries something. It fails. You end the session. You come back the next day, and the AI tries the exact same thing again — with identical confidence.

The model has no memory. It does not know it already tried this. It does not know this was a dead end. It will keep trying dead ends forever unless you, the human, remember to tell it not to.

For most software projects, this is an annoyance. For the kind of work I was doing — porting a large C codebase from x86 to ARM64, dealing with processor-specific behaviors, subtle ABI rules, and performance regressions that only show up on specific hardware — it is a serious problem. An incorrect approach does not just waste an hour. It can waste a week while you rule out everything else first.

I built a system to fix this. I called it Veda.

---

## The Problem: Six Weeks, Zero Memory

The project was ARM64 NEON porting for FFmpeg — a task that sounds simple until you are actually doing it.

FFmpeg is one of the most widely used open-source projects in the world. It is the software that encodes and decodes video on hundreds of millions of devices. Most of its performance-critical code is written using x86 SIMD intrinsics: processor-specific instructions that can process eight or sixteen values simultaneously. Those instructions do not exist on ARM processors. To get the same performance on AWS Graviton3 or Apple Silicon, you have to rewrite the hot paths using ARM's equivalent instruction set, called NEON.

This is painstaking work. Each function requires understanding both the original x86 code and the ARM64 calling conventions. ARM and x86 handle memory differently — a data race that x86 silently tolerates can corrupt state on ARM. Integer arguments passed to functions have different rules about their upper bits. Register preservation rules are different. And every new function must be verified correct *and* faster than the scalar fallback, or it is worse than useless.

Across six weeks of AI-assisted sessions, I was running into the same structural problem: each new session started with no knowledge of what the previous sessions had learned. If a particular approach to loop unrolling had caused a register corruption bug last Tuesday, the AI would invent it again next Monday. If a specific way of handling function arguments only showed up as broken when called through FFmpeg's test harness — not in direct testing — that subtlety would be re-discovered each time rather than remembered.

---

## Veda: Four Components

Veda is an attempt to give an AI system the institutional memory it structurally lacks. It has four parts.

**The Evidence Corpus.** A PostgreSQL database that stores every attempt ever made on the project — successes, failures, and partial results — with exact parameters. Not summaries. Not notes. Exact function names, exact flags, exact error messages, exact cycle counts. Every entry is tagged by domain (`neon`, `inline_assembly`, `memory_model`, `security`) and tiered by confidence:

- `ground_truth` (45 entries): verified facts. Do not contradict these.
- `candidate_evidence` (10 entries): things tried with known outcomes.
- `hypothesis` (2 entries): unverified starting points.
- `dead_end` (6 entries): confirmed failures. Hard-blocked.

**The Dispatch Engine.** Before attempting any non-trivial implementation, a proposal is submitted to a router (`dispatch/router.py`). The router checks the proposal against the corpus using tags and semantic similarity. It returns one of three verdicts: `ROUTE` (proceed, here are the relevant corpus entries), `BLOCKED` (this is a known dead end, stop), or `SKIP` (no relevant history, proceed with caution). Across six weeks, the router was called 91 times.

**Specialists.** Nine domain-expert prompts — covering areas like ARM intrinsics, AArch64 ABI, memory model safety, float precision, and security — that review each proposal before implementation begins. They return `APPROVE`, `REJECT`, or `MODIFY` with reasoning. A single `REJECT` stops the work.

**The Auditor.** After each attempt, a result block is emitted — a structured JSON record with the exact outcome, what worked, what failed, and which corpus tier the entry should be promoted to. The auditor ingests these and maintains the corpus. No result, no learning.

The intent: before the AI writes a single line of code, it checks what is already known. If this approach was tried and failed, it is blocked before any time is wasted. If it was tried and succeeded, the corpus entries tell it exactly how.

---

## The FFmpeg Work: What Actually Got Done

Over the course of the project, roughly thirty FFmpeg modules were ported from x86 SIMD to ARM64 NEON, validated with both correctness tests and performance benchmarks on AWS Graviton3.

Every ported function had to pass a three-phase gate before being committed:

1. **Hardware PMU backend check.** The benchmarking framework verifies it is measuring actual CPU cycles via the Linux Perf Monitoring API — not software timer ticks, which are not comparable across runs.

2. **Correctness check.** FFmpeg's own `checkasm` framework runs the NEON function against the C reference and compares outputs. No tolerance. An exact match is required.

3. **Performance gate.** The NEON function must run at least 0.80x as fast as the baseline. Below that threshold, the implementation is wrong or the approach is fundamentally unsuitable for NEON.

A sample of what passed:

| Module | What it does | Speedup |
|---|---|---|
| `scene_sad16` | Scene change detection | 28.8x |
| `pixelutils sad_32x32` | Pixel-level sum of absolute differences | 15.0x |
| `exrdsp reorder_pixels` | OpenEXR pixel format conversion | 10.7x |
| `llvidencdsp sub_median_pred` | Lossless video encoder prediction | 12.5x |
| `hevcdsp add_residual` | HEVC decoder residual add | up to 10.6x |
| `hevcdsp idct_16x16/32x32` | HEVC inverse DCT (depth 12) | 4.6–6.6x |
| `mpegvideoencdsp denoise_dct` | MPEG video encoder DCT denoising | 3.6x |
| `aacpsdsp hybrid_synthesis_deint` | AAC+ stereo deinterleave | 3.7x |
| `cavsdsp QPEL` | CAVS video motion compensation | 2.0–4.4x |

Some functions showed more modest gains — `h264chroma` motion compensation at 3.4–6.1x, `diracdsp` weighted overlay at 5.4–7.2x. A handful were deliberately left as C-only where the NEON overhead actually made them slower: `h263dsp` and `vp3dsp` horizontal loop filters both came in at 0.76–0.84x due to the cost of loading individual lanes from non-contiguous memory, below the 0.80x gate.

By the end, a full `checkasm` run showed that of roughly 85 testable modules, 75 had NEON paths — either from our branch or from upstream FFmpeg's existing ARM64 code. Ten modules remained C-only. The project was approximately 88% complete.

---

## The Bug That Veda's Memory Caught — This Week

The most instructive moment of the entire project happened in the session that became this post.

I resumed work after a session had been interrupted mid-implementation. The files in version control had two new functions written but never tested: `ff_ps_hybrid_synthesis_deint_neon` and `ff_ps_hybrid_analysis_ileave_neon` — routines that scatter stereo audio data from an interleaved format into separate left/right channel arrays, and vice versa.

The code built cleanly. I ran the test suite:

```
ps_hybrid_analysis_ileave_neon (fatal signal 11: Segmentation fault)
ps_hybrid_synthesis_deint_neon (fatal signal 11: Segmentation fault)
```

Both functions crashed immediately.

The bug was in these two lines, one in each function:

```asm
lsl  x4, x2, #2   // x4 = i * 4
lsl  x5, x2, #8   // x5 = i * 256
```

`x2` holds the function's third argument — an integer `i` representing a starting column index. In direct calls, integers arrive in 32-bit registers with the upper 32 bits cleanly zeroed. But FFmpeg's test framework calls functions through a trampoline that loads *all* arguments from a stack frame using 64-bit loads (`ldp`). The upper 32 bits of `x2` are whatever was in memory above the stack frame — garbage.

`lsl x4, x2, #8` shifts the full 64-bit value of `x2`. When `i = 3` and the upper 32 bits of `x2` are, say, `0xDEAD0000`, the result is `0x7AB600000000_0300` — a wildly wrong offset. Adding that to the output pointer and attempting a store produces the segfault.

Veda's corpus had this documented. The relevant ground truth entry, written after a similar bug was found in a different function months earlier:

> *AAPCS64 stack-passed int args have unspecified upper 32 bits when loaded through the checkasm trampoline via ldp. Always use cmp wN, wM for loop bounds and lsl wN, wN, #k for address arithmetic on int parameters — never lsl xN, xN, #k.*

The fix: two characters per line.

```asm
lsl  w4, w2, #2   // 32-bit shift → result stored in w4, zero-extended to x4
lsl  w5, w2, #8   // 32-bit shift → result stored in w5, zero-extended to x5
```

Writing to the 32-bit alias `wN` automatically zeros the upper 32 bits of `xN` in AArch64. The functions passed correctness tests immediately after.

Without the corpus, this bug would have been rediscovered from scratch — possibly after hours of confusion, because it only manifests through the test framework's trampoline, not in direct function calls. The corpus made it a two-minute fix.

---

## What Broke Down: The Audit Gap

The corpus is only as good as what gets written into it. This is where the system failed.

A Veda session works like this: propose → route → implement → test → emit result block → audit result into corpus. The last two steps are where the value gets captured. If the session ends before the result is audited, the learning is lost.

On April 10th, across multiple interrupted sessions, the routing log showed 27 proposals submitted. The attempt_result table showed one result audited. The other 26 — including several implementations that were committed to the repository — were never fed back to the corpus.

More critically: nine proposals were routed and produced no commit. Something happened — either the implementation failed, the tests failed, or the session ran out of quota mid-implementation. I do not know which, because no result block was ever submitted. Those nine potential dead ends are invisible to the corpus. If a future session tries the same approach on `vf_eq` or `vf_gblur`, the router will have no memory of why it did not work before.

The routing log, queried directly from the database, shows what was attempted:

```
11:43 — vf_eq NEON process function         → no commit
11:49 — af_afir complex multiply-accumulate → no commit
11:55 — vf_gblur postscale/verti slices     → no commit
12:04 — vf_hflip byte and short rows        → no commit
12:11 — vf_blackdetect count_pixels         → no commit
12:14 — idetdsp filter_line                 → no commit
12:23 — jpeg2000dsp rct_int and ict_float   → no commit
12:40 — v210 planar pack                    → no commit
12:56 — v210 planar unpack                  → no commit
```

These nine modules remain in an unknown state. The system did not fail to route them. It failed to record what happened next.

The lesson: **the audit step needs to happen before the commit, not after.** A session that runs out of time after committing but before auditing loses exactly the information the whole system was designed to preserve.

---

## What Veda Got Right

Despite the audit gap, the corpus did real work. The six dead-end entries alone saved time that is hard to quantify but easy to believe:

- Intel MKL does not exist on ARM64 and never will. Any session that tries to install it is blocked immediately.
- `_mm_movemask_epi8` has no single NEON equivalent. Proposals that try to find one are rejected before any code is written.
- `volatile` for inter-thread signaling is undefined behavior on ARM. x86's memory model hides this bug; ARM does not.
- `-mfpmath=sse` causes silent failures on ARM64. Dead-ended in the corpus, blocked on first occurrence.

The specialist review step caught structural issues before they became bugs. The testing gate — correctness then performance, in that order — prevented a category of mistake where fast-but-wrong NEON code gets committed and breaks downstream users silently. Every function that reached the commit gate was both correct and faster than the C reference.

And the aacpsdsp bug this week was the clearest demonstration of the core idea working as intended: the corpus remembered something the current session did not know, and surfaced it at exactly the right moment.

---

## The Honest Business Assessment

I built Veda because I thought it could become a product: a pipeline for porting performance-critical codebases to ARM64, with the corpus as a moat that competitors could not easily replicate.

I do not think that anymore.

The one-time transaction problem is real. You port a codebase once. The customer does not come back. Recurring revenue requires recurring work, and ARM64 porting is by nature a finite job. The first company to pay you ports FFmpeg. There is no second FFmpeg to port.

The corpus transferability problem is also real. Of 63 corpus entries, roughly 45 — the general ARM64 memory model rules, the dead ends on x86-only libraries, the float precision gotchas — apply to any codebase. The other 26 are specific to FFmpeg's internal patterns, its NEON assembly conventions, its checkasm framework. Those entries help nobody porting a database or a web server.

And the "just use Claude" problem is real. An experienced engineer with Claude and a few hours can figure out most of what Veda provides through general knowledge and careful prompting. The corpus is more efficient for repeated work on the same codebase. But there are almost no customers who will pay for efficient repeated work on the same codebase, because efficient repeated work on the same codebase is not a use case that most engineers face.

What Veda got right was the insight: **stateless AI systems lose information on complex, long-running projects, and that information loss has real costs.** What it got wrong was the business model wrapped around that insight. A tool that compensates for AI amnesia has value. A service that sells ARM64 porting using that tool has limited recurring revenue and a market that is already shrinking as ARM64 adoption matures.

---

## What the Experiment Taught Me

A few things worth keeping.

**The testing gate matters more than the AI.** The bench-validate harness — hardware PMU verification, correctness check, performance gate — is the part of this project I would take to any future project. The AI writes code. The harness proves the code is correct and fast. Removing either half degrades the other: without the AI, the harness has nothing to test; without the harness, you cannot trust what the AI wrote.

**Failure records are more valuable than success records.** The 45 ground truth entries in the corpus are useful. The 6 dead-end entries are essential. Knowing that MKL does not exist on ARM64 is worth more than knowing how to implement a particular NEON function, because the dead end would be re-attempted; the success would not. If I rebuild this system for a different application, I would invest disproportionately in capturing and indexing failures.

**The audit step is the system.** Everything else — the router, the specialists, the proposal format — exists to create clean, auditable records of what was tried and what happened. The value is in the corpus. The corpus is built by the auditor. If the auditor does not run, nothing else matters. This seems obvious in retrospect and was not obvious enough during the project.

**Completion without a use case is just completion.** The FFmpeg port is roughly 88% done. Finishing the remaining ten modules — `mpeg4videodsp`, `qpeldsp`, `vf_convolution`, a handful of others — would make it 100% done. I am not sure what that accomplishes. The working code exists, runs on Graviton3, and produces correct output faster than the C reference. The incremental value of the final 12% is close to zero without a specific reason to need those particular codecs optimized. Completion for its own sake is completion anxiety, not engineering judgment.

---

The code for the ARM64 branch is at [github.com/MadrasAI/FFmpeg](https://github.com/MadrasAI/FFmpeg) on the `arm64-port` branch. The Veda infrastructure lives at [github.com/MadrasAI](https://github.com/MadrasAI).

*All performance numbers are from on-device measurements on AWS Graviton3 using Linux Perf Monitoring API hardware counters. All corpus statistics are queried directly from the PostgreSQL database.*
