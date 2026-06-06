---
title: "ECU-Insight: Autonomous Vehicle Diagnostics with a Framework-Free Agent Loop"
author: "M S Ramaseshan"
date: 2026-06-06
---

ECU-Insight is an autonomous agent that diagnoses automotive Electronic Control Unit (ECU) faults by reasoning over live OBD-II data — signal streams, diagnostic trouble codes, and real-time sensor readings from a running engine. The agent navigates a 62-tool registry across 9 namespaces, orchestrates isolated subagents for focused circuit investigations, actively manages a context window that would otherwise collapse under the weight of live sensor payloads, and produces a prioritized repair plan grounded in evidence it collected and ranked itself.

The flagship evaluation: a 2020 Ford F-150 intermittent no-start with **zero stored DTCs**. No code to look up. ECU-Insight had to reason from live signals. It completed in **23 tool calls** against a reference budget of 40, survived **4 context compactions**, correctly identified a crankshaft position sensor dropout as root cause, and passed all 5 rubric dimensions.

This post is about the engineering decisions that made that possible — and the ones I had to make from scratch because no existing framework gave me the guarantees I needed.

The repo is at [github.com/ramaseshanms/Autonomous_ECU_Diagnostic_Agent](https://github.com/ramaseshanms/Autonomous_ECU_Diagnostic_Agent).

---

## The Domain Problem

Most people encounter OBD-II as that port under the dashboard that lights up at emissions inspection. What it actually represents is a standardized real-time telemetry bus that exposes hundreds of engine parameters at 10–40 Hz: crankshaft position, camshaft timing, manifold absolute pressure, fuel trim, injector pulse width, and more.

Modern ECUs run closed-loop control algorithms over these signals continuously. When something breaks — a sensor dropout, a marginal connector, a failing actuator — the control loop degrades before it fails completely. The intermittent no-start is the diagnostic nightmare: the vehicle runs fine on the lift, stores no fault codes, and returns with the same complaint three days later.

Human technicians address this with oscilloscopes and pattern recognition built over years of failure-mode exposure. The question I wanted to answer was whether a well-architected LLM agent could perform the same reasoning chain: capture signals, detect anomalies, correlate across channels, form hypotheses, rule them out systematically, and converge on a root cause — without a human in the loop.

That requires the agent to do several hard things simultaneously:

1. **Know what to look for** — fault trees, symptom patterns, DTC semantics
2. **Know how to look** — OBD Mode 01, UDS services, signal capture, statistical analysis
3. **Manage its attention** — 62 available tools is noise without a way to focus
4. **Preserve its reasoning** under context pressure — raw sensor payloads are large
5. **Delegate precisely** — circuit checks require fine-grained tool access, not everything
6. **Produce a verifiable, graded answer** — not just plausible text

These aren't soft requirements. Each one drove a hard architectural decision.

---

## Why I Built the Loop From Scratch

The first real decision was whether to use a framework. I evaluated LangGraph and LangChain. Both solve real problems. Both also impose abstractions that hide the exact behaviors you need to observe, test, and guarantee in a production diagnostic agent.

Specifically:

- **Context management** in most frameworks is either opaque or delegated to the model (summarization prompts with no deterministic behavior). For a diagnostic agent, silent context loss is a patient-safety-class defect — the agent might forget it already ruled out a hypothesis and re-investigate it, or worse, forget evidence and flip to the wrong conclusion.
- **Tool routing** is often middleware-based or chain-based. I needed O(1) lookup and a first-class search capability that could narrow 62 schemas to a relevant 5–8 per turn.
- **Subagent isolation** — scoped tool access enforced structurally, not by prompt — doesn't exist as a primitive in the major frameworks. You can approximate it with prompt instructions, but prompt-instructed scope boundaries are prompt-injectable.
- **Evaluation** requires deterministic replay without real LLM calls. Framework-wrapped agents make this significantly harder.

The cost of building in-house: approximately 300 lines of loop and context management code. The benefit: every critical property is readable, testable, and provable.

---

## The Tool Registry: O(1) Lookup, Generated Schemas, Scored Search

The registry is the architectural center of gravity. 62 tools. Nine namespaces: `obd`, `sensor`, `signal`, `knowledge`, `diag`, `report`, `session`, `uds`, `agent`. Each tool is a typed Python function decorated with `@tool`.

The `@tool` decorator is where the schema-generation decision lives:

```python
@tool("obd.read_pid", summary="Read a Mode 01 PID by number.")
async def read_pid(
    pid: Annotated[int, Field(ge=0, le=255)],
    transport: Annotated[ObdTransport, Injected],
) -> PidReading:
    ...
```

The decorator calls `get_type_hints()` on the function, partitions parameters into model-supplied and `Injected`, and calls `create_model()` to synthesize a Pydantic model. That model's `model_json_schema()` is the tool's advertised schema. The schema cannot drift from the implementation because it *is* the implementation's type annotations — changing the function's signature changes the schema automatically.

`Injected` parameters (transport, scratchpad, knowledge store) are physically excluded from the schema but forwarded at invocation time via a dependency bag. The agent never sees them; they're infrastructure. The tool signature makes this explicit rather than hiding it in a wrapper layer.

At invocation, `Tool.invoke()` validates model-supplied arguments through the synthesized Pydantic model before calling the function:

```python
async def invoke(self, arguments: Mapping[str, Any] | None = None, /, **injected: Any) -> Any:
    validated = self.input_model.model_validate(dict(arguments or {}))
    call_kwargs = {field: getattr(validated, field) for field in self.input_model.model_fields}
    for name in self.injected_params:
        call_kwargs[name] = injected[name]
    result = self.func(**call_kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
```

A `ValidationError` here is a first-class outcome: it means the model produced syntactically invalid tool arguments, which the eval harness records as `input_valid=False` against the `integrity` rubric dimension.

**The search problem:** Dumping all 62 schemas into every LLM request wastes input tokens and degrades tool-choice accuracy. The registry implements `search(query)` that scores each tool by relevance:

- Whole-word verb match: **5 points**
- Substring match anywhere in name: **3 points**
- Namespace match: **2 points**
- Summary match: **1 point**

At any given turn, the agent selects tools from a scored subset of roughly 5–10, not the full 62. This is both a token budget optimization and a precision improvement — a model that sees 10 relevant tools makes fewer irrelevant tool calls than one that sees 62.

The registry itself is a plain dict: `self._tools: dict[str, Tool]`. Resolution is O(1). There are no dispatch chains, no middleware stacks. Adding a tool is adding an entry. Nothing else needs to change.

---

## Context Engineering: The Part Nobody Talks About

The hardest engineering problem in long-horizon agents isn't tool calling. It's context management.

A single `signal.capture_stream` call can return several thousand characters of time-series data — timestamps, amplitude readings, metadata. Multiply that by three signal channels (crankshaft, camshaft, fuel pressure), add DTC lookups, fault tree retrievals, hypothesis records, and subagent results, and a multi-turn diagnostic session can easily exceed any practical context window within 10–15 tool calls.

The naive response is to increase the context window. That's not a solution. It's a budget deferral. The real problem is that most of the raw payload is not useful after the first read — the agent extracted what it needed and moved on. What needs to survive is the *interpretation*, not the data.

The `ContextManager` handles this with three interlocking mechanisms.

### Token Budget

The budget is configurable (`ECU_AGENT_LLM__CONTEXT_TOKEN_BUDGET`, default 180,000 tokens). Each turn, the manager estimates current token usage with a character-to-token heuristic (`ceil(chars / 4)`) and checks it against the budget before calling the model. A real tokenizer can swap in later without changing the budgeting interface.

For the flagship evaluation scenario, the budget was intentionally set to **2,000 tokens** — a tight cap designed to force the context manager to exercise compaction on a realistic multi-turn run.

### Compaction: Oldest-First, Never Silent

When the budget is exceeded, the context manager finds the oldest complete tool cycle — an assistant `tool_use` block paired with its `tool_result` — and replaces the raw payload with a one-line summary:

```
[compacted] signal.capture_stream result (2,847 chars) summarised; raw payload dropped.
```

The `tool_use`/`tool_result` message pair is preserved because the Messages API requires that every `tool_use` id has exactly one corresponding `tool_result`. Dropping either member corrupts the conversation structure and produces a 400 on the next turn. Compaction replaces the *content* of the result, not the message itself.

The critical invariant: **compaction never happens silently**. If the manager cannot reduce usage below budget by compacting all available cycles, it raises `ContextBudgetExceededError` rather than truncating. A silent truncation at the wrong position could drop an intermediate reasoning step and leave the agent with a structurally invalid transcript. The loud failure is the correct behavior.

In the flagship run: **4 compactions**, all recovered. The agent crossed the 2,000-token budget four times, each time compacting the oldest raw payload, and continued to the correct diagnosis.

### The Diagnostic Scratchpad

Compaction removes raw tool payloads. But the agent's working hypotheses, the evidence it tagged as significant, and the causes it ruled out cannot be dropped. These are the diagnostic state that accumulated over the whole session.

The `DiagnosticScratchpad` is a small typed structure — hypotheses with confidence scores, an evidence list, a ruled-out list — that is re-injected into the system prompt on **every turn**, fresh from the live Python object:

```python
class DiagnosticScratchpad:
    def render_compact(self) -> str:
        lines = ["[diagnostic-scratchpad]"]
        if self._hypotheses:
            lines.append("hypotheses:")
            lines.extend(
                f"  - ({h.confidence:.2f}) {h.statement}"
                for h in self.hypotheses
            )
        # evidence, ruled_out ...
        return "\n".join(lines)
```

Because the scratchpad is re-injected from the live Python object rather than being part of the compactable message history, it survives compaction completely. Even after four rounds of raw payload removal, the agent's reasoning state — `hypothesis: crankshaft position sensor (0.90)`, `ruled_out: fuel delivery` — is present at the top of every turn.

This is the architectural equivalent of a technician's notepad. The oscilloscope trace can be put away; the notes stay open on the bench.

---

## Subagent Orchestration: Isolation by Design

The flagship scenario requires a circuit-level investigation of the crankshaft sensor. That investigation needs access to low-level signal tools but should not have access to `report.build_repair_plan` or `session.end_session`. Those would be premature or dangerous to call from within a focused sub-investigation.

Most frameworks handle this with a system prompt instruction: "only use these tools." A prompt-instructed boundary is prompt-injectable.

The `SubagentRunner` enforces scope structurally. When the parent agent calls `agent.spawn_circuit_investigator`, the subagent is instantiated with a `ToolRegistry` built from a fixed scope — in this case, `{sensor.*, signal.*, obd.read_live_pid}`. The subagent physically cannot call tools outside that set because they are not in its registry. There is no prompt that can override this; the tools don't exist from the subagent's perspective.

The subagent runs its own complete agent loop with its own `ContextManager` and its own message history. The parent's transcript is invisible to it. Its own intermediate tool calls — in the flagship run, `signal.capture_stream` and `signal.detect_dropout` — never appear in the parent's transcript. From the parent's perspective, calling `agent.spawn_circuit_investigator` produces exactly one `tool_use` and one `tool_result`, the latter containing a typed return value.

That typed return — a `CircuitVerdict` with `healthy: False, confidence: 0.85` — is what the parent reasons over. The 2 intermediate calls the subagent made to arrive at that verdict are fully encapsulated.

**Why this matters beyond isolation:** It gives us *context economy*. A subagent with 2 internal tool calls contributes exactly one message pair to the parent's context. Without isolation, those 2 calls would appear inline in the parent's transcript, consuming context budget and adding noise to the parent's reasoning chain.

The subagent must produce a valid `CircuitVerdict` or the runner raises `SubagentFailedError`. There is no "just return some text" escape hatch. The typed contract between parent and subagent is enforced by Pydantic validation at the runner boundary.

---

## LLM Integration: Prompt Caching and the 8K Response Floor

The `AnthropicClient` is a typed wrapper over the Anthropic Messages API. Its primary responsibilities are schema translation, error mapping, and two specific optimizations.

**Prompt caching with ephemeral breakpoints:**

The tool schemas are stable within a session. The system prompt — which includes the re-injected scratchpad — changes every turn. These have different caching characteristics.

```python
# Mark the last tool schema with a cache breakpoint.
# Render order is tools → system → messages, so this covers all 62 schemas.
last = dict(api_tools[-1])
last["cache_control"] = {"type": "ephemeral"}
api_tools[-1] = cast("ToolUnionParam", last)

# System prompt gets its own ephemeral breakpoint.
api_system = cast(
    "list[TextBlockParam]",
    [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
)
```

An ephemeral breakpoint tells Anthropic's infrastructure to cache the prefix up to that point within the current session. Since the tool schemas don't change turn-to-turn, they benefit from cache hits. Since the system prompt includes a fresh scratchpad every turn, it is re-evaluated but benefits from the tool schema cache below it.

**The 8,192-token response floor:**

`DEFAULT_MAX_TOKENS = 8192`. This is not arbitrary. A complex assistant turn might produce multiple parallel `tool_use` blocks in a single response. If `max_tokens` is set too low — 2,048 is a common default — the response can be truncated mid-`tool_use` block. A truncated `tool_use` block has no corresponding `tool_result`, which corrupts the conversation history and produces a 400 error on the next API call. The 8K floor prevents this.

**Error taxonomy:**

SDK exceptions are mapped to a typed hierarchy before they leave `_invoke()`:
- `APITimeoutError` → `LLMTimeoutError` (retryable)
- `RateLimitError` → `LLMRateLimitError` with `retry_after_seconds` extracted from the `retry-after` header (retryable)
- `APIStatusError` → `LLMAPIError` with status code (non-retryable for 4xx)
- `AnthropicError` → `LLMAPIError` (catch-all)

Only `LLMRateLimitError` and `LLMTimeoutError` are in `_LLM_RETRYABLE`. The retry loop never touches 4xx responses. An invalid API key or a malformed request should propagate immediately, not be retried five times while burning quota.

---

## The Evaluation Framework: Graded, Reproducible, Agent-Blind

Most LLM agent evaluation is vibes-based: did the output look right? That's fine for demos. For a system you want to trust, you need a rubric with defined pass conditions and deterministic replay.

The eval framework has two layers.

**The scenario** is a YAML file with an objective, emulator configuration, token budget, reference call budget, and a `ground_truth` section:

```yaml
id: crank_intermittent
title: 2020 Ford F-150 3.5L - Intermittent No-Start (no DTCs)
context_token_budget: 2000
reference_call_budget: 40
long_scenario: true
expected_subagent: agent.spawn_circuit_investigator
ground_truth:
  root_cause: crankshaft position sensor
  related_dtc: P0335
```

The `ground_truth` section is read only by the rubric. The agent receives the `objective` and nothing else.

**The rubric** grades on five independent dimensions:

| Dimension | Pass Condition |
|---|---|
| **correctness** | Leading hypothesis matches `root_cause` by substring or ≥50% token overlap |
| **efficiency** | Tool calls ≤ reference budget; score = `budget / actual_calls` |
| **integrity** | 0 hallucinated tool names; 0 schema-invalid inputs |
| **coherence** | For `long_scenario`: budget crossed ≥1 time AND run completed |
| **subagent** | `expected_subagent` was invoked; 0 `SubagentFailedError`s |

The `coherence` dimension specifically tests the context management system. A long scenario with `long_scenario: true` is expected to cross the token budget; if it doesn't, either the run was trivially short or the budget was misconfigured. The agent must cross the budget *and* recover via compaction and continue to completion.

**Deterministic replay:** The framework includes a `FakeLLM` that serves pre-scripted `LLMResponse` sequences in order. The flagship run is captured as a scripted replay so the 5-dimension result can be verified in CI without an API key.

**The flagship scorecard:**

```
tool calls: 23   context compactions: 4   run: completed
leading hypothesis: intermittent crankshaft position sensor dropout

Scenario: crank_intermittent  [PASS]
  correctness  PASS  (1.00)  top hypothesis matches root cause 'crankshaft position sensor'
  efficiency   PASS  (1.73)  23 tool calls vs reference budget 40
  integrity    PASS  (1.00)  0 hallucinated, 0 schema-invalid tool call(s)
  coherence    PASS  (1.00)  budget crossed (4 compaction(s)); run completed
  subagent     PASS  (1.00)  agent.spawn_circuit_investigator invoked; 0 subagent errors
```

The efficiency score of 1.73 — 23 calls against a budget of 40 — reflects a diagnostic path that didn't pad. The agent read DTCs (none), captured three signal streams, detected crank dropouts while cam was steady, correlated them, spawned a targeted subagent, ruled out fuel delivery with live fuel trim data, and built a repair plan. That's the correct investigation path for this failure mode.

---

## The 23-Step Diagnostic Path

The flagship run is worth walking through because it demonstrates the reasoning chain, not just the result.

**Steps 1–3: Baseline.** Read DTCs — none stored. Read supported PIDs. Read vehicle info and VIN. No codes. The investigation must proceed from live signals.

**Steps 4–6: Signal capture.** Capture three streams: crankshaft RPM, camshaft timing, fuel pressure. Each returns a `SignalCapture` object with timestamped readings.

**Steps 7–8: Dropout detection.** Run `signal.detect_dropout` on the crankshaft capture. Detects gaps. Same analysis on camshaft: clean. Fuel pressure: steady.

**Steps 9–10: Statistics and correlation.** Compute signal statistics. Correlate crankshaft against camshaft — crank is the outlier.

**Steps 11–13: Hypothesis formation.** Search DTC knowledge base. Record hypotheses: crankshaft position sensor at 0.60 confidence, fuel delivery at 0.40.

**Step 14: Subagent delegation.** Spawn `agent.spawn_circuit_investigator` with scope `{sensor.*, signal.*, obd.read_live_pid}`. The subagent runs 2 internal tool calls and returns `CircuitVerdict(healthy=False, confidence=0.85)`.

**Steps 15–18: Fuel delivery ruling-out.** Read short-term and long-term fuel trim. Read live fuel pressure. Trims within normal band. Pressure stable.

**Steps 19–22: Evidence consolidation.** Add circuit-failure evidence (from subagent verdict). Add fuel-delivery evidence (ruled out). Update crank sensor confidence to 0.90. Rank hypotheses.

**Step 23: Finalization.** Look up P0335 fault tree. Build repair plan.

**Result:** Correct root cause. 23 tool calls. 4 context compactions survived. All 5 rubric dimensions pass.

The context management system is why this works end-to-end. By step 23, the raw signal payloads from steps 4–6 have been compacted away. What remains are the agent's derived conclusions: dropout detected, cam clean, fuel eliminated, crank circuit hardware-failed. The scratchpad carries those forward at the cost of a few dozen tokens, not thousands.

---

## Production Scaffolding

**Exponential backoff with jitter:**

```python
BackoffPolicy(
    base_seconds=0.1,
    cap_seconds=10.0,
    max_attempts=5,
    jitter=True,
    retryable=(LLMRateLimitError, LLMTimeoutError),
)
```

Delay is `base × 2^(attempt−1)`, clamped to `cap`, spread uniformly over `[0, delay]`. Jitter prevents thundering herd on rate-limit recovery. The `retry-after` header, when present, takes precedence over the computed delay.

**Token-bucket rate limiting:** A `TokenBucket.per_minute(60)` governs LLM requests. Async-safe with a lock, injected clock for deterministic tests. A separate bucket governs the transport socket — LLM saturation doesn't block sensor reads.

**Transport layer:** The `Elm327Transport` is an async TCP client for the ELM327 OBD-II adapter. It handles the initialization handshake (`ATZ`, `ATE0`, `ATL0`, `ATH0`, `ATS0`), response framing, and error typing. All transport failures are `TransportError` subclasses.

**Knowledge store:** An in-memory SQLite database built from curated JSON at startup — 13 fault trees, PID definitions, DTC descriptions sourced from public OBD-II references. The store backs the `knowledge.*` namespace. SQLite gives indexed full-text search without a network dependency.

**Structured logging:** Every log event carries a `run_id` via `structlog`. The `Tracer` protocol wraps operations in named spans; the default is a no-op, swapped for a real implementation by injection. Production and test agents share all logic except the tracer.

---

## Numbers

| Metric | Value |
|---|---|
| Tools | 62 across 9 namespaces |
| Default context budget | 180,000 tokens |
| Flagship context budget | 2,000 tokens (deliberately tight) |
| Flagship tool calls | 23 (efficiency score: 1.73) |
| Flagship compactions | 4 |
| Agent max steps | 12 |
| Subagent max steps | 8 |
| LLM max tokens per response | 8,192 |
| Rate limit (default) | 60 req/min |
| Backoff: max attempts | 5 |
| Backoff: cap delay | 10s |
| Fault trees in knowledge store | 13 |
| Test files | 26 (unit + integration) |
| Code coverage gate | 85% |

Aggregate pass rates across the full 6-scenario suite will be documented in a subsequent post as the scenario library expands. The flagship numbers above are the only ones I'm prepared to state precisely, because they're the only ones with a complete reproducible record.

---

## What This Is Actually About

An ECU diagnostic agent is a specific application. But the architectural decisions here — data-driven tool registries, typed subagent contracts, explicit context budgeting with loud failure, graded evals with deterministic replay — are domain-agnostic. They're the decisions that determine whether an LLM agent is an expensive demo or a deployable system.

The tension I keep returning to is between **expressiveness** and **auditability**. Frameworks maximize expressiveness at the cost of auditability. A 300-line bespoke loop gives you a system where every behavior is a line of code you can read, a test you can run, a rubric dimension you can grade.

For a diagnostic agent touching a vehicle that someone will drive, "the model probably got it right" is not an acceptable confidence level. Every tool call is validated against a generated schema. Every compaction is recorded. Every subagent result is typed and contract-enforced. Every hypothesis that contributed to the repair plan is traceable to the evidence that produced it.

That's not over-engineering. That's what "production" actually means for an agentic system operating in a consequential domain.
