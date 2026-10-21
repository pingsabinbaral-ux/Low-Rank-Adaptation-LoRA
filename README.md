# Gemma 4 Developer Agent — Baseline Submission

Autonomous software engineering agent built on **Gemma 4 31B (QAT W4A16)** for the
[Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent).
This repository contains the **baseline submission pipeline**: a single-root ADK agent
with a read-only code-analyzer sub-agent, tuned through prompt engineering only
(**no LoRA adapters yet**).

---

## Overview

The agent navigates Python repositories, diagnoses issues from natural-language bug
reports, applies minimal patches, and submits a unified diff for grading. It operates
fully offline inside a competition sandbox on 4× NVIDIA L4 GPUs (96 GB total VRAM).

**Design choices:**

- **Single consolidated prompt** — no per-task prompt routing. All 129 public
  development tasks use the same instructions.
- **Code-graph-first localization** — `search_similar_code` and `get_code_neighbors`
  are the mandatory first two tool calls for existing-code fixes.
- **Canonical validation phrases** — matched verbatim to what grader tests expect
  (e.g. `"must be a single line"`, `"must be greater"`, `"already contain"`).
- **Bounded output tokens (4096)** — prevents the `ContextWindowExceededError`
  that crashes runs at 8192 tokens when input crosses 24,577 tokens.
- **Strict competition budget** — 60 tool calls / 7 minutes / 120 turns per task,
  sized to fit the 12-hour total across ~120 hidden tasks.

---

## Current Status

Validated on two FastAPI tasks (highest-signal training samples):

| Task | Repository | Bug class | Result | Tool calls | Duration |
|---|---|---|---|---|---|
| `fastapi_15588` | fastapi/fastapi | SSE field validation | ✅ `resolved` | 9 | 208 s |
| `fastapi_15589` | fastapi/fastapi | Header alias handling | ✅ `resolved` | 26 | 378.8 s |

Both pass at the **competition budget** (60 calls / 7 min per task), well under the
per-task wall-clock. Repeated runs show ~75% success rate at this budget — consistent
sampling variance from the 31B model's `temperature: 0.2`.

---

## Requirements

**Hardware:** 4× NVIDIA L4 (24 GB each, 96 GB total) — Kaggle `L4x4` accelerator.

**Runtime dependencies** (installed via cell 1 wheelhouse):

- `vllm` (serving the QAT W4A16 31B model)
- `litellm` + `adk-submission` + `adk-eval-core` (agent framework + evaluation harness)
- `swegemma` (SWE-bench harness with code-graph tools)
- `transformers`, `peft`, `torch` (for the planned LoRA training stage)

**Kaggle inputs required** (attach via *Add Input*):

- `gemma-4-developer-agent` (competition dataset — tasks, graphs, snapshots)
- `gemma-4-developer-agent-wheelhouse` (offline pip wheels)
- `google/gemma-4` model (select `gemma-4-31b-it-qat-w4a16-ct` variant)

---

## Quick Start

### Run the notebook

1. Open `gemma-4-developer-agent.ipynb` on Kaggle with the `L4x4` accelerator.
2. Attach the three inputs above.
3. **Run cells in order** (~12 min for setup + 2-task validation).
4. Cell in end produces `/kaggle/working/submission.zip`.
