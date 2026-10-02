---
title: "feat: ICU GEM — a clinically coherent, instruction-followable generative ICU model"
type: feat
status: active
date: 2026-09-25
---

# feat: ICU GEM build plan

## Purpose

Turn the from-scratch CLIF-native trunk into an **ICU-specific generative EHR model ("GEM")**:
a decoder that *generates* CLIF token trajectories which are clinically coherent, calibrated to
real event rates, and conditionable on structured clinical instructions — then improved by RL
post-training. This is the **InstructGPT recipe applied to ICU tokens**: base next-token model →
conditioning → reward-driven alignment, where the 2022 finding still holds — a small *aligned*
model beats a much larger raw one at doing what its user actually wants.

This plan supersedes nothing in the U-roadmap (`docs/plans/2026-09-01-001-...`): federation,
ablations, and scaling stay. The GEM is the *generative* track the Cliffordtron meeting roadmap
asked for (viewer → RL coherence → architecture updates → one focused use case).

## Evidence base (researched 2026-09-25, PubMed/arXiv via paperclip + web)

| Source | What it settles |
|---|---|
| Ouyang et al., InstructGPT (openai.com/index/instruction-following, 2022) | The alignment recipe: base NTP → SFT → reward → policy optimization; **pretraining-mix avoids the alignment tax**; alignment is <2% of pretraining compute — it *unlocks* existing capability |
| Xiao et al., arXiv:2609.12277 (Sep 2026) | RL over patient trajectories on ETHOS 9M/56M: **GRPO/DAPO with verifiable rewards**; naive SFT *underperforms* the PT backbone; 9M+RL beats 56M PT (AUROC); ECE 0.082→0.033; **rollout filtering** (exclude inconclusive rollouts, filter overflow/follow-up-poor anchors); balanced subsampling vs reward hacking; multi-task positive transfer |
| Pellegrini et al., EHR2Path, arXiv:2506.04831 (v2 2026) | ICU/ED/ward **pathway generation** by next-state prediction rolled out iteratively; **noisy Length-of-Stay indicator** makes simulated stays terminate; hourly aggregation is a workable event clock |
| Adam et al., SMB world model, arXiv:2601.22128 (Jan 2026) | AR training lets the encoder **defer trajectory reasoning to decode time**; latent (JEPA-style) prediction of masked future improves dynamics (curriculum SFT→JEPA: 0.731 AUC disease progression) — grounds the sandwich arm, as latent/representation objectives rather than token reconstruction |
| Chandak et al., EveryQuery, arXiv:2603.07900 (May 2026) | **Query/task conditioning** via a prepended token reshapes representations; beats AR rollout inference on 82% of 39 tasks and is **prevalence-invariant** (AR rollout estimates degrade on rare events, ρ=0.64 with prevalence) — conditioning evidence + a caution for rollout-based scoring |
| Yu et al., DAPO (NeurIPS 2025); Shao et al., GRPO/DeepSeekMath | Long-rollout stabilizations: **token-level loss, clip-higher, dynamic sampling**; no critic needed; group-relative advantages are within the optimal policy-gradient class |
| arXiv:2605.26194 (May 2026) | Local-Completion infilling objective validated for clinical time series (Arm B evidence) |
| Pickard et al., EHR-MPC (2026) | **Inference-time control** over a generative patient model for treatment decisions (sepsis) — the counterfactual / digital-twin mechanism that never trains treatment targets |
| Zhang et al., scaling laws, arXiv:2505.22964; SCOPE/REACH arXiv:2602.03730 | ~28M saturation on MIMIC-scale data; RL beats data-constrained scaling; rare-event rollout variance (supports EveryQuery caution) |

## Repo audit (2026-09-25) — starting state

**Solid:** training engine (DDP/bf16/compile/resume, fail-closed), packing + document-isolated
varlen attention, clinical-segment bins + deciles + soft discretization, TRIPOD+AI panel,
release-trust + federation E2E, 389 + 32 data-free tests green, token-sequence viewer
(`src/viewer/sequence_viewer.py`).

**Missing for a GEM (all resolved by G0 or scheduled below):**
1. No inference path for the from-scratch trunk — no `generate()`, KV cache, sampler, or decode.
2. No RL/reward hooks anywhere in `src/`.
3. No instruction conditioning; no prefix-LM (infilling) attention support.
4. No generative evals (perplexity, event-rate calibration, key-event recall, distance-to-observed).
5. No bridge packing our canonical fused-token shards (clinical-segment bins) into 8192-row packs for the dense trunk.

**Bugs found and fixed in G0:** inert NTP warmup in `joint_pretrain.py` (frozen-toggle → no trunk
gradients), `"total": None` crash in `run_arm.py`, duplicated (shadowed) `build_clinical_segment_bins`
in `tokenize.py`, stale `CLIF_DATA_DIR` path in AGENTS.md (tables live at `~/Data/clif-source/`),
dead `heads.*.weight` config keys (now wired — enables a pure-NTP run without code edits).

**Flagged, not fixed in G0:** (1) `notes_encoder.py` rebuilds its MLP lazily *after* construction, so
an optimizer created earlier never sees those params; (2) the untied 10k-vocab model measures
**45.2M, not "~30M"** — resolve (tied embeddings or vocab trim) before the size claim; (3) TextCode
and continuous-fused tokenization arms are stubs; (4) the P/F Berlin forced edge is unimplementable
as configured (`pao2_fio2` is not in `target_concepts`); (5) SCOPE/REACH rare-event estimators are
cited in notes but unimplemented — EveryQuery shows rollout-based probability estimates degrade on
rare events (ρ=0.64 with prevalence), and ICU danger events are rare events.

## Locked GEM decisions (change only with new evidence)

- **D-G1 — Base objective: pure NTP.** The generative base trains on next-token prediction alone
  (`heads.*.weight`: ntp=1.0, others=0.0 — now config-driven). Rationale: every successful
  RL-over-trajectories result starts from a pure NTP backbone; mixing TTE losses muddies the token
  distribution RL later shapes; InstructGPT: post-training unlocks, it doesn't install. The
  ORA/TTE objective remains the *prediction* arm (heads attach to a trained trunk — unchanged).
- **D-G2 — RL recipe: GRPO + DAPO-style stabilizations, verifiable rewards.** No learned reward
  model, no PPO critic. Rewards are computable per rollout: (1) rank by distance to the observed
  continuation (the "sample K, rank" idea == GRPO group normalization), (2) key-event presence
  (hard endpoints: ICU admission, death, vasopressor start, extubation), (3) population event-rate
  calibration (e.g. % post-extubation HFNC matches the real rate). Guardrails: Xiao-style rollout
  filtering (exclude inconclusive rollouts from advantage normalization; filter anchors whose
  outcome window exceeds the token budget), balanced label sampling, and InstructGPT
  pretraining-mix (small NTP term during RL).
- **D-G3 — Conditioning: structured prompt-prefix from existing vocab.** Instructions render as
  prefix tokens we already have (demographics/elixhauser/anchor-state tokens; narratives already
  begin this way). Dedicated instruction/control tokens are a later *deliberate* decision — they
  touch the frozen-vocab federation rule (hard rule 2).
- **D-G4 — Sandwich/infilling is Arm B, not the base.** Needs prefix-LM attention support in the
  trunk (G7). SMB/Local-Completion evidence supports it as an ablation against the NTP base.

## Units

| Unit | Deliverable | Entry gate | Verification |
|---|---|---|---|
| **G0 (this change)** | Audit fixes + generation stack (`src/model/generate.py`: KV-cached sampling, decode, simulations parquet writer, CLI) + this plan | none (data-free) | new unit tests + full suites green |
| **G1 — data** | Run tokenization ETL on staged MIMIC (`~/Data/clif-source/`), pack shards to 8192 rows, inspect real token sequences in the viewer | MIMIC staged (✓ this Mac) | ETL stats + viewer screenshots; vocab hash recorded |
| **G2 — GEM base** | Pure-NTP pretraining of the from-scratch trunk on MIMIC 2.1 (L40 box), with an EHR2Path-style **noisy Length-of-Stay indicator** so simulated stays terminate | G1; L40 reboot (driver) | val perplexity plateau; sample rollouts legible in viewer; rollouts terminate |
| **G3 — generative evals** | Perplexity panel, event-rate calibration, key-event recall, distance-to-observed; wired into eval + viewer | G2 checkpoint | report on held-out stays; baseline table pre-RL |
| **G4 — conditioning** | Instruction-prefix rollout CLI (cohort/anchor/horizon prompts from existing tokens) | G2 | instruction-following spot-checks; rate calibration per stratum |
| **G5 — RL post-training** | GRPO (+DAPO stabilizations) with D-G2 rewards and guardrails, 2× L40, compute-scoped. Treat the L1 rank reward as a **noisy verifier**: Xiao-style balanced label subsampling + exploit-surface monitoring, not just the reward curve | G3; G4 | coherence + calibration metrics improve vs G3 baseline; no perplexity collapse (pretraining-mix check) |
| **G6 — use case** | Post-extubation device decision (HFNC vs NC vs none): calibrated generated rates + decision-curve panel, obtained by **inference-time steering** (EHR-MPC-style context forcing at rollout time — treatments remain inputs, never targets) | G5 | preprint-ready figure set |
| **G7 — Arm B** | Prefix-LM masks in trunk + sandwich infilling objective; compare vs base | G2 checkpoint | ablation table (plausibility + downstream) |

## Hard rules (unchanged, restated for the GEM track)

1. **Treatments are model inputs, NEVER prediction targets.** In RL terms: reward *realism* of
   generated treatment patterns (calibration to observed rates), never reward the model for
   *choosing* treatments as outcomes — and never condition rewards on treatment targets.
2. Frozen mCIDE vocab across sites; no cross-site raw pooling.
3. Only pre-anchor notes are features; retrospective reports are label sources only.
4. `storetime` ordering; no look-ahead.
5. No data leaves its node; MIMIC stays PhysioNet/DUA-compliant (local or governed tenant only).

## Sources

- OpenAI, "Aligning language models to follow instructions" (InstructGPT), 2022.
- Xiao et al., "Reinforcement Learning over Patient Trajectories for Clinical Reasoning in EHR
  Foundation Models," arXiv:2609.12277 (ML4H 2026 submission).
- Pellegrini et al., "EHR2Path," arXiv:2506.04831v2.
- Adam et al., "The Patient is not a Moving Document" (SMB), arXiv:2601.22128.
- Chandak et al., "EveryQuery," arXiv:2603.07900.
- Shao et al., DeepSeekMath/GRPO; Yu et al., DAPO (NeurIPS 2025).
- Pickard et al., "EHR-MPC: Inference-time control for sepsis treatment with generative patient
  digital twins," 2026.
- arXiv:2605.26194 (Local Completion, clinical time series); arXiv:2505.22964 (EHR FM scaling laws);
  arXiv:2602.03730 (SCOPE/REACH).
