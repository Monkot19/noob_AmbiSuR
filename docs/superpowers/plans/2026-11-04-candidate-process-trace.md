# Candidate Process Trace Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. Preserve D workspace and reviewed prior artifacts.

**Goal:** Observe one unchanged native7K→30K position-check run, separating anchor/proposal/action/immediate surface/late coverage.
**Architecture:** Standalone runpy launcher plus scoped Python return observer; extract/reuse the actual static-record constructor. Original training and algorithm files remain byte-identical.
**Tech Stack:** Python3.10+, existing PyTorch/NumPy, existing metadata and schema1.
**Spec:** docs/superpowers/specs/2026-11-04-candidate-process-trace-design.md

## Constraints and review focus

No GT/server actions/new loss/LR/RNG-based sampling. No writes to historical outputs. Record schema1 joint ALR acceptance separately from actual move subset, including empty/late events. Copy arrays before inplace movement. Reference main render is already post-move; fixed probe is readonly, A-only, no selector invocation. Restore profiler on failure; metadata records actual dirty/clean state. Ordinary train.py must not import trace.

### Task 1: shared observer record and real CPU trace events
- [ ] Add failing cases to existing tests/test_candidate_selector.py for trace schedule, shared true selector observation, move subset/empty, copies, no producer RNG/Parameter mutation.
- [ ] Extract selection_record(model,reference,depth,result,captured) in scripts/evaluate_candidate_actions.py; unchanged old observer delegates to it.
- [ ] Implement CandidateTrainingTrace in scripts/trace_candidate_training.py: handlers for actual spread/core/helper return; versioned schema1 record plus auxiliary NPZ and event JSON.
- [ ] Run real CPU tests RED→GREEN; confirm existing static record regression remains green.

### Task 2: original training launcher and readonly fixed probe
- [ ] Add failing cases for profile context restoration/normal and exceptional exits, actual training-helper ordering with CPU renderer/model surrogate, probe never moves/optimizes and late empty move meaning.
- [ ] Implement parse launcher flags before `--`, runpy original train.py, metadata and final manifest; no production hooks. Fixed-view render uses original render with A-only labels/no_grad and does not select or move.
- [ ] Run unified CPU suite and existing native skips; source scope verifies production files identical6172bc45. No local native-render claim.
- [ ] Independent whole-change review; address material issues; fixed commit/push and launch guide with original parameters/newview/output. User performs server run, no extra smoke/baseline precondition.
