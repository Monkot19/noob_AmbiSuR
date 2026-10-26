# Native C0 checkpoint consumption implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. Follow the already approved native-C0 contract;do not change baselinebranch or add GPU smoke.

**Goal:** Candidate runs restore C0 Gaussian/Adam state and the Appweights from the checkpoint's own run/exactiteration,not an empty destination/latest30K.

**Architecture:** Retain the existing native capture/restore and train checkpoint branch. Only fix AppModel.load_weights arguments. Extend the existing unified test file with a real Torch serialization/Adam roundtrip using actual productioncapture/restore/Appmethods and the actualtrain startupblock. No new productionmodule/runner.

**Tech Stack:** ExistingPython/Torch/unittest/AST extraction pattern.

**Spec:** docs/superpowers/specs/2026-10-26-shared-7000-prefix.md

## Global constraints
- C0branch exactlyd6f15c8;baseline/source/data/outputs read-only. D stayssecond-run.
- Native tuple and existingcheckpointCLI unchanged;RNG/cameraqueue/Appoptimizer omissions remain disclosed.
- Candidate schedule total30000,checkpointiteration7000;candidatealgorithm/threshold/loss unchanged.

## Review focus
- Parent baseline may now have both7K/30KAppweights:load7Kexplicitly.
- Newcandidate output is different fromcheckpoint parent:neverloaditsAppweights.
- RestoredGaussianAdam must preserve nonzero moments/step and yield matchingnextCPUupdate.
- Native17-itemproducer/capture and realstartupconsumer must be exercised,not a handwrittencapturetuple or mockonly.
- Baselinebranch/scene/app/CUDA files must remain unchanged;no defaultpath regression.

## One bounded task
- [x] Add one focused case to tests/test_candidate_selector.py;observeRED against wrongcurrentdestination/latestApp loading.
- [x] Change only AppModel.load_weights source/iteration in train.py checkpointbranch.
- [x] FocusedGREEN thenexistingunifiedsuite;24run/19CPU pass/5native skipped in2.293s,no standaloneGPUtest/newdependencies.
- [x] One scopedreview foundnoactionableissues;reviewer independently passed focusedCPUcase andverifiedC0tree.
- [ ] Commit/push and exactversionhandoff after userreturns realcheckpoint/exitcode receipt.

Currentbaseline receipt:full30K log/Trainingcomplete,50:21,PSNR31.072280120849612,L10.016401121392846107,N1291587;exitcode0/nativecheckpoint1074511797B/7Kapp14349B/30KPLY320315108B confirmed. No finalgeometrybenefitclaim.
