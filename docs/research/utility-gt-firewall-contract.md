# Utility GT firewall: record transport (Task 6)

This is an engineering envelope for the approved two-release firewall, not a
geometry hypothesis, scientific approval, or experiment authorization. There is
currently **no real geometry release**. The earlier
`NO_SEMANTIC_REPAIR_JUSTIFIED` audit cannot release Utility GT.

## Interfaces and reuse

`authorize_first_gt_access(prior_record, geometry_record, access_log_path=...)`
consumes two exact identity handles `{path, sha256}`, not their in-memory JSON
contents. It reloads the existing prior-transfer confirmation with its
production schema validator and protected file reads. The caller must obtain the expected digests from reviewed
immutable records; these are integrity checks, not digital signatures or proof
of human approval.

Task 8 must separately bind all three qualified runs and obtain GT-probe
authorization, then call the firewall immediately before its mesh boundary.
Task 6 alone does not parse a mesh, evaluate a probe, authorize a run, or decide
that geometry research should terminate. It does not change M0/M1 or any solver,
fold, bootstrap, domain, or scientific gate.

## Geometry release envelope

The canonical JSON record has exactly these fields:

```text
schema_version: 1
kind: geometry_release
release_id: safe identifier
created_utc: timezone-aware UTC timestamp
repository: {root: absolute path, commit: exact 40-character SHA, clean: true}
stage: G-A | G-B | G-C
outcome: STAGE_G_C_CANDIDATE | NO_ACTION_SPECIFIC_SIGNAL
specifications: {charter: identity, stage_specification: identity}
evidence: nonempty list of identities
frozen_contract: candidate contract | null for termination
approval: identity of the separately reviewed approval record
candidate_admitted_to_utility: true only for the G-C candidate
utility_cannot_reopen: true
utility_gt_access: METADATA_ONLY
```

Every identity is `{path, sha256}`. At guarded authorization, referenced bytes must exist and hash-match;
the release itself also requires `<release-path>.sha256` containing the digest
and a newline (LF or CRLF). Record JSON uses sorted keys, two-space indentation,
one LF terminator, and disallows nonfinite numbers.

A candidate is admitted only at G-C. Its reviewed `frozen_contract` has exactly
the following nonempty sections (this envelope supplies no scientific values):
`action`, `cluster_unit`, `affected_parameters`, `outcome`, `horizon`, `formula`,
`constants`, `validity`, `state`, `topology_lineage`, `thresholds`, `metrics`,
`coverage`, `compute_budget`, `stop_rules`.

The separate approval JSON has `schema_version=1`,
`kind=geometry_release_approval`, and the release's exact `release_id`, `stage`,
`outcome`, `repository`, `specifications`, `evidence`,
`candidate_admitted_to_utility`, `utility_cannot_reopen`, `utility_gt_access`,
plus `frozen_contract_sha256` of the canonical contract bytes (including `null`
for termination). This binds later contract changes to a different review
identity; it cannot manufacture the required human approval. A real record may
only be created/reviewed under the separate geometry-stage approval process.

`validate_geometry_release` is shape-only and performs no reference reads.
`load_geometry_release` verifies only canonical release JSON and its detached
SHA, not its evidence/specification/approval files. Neither API grants access;
full reference verification occurs only in `authorize_first_gt_access` after
protected-path admission. The opened descriptor's filesystem identity is checked
before reading bytes, in addition to path and hardlink metadata checks.

## One-shot first access

Before returning a token the firewall checks both confirmations, their
references, chronology, project identity and the preregistered access-log path.
It refuses existing probe output/staging or any existing access log, including
an identical one. GT path aliases are rejected before opening reference files.
The access log contains both input identities, opaque GT metadata and release
outcome. It is published by an exclusive hard link from a flushed temporary
file, then inputs and output absence are rechecked. Unsupported hard-link
publication fails closed; no overwrite fallback exists.

A failure after publication leaves the consumed log intact. It must not be
deleted to retry; recovery requires an explicitly reviewed audit trail. Later
pipeline tasks must bind and verify this token, not recreate a first-access
record or treat a synthetic test fixture as an actual release.

## Post-token mesh admission (Task 7)

`audit_utility_mesh(mesh_path, source_root, confirmation, access_token=token)`
consumes that exact on-disk token. It reloads and verifies the log, both referenced
identities, chronology, the original confirmation and frozen source before any
mesh parse. A token is an integrity transport, not new human approval; Task 8
still owns completed-run binding and the separately authorized execution boundary.

The result is `UtilityMeshAdmission(outcome, reasons, summary)`: either `ADMITTED`
or `INCONCLUSIVE`. It contains no Gaussian mask, transformed mesh or repaired
domain. The existing full-surface loader/distance query are reused; all finite,
non-degenerate faces remain. Nonfinite vertices, identity/parse/alignment/coverage
failure stop admission. No fallback changes a coordinate frame or threshold.

The deterministic sparse-point encoding is unsigned 64-bit little-endian point
ID followed by three little-endian float64 world coordinates. Sort SHA256 digests
ascending, break a digest tie by point ID, and use at most 50,000 points. The
8x6 camera grid samples the equal-area full-frame cell centers
`((j+0.5)*width/8, (k+0.5)*height/6)` through the original intrinsics. W2C inversion
uses the established camera-ray calibration helper; all 147 cameras are used.
Only the 7,056 admission rays are batched against one full-surface Open3D scene;
this is not rendering, candidate selection, or a replacement statistical probe.

Mesh/source/access identities are checked again before returning `ADMITTED`.
Mesh and source filesystem identities (device, inode, size, nanosecond modification
and change timestamps) are retained across parsing and queries, not only across
each hash call. A transient write-and-restore or replacement therefore fails
closed even when the final content SHA matches the preregistered bytes.
Synthetic tests do not establish that the real Utility mesh is admitted. That
first parse remains blocked by the real geometry release and explicit probe
authorization, irrespective of code-test results.
