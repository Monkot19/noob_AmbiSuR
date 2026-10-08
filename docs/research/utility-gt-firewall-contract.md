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
