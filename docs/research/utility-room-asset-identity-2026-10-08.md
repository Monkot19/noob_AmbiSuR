# Utility Room asset identity — 2026-10-08

Status: user-operated AutoDL output recorded by the assistant; no direct server access.
This is an identity/provenance record, not a geometry admission or scientific result.

## GT metadata-only audit

- Path and resolved path: `/root/autodl-tmp/ambisur_data/gt/ScanNetpp/Utility_Room/mesh_aligned_0.05.ply`
- Bytes: `24490351`
- SHA256: `213dbdfff9ba992000039533463e4fcd941708d9fd8fe077df53a8495b63cd75`
- Audit: regular, nonempty, non-symlink file; device/inode/size/mtime/ctime identical before and after SHA256 calculation; shell return code 0.
- File bytes were read solely for hashing. No PLY parsing, vertex/triangle inspection, visualization, distances, labels or evaluation occurred.
- Identity does not establish alignment, units, coverage or geometric correctness. Utility GT firewall remains closed to all semantic access.

## Frozen DA3 snapshot

- Preprocessing/qualification commit: `81c0acbe9761164ed58095cc3aea8b81790fb5c3`
- Source audit record: `/root/autodl-tmp/ambisur_diagnostics/Utility_Room/prior-transfer/utility_room_source_147_v1.json`
- Source audit record SHA256: `cd172697f1d5d831dd314dd1d85675777f6a95a7359735192a722cd883dd2b47`
- Source manifest SHA256: `8da6df49430e55223d3a563789786c6c1520101f4f7df13f343dce86dabab3a1`
- Snapshot root: `/root/autodl-tmp/ambisur_work/data_snapshots/utility_room_da3_147_20260930_v1/colmap_undistorted`
- Snapshot record: `/root/autodl-tmp/ambisur_diagnostics/Utility_Room/prior-transfer/utility_room_da3_snapshot_147_v1.json`
- Snapshot record file SHA256: `522bad824116e9910636eb734efe2e222b2d5978abc3d384a0bf5f51e30cb143`
- Internal snapshot SHA256: `307b176e41111af403a565db94cfe8ada0a7d739361e1e380dcfaca08f49fc22`
- Preprocessing v2 confirmation SHA256: `da3a8710a641d415ddbf420ac40a2604bc14f384ee4e51ee25ece3b0ad3b6efe`
- Preprocessing v2 runtime-binding SHA256: `6793743fba60b63770536e3914f5ce62b7eb52389a4931cc1a70b68f32a46447`
- Launcher exit code: 0; wall seconds: 87; end UTC: `2026-10-08T02:52:11.270759+00:00`.
- 147 images; finalizer admitted depth/confidence arrays and aligned models. RANSAC reported 143/147 inliers. These are preprocessing checks, not GT alignment validation.
- Preserve old v1/v2 records and this single snapshot. All three future seeds reuse the same snapshot; no DA3 regeneration authorized.

## Gate E authorization and boundaries

- User explicitly authorized Gate E after metadata-only GT audit.
- Canonical prior-transfer confirmation is pending server creation. It must precede every seed run/view/state/launcher/qualification and probe target.
- A later documentation-only commit may bind training/confirmation identity without rewriting the snapshot's original preprocessing commit.
- Seeds 0/1/2 training launches each require separate approval. GT parsing/evaluation also requires the frozen geometry release and explicit later approval.
- Evidence remains v4. Geometry repair Tasks 4–13 remain stopped; no five-state routing or C1 authorization.
