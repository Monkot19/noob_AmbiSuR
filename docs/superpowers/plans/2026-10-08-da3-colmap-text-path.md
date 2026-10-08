# DA3 COLMAP text-path compatibility repair

User-approved scope: discover sparse/0 text models, without changing alignment,
RANSAC, weights, data, training or GT boundaries.

1. Add real filesystem tests for sparse/0 text, production COLMAP loading,
   unchanged source bytes, existing layouts, binary precedence and missing input.
2. Observe RED for the unsupported text layout before modifying production.
3. Add only the missing text lookup after the existing zero-directory binary lookup.
4. Run focused and Utility regressions, static checks and available full discovery.
5. Obtain fresh-context review; commit and push the bounded repair.
6. Qualify the exact commit on AutoDL before creating a replacement confirmation.

Existing confirmation records remain immutable. This repair does not start DA3,
create staging, read Utility GT or authorize Utility training/C1.
