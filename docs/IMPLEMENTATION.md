# Specified defaults and reference conventions

## Settings explicitly stated in Table 12

| Setting | Value |
|---|---:|
| Generated trajectories / maximum states | 3 / 8 |
| Candidate nodes / repair candidates | 4 / 3 |
| Equivalent realizations | 3 |
| Symbolic / visual critic weights | 0.35 / 0.65 |
| Consistency / necessity / sufficiency / root thresholds | 0.58 / 0.08 / 0.04 / 0.18 |
| Competitive root margin | 0.03 |
| Candidate objective consistency / probability / similarity weights | 0.55 / 0.25 / 0.20 |
| Acceptance repeats / one-sided confidence | 3 / 90% |
| Acceptance margin | 0.02 |

The manuscript supplies a description, not the original source code or complete
prompt text. The following choices make that description concrete without
representing unstated values as original experiment settings.

## Underspecified choices

- Necessity coefficients default to (0.5, 0.5); sufficiency coefficients to
  (1/3, 1/3, 1/3); geometric root weights to equal weights. They are configurable.
- JS divergence uses natural logarithms. Confidence is maximum option probability.
- Family mass is the sum of N + U over all valid paired realizations. Family
  N/U/Q averages are weighted by normalized mass; zero effect mass gives zero.
- Q is modal answer agreement multiplied by exp(-4 times the sum of variances
  of JS effects and signed consistency effects). Destructive/corrective stability
  are averaged. At least two valid equivalent realizations are required. Q has
  no positive manuscript-specific threshold; Q must be nonzero by default.
- Similarity is mean per-state SequenceMatcher similarity of structured state
  records. Identical immutable states receive similarity 1.
- Eq. (8) uses the standard normal one-sided quantile z(0.90), and sample SD
  with ddof=1. Original and candidate verifier seeds are independent; candidate-0
  draws are shared across candidates to preserve paired gain estimation.
- Multiple competitive non-ancestral roots cause abstention (Appendix B);
  otherwise shallowest depth, descending R, then index break deterministic ties.
- Symbolic/visual disagreement is retained. A difference above 0.25 makes the
  diagnostic `high_confidence` flag false. Root selection still records its
  selected hypothesis; candidate verification and acceptance decide revision.
  This tolerance is a configurable reference convention, not a reported setting.
- Conditional diagnosis skips probes when no nonterminal state is below tau_C.
  The top-K-plus-threshold screen applies once diagnosis activates.
- Symmetry and inverse checks reject explicit contradictions; they do not
  require a redundant inverse statement for every relation.
- `no_graph` removes competitive ancestry and regenerates the complete trace.
  `earliest_root` removes the R competition band and picks the shallowest passing
  root. These are explicit conventions for otherwise unspecified ablation details.
- Temperature 0.7, output cap 4096, image maximum side 1024, bootstrap repeats
  2000, low-resolution control 256, and integer seeds 42/43/44 are reference
  settings. Freeze them before comparisons; replace them with original settings
  if those become available.
- If a declared closure does not include the terminal answer state, it cannot
  change the final distribution or target grounding. Such a repair is rejected
  when it attempts an unauthorized final-answer change.

## Labels, malformed outputs and failure handling

Examples expose only id, question, images, options and task group. Additional
serialized fields, including answer and annotations, are dropped. The inference
CLI has no label argument. Evaluation labels and human/synthetic root annotations
are read only by the evaluator after prediction files exist.

Each rejected probe retains its validity status; invalid samples contribute no
effect or stability mass. Invalid candidate closures are discarded. Failure to
score a baseline/candidate does not create an optimistic gain. A failed acceptance
reference preserves the trace. All typed trajectories failing invokes a direct
answer request; failure of that request is surfaced rather than invented.

## Cost and results

Usage fields contain measured physical backend requests and token counts.
They are not set to numbers reported in the manuscript. Server-side batching,
token-probability scoring, and deployment determine actual latency and usage.
No reported accuracy, localization, latency or token figures are inserted into
prediction/evaluation outputs. Supplied results would need their original
example-level predictions and execution environment to be reproduced.
