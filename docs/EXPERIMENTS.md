# Evaluation and interfaces

## Benchmarks

- MindEdit-Bench: map the six released groups to L1, L2, L3a, L3b, L4 and L5.
- ViewSpatial: map camera/person partitions to stable group identifiers.
- MultihopSpatial: use hop-specific groups; supply normalized target boxes and
  target view IDs in the evaluation-only labels file for grounded accuracy.

No dataset is included or downloaded. A dataset-specific converter must use the
actual release schema to populate the generic inputs. Keep label files separate.
An image path is local, not a URL. Optional paths are resolved against image_root.

## Accuracy and statistical units

`accuracy` is an equal-group macro mean; `micro_accuracy` is example accuracy.
Use groups matching the benchmark's official aggregation. Group means reproduce
the MindEdit table's six-group raw-average definition. Chance-normalized accuracy
uses each question's option count. Grounded answers require correct answer,
matching view and IoU >= 0.5; convert native boxes to normalized xyxy beforehand.

W->C and C->W are fractions of ALL evaluated examples, aggregated with the same
group macro convention. Their difference equals the macro accuracy gain relative
to the separately generated Direct predictions. The unchanged typed trajectory
is candidate 0, but it is not silently substituted for Direct in this comparison.

Bootstrap draws are paired by example, stratified by group and recomputed as
macro gains; returned intervals are 95% percentile intervals. Exact McNemar
uses pooled discordant paired predictions and reports both counts. Repeat the
entire inference/evaluation process for three seeds and report mean/sample SD.
Do not treat the same example under three seeds as three independent examples.

## Localization and annotation

Annotations may identify a particular generated `trajectory_index`; otherwise
the winning representative trajectory is evaluated. Ambiguous human cases are
reported and excluded. Exact and within-one metrics count abstention as failure.
Distance is shortest undirected declared-graph distance; abstention/disconnected
roots receive a graph-size penalty. This distance convention is explicit because
the manuscript does not specify its implementation.

When symptom_indices are available, root-versus-symptom discrimination separately
applies component passing and competitive ancestry to each annotated root/symptom
pair. It does not duplicate global exact-root accuracy. Pair construction and
distance conventions must be held fixed across compared localizers. Actual
human annotation and adjudication are external inputs, not simulated labels.

Pass `--typed-trajectories path.jsonl` for diagnosis of preconstructed corrupted
traces. Each record contains only `id` and `trajectories` (a list of standard
trajectory objects); the CLI never forwards root annotations to the model.
Use `--localization-controls` with root annotations to evaluate saved-trace
uncertainty/anomaly/necessity/graph-free paired localization controls separately.

The synthetic interface takes an independently validated clean trajectory and a
specified unsupported corruption. It preserves preceding states, propagates the
corruption only through declared descendants, and returns a separate annotation.
Later descendant errors require annotation to distinguish symptoms from unrelated
errors; the code does not automatically mark every descendant as a symptom.

## Safety and coverage

The evaluator reports apply rate, accepted-answer error risk, ECE, C->W harm and
no-harm among applied originally-correct examples with explicit denominators.
The manuscript does not define the denominator of its named no-harm metric;
this implementation avoids claiming its value uses an identical convention.

Margin sweeps replay stored candidate gains and recompute trajectory voting at
each threshold. They use the maximum-LCB candidate and consistency guard; they
do not merely filter final predictions from the default threshold. ECE uses ten
equal-width confidence bins. Final vote confidence is empirical weighted answer
agreement; per-trajectory option confidence remains available in audit traces.

## Baselines and fairness

Direct, CoT, self-consistency, typed-state reranking and full regeneration are
implemented as common-interface controls. Full regeneration uses the same repair
and candidate-scoring prompts and settings but permits all states to be edited.
Third-party Patient-VoT, SceneGraph-CoT, SVPR, VisuoThink, Self-Refine, R3V, WMiW
and ESC need adapters to their published mechanisms before their names may appear
in experimental comparisons. `typed_rerank` is a generic typed-verifier control,
not a claim to reproduce WMiW.

Match model revision, view preprocessing, example order, option order,
temperature, output limit, number of generated candidates and scoring definition.
Generation budgets and inference cost are different: interventions/verifier calls
must also be counted. Token-matched experiments require recording actual total
output tokens, adjusting the generation policy on development examples and
freezing it. No unmeasured configuration is labeled token-matched by this package.

## Language, grounding, threshold selection and leakage

`controls.paraphrases` implements the five language families and returns validity
records. Human validation of held-out equivalence templates remains an explicit
external step. Visual controls operate on existing local files. Root-type
consistency is reported only for jointly diagnosed roots, with the pair count;
answer consistency covers every aligned example.

`calibration.select_on_development` accepts a dev-only runner and evaluator.
Write selected settings to a frozen config and reuse on external benchmarks.
Threshold replay refuses to use repair candidates generated for another root.
Prompts, intervention families and acceptance margin must also remain frozen.

Input loading drops all nonallowlisted fields. Offline boundary checks verify
that adding/removing/permuting input label fields leaves exposed inputs unchanged.
A full model-level label permutation audit can rerun from the same seeds and
compare persisted predictions with `scenerepair.audit.compare_predictions`.
