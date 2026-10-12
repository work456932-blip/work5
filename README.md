# SceneRepair

Anonymous code for *From Reference Frames to Wrong Answers: Causal Root Localization and Selective Repair for Language-Grounded Spatial Reasoning*.

SceneRepair is a label-free test-time framework for spatial reasoning with frozen vision-language models. It represents a model rationale as a typed dependency graph, uses structured interventions to locate the earliest consequential unsupported state, and selectively repairs its dependent reasoning steps through a lower-confidence-bound gate. The framework supports targeted, auditable correction across MindEditBench, ViewSpatial, and MultihopSpatial while reducing harmful answer revisions.
