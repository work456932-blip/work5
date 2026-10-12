# Equation and algorithm map

| Manuscript | Implementation | Concrete behavior |
|---|---|---|
| Eq. (1), Sections 3/4.1, Appendix H | `schema.State`, `schema.Trajectory`, `prompts.STATE_SCHEMA` | All twelve required state fields; original option space; terminal answer exclusion |
| Declared edges / inferred context, Figure 6 | `graph.build_graphs`, `graph.DAG` | Declared DAG defines edits; cycle edges rejected; verifier-confirmed type priors only enter diagnostic graph |
| Eq. (2) | `critic.symbolic_score`, `critic.Critic.score` | Separate symbolic/VLM scores retained; 0.35/0.65 hybrid consistency; anomaly is 1 minus consistency |
| Section 4.2 shortlist | `diagnosis.screen` | Top four lowest-consistency nonterminal states plus every state below 0.58, conditional on suspicious evidence |
| Section 4.2 paired families | `interventions.FAMILIES`, `interventions.probe_node` | Target-specific referent, evidence, viewpoint, operation, scope, relation, visibility and view-dropout probes |
| Eq. (3) | `math_utils.descendant_consistency` | Mean consistency over the node and diagnostic descendants |
| Eq. (4) | `math_utils.necessity` | JS decision effect plus positive descendant-consistency loss |
| Eq. (5) | `math_utils.sufficiency` | Positive descendant recovery, local recovery and maximum-answer-confidence gain |
| Section 4.3 equivalent realizations | `math_utils.stability` | Modal answer agreement penalized by variation in decision/consistency effects |
| Section 4.3 effect-mass weights | `math_utils.aggregate_families` | Total valid effect mass per family; aggregation never takes the best single realization |
| Eq. (6) | `math_utils.root_score`, `diagnosis.select_root` | Weighted geometric aggregate; component thresholds; 0.03 competition margin; ancestor filtering; shallowest root; explicit abstention |
| Section 4.4 edit restriction | `repair.regenerate` | Only root and declared descendants returned; other states copied; types and prerequisites immutable |
| Eq. (7) | `repair.objective`, `repair.similarity` | 0.55 consistency + 0.25 max option probability + 0.20 trajectory similarity |
| Eq. (8) | `math_utils.lcb`, `repair.choose_repair` | Three paired verifier draws; one-sided normal 90% LCB; maximum LCB must exceed 0.02 and preserve mean consistency |
| Appendix B step 9 | `pipeline.weighted_vote` | Consistency-weighted hard-answer voting across accepted/preserved trajectories |
| Section 5.3 | `synthetic.inject_verified_corruption`, `metrics.localization_metrics` | External validated clean traces and human-root annotation interface; exact, within-one and graph-distance metrics |
| Section 5.4, Appendix D | `controls` | Five language families, equivalence checking, visual/ordering perturbations |
| Section 5.5 | `data`, `cli`, `metrics` | Inference input allowlist; separate labels; written predictions before evaluation; paired bootstrap and exact McNemar |
| Appendix A/G | `backends.ModelSession`, `config.Config` | Prompt/config hashes, model revision, separate seeds, usage, latency, parse events and audit records |
| Table 12 | `configs/default.yaml` | All listed numeric defaults preserved |

Section 4.4 and Appendix B permit preservation and abstention. A parse failure,
invalid intervention, ambiguous root or insufficient candidate LCB therefore
remains an explicit outcome, rather than producing a fabricated successful repair.
