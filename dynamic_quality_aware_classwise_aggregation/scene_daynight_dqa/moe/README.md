# Scene-Daynight DQA-MoE

This folder is a new branch of the scene/day-night DQA experiments.  It tests a
MoE-inspired DQA policy without changing the YOLO architecture yet.

The first notebook uses a **checkpoint-level expert pool**:

- `K=4` expert checkpoints are initialized from the warmup model.
- Every round trains the same client updates as the 02 head-to-full schedule.
- DQA routes client updates into specialized experts:
  - expert 0: highway clients
  - expert 1: citystreet clients
  - expert 2: residential clients
  - expert 3: night/hard clients
- The deployable model is a soft mixture of the expert residuals followed by
  server repair.
- Final evaluation includes the deployable mixed model and the individual
  experts, so we can estimate whether a real router would be useful.

This is deliberately not a full architectural MoE implementation yet.  It is a
lower-risk pilot to answer whether preserving client/domain specialization is
more promising than collapsing everything into one DQA aggregate.

## 02 FedMox Post-hoc Five-loop Sweep

`notebooks/02_fedmox_posthoc_five_loop.ipynb` is the follow-up after the first
manual checkpoint-level MoE did not produce a clear gain.

It reuses the already trained `scene_daynight_dqa/output/02_head_to_full_long_dqa`
checkpoints and evaluates FedMox-inspired candidate checkpoints without another
full FL run:

- FedMox-style Soft-Mixture of previous server and current aggregate.
- Server-anchored class-only DQA updates.
- Day-only and night-only class-only probes.
- The normal 02 warmup/aggregate/repair checkpoints are evaluated in the same
  table for direct comparison.

This still is not the paper's true sparse spatial router.  It is a fast
decision test: if these post-hoc variants cannot recover the lost day/night
signal, the next meaningful MoE step is architectural head/router work rather
than more manual expert assignment.

## 06 Local-Region Expert Fifteen-loop Screening

`notebooks/06_spatial_expert_fifteen_loops.ipynb` moves the MoE idea away from
client-level mixing.  It treats each pseudo-GT box/region as the routing unit
and asks which expert design should be promoted to the next full experiment.

The notebook is a fast design screen, not fifteen full YOLO trainings.  It uses
the completed `03_main_bn_residual_dqa_experiment` metrics plus the previous MoE
router results to evaluate fifteen hypotheses inspired by FedMox, Soft MoE,
Expert Choice Routing, MMoE/PLE, FedBN, and DAMEX.  The selected direction is
written to `output/06_spatial_expert_fifteen_loops/stats/06_selected_full_experiment_candidate.json`.

## 07 Non-residual MoE Theory Fifteen-loop Screening

`notebooks/07_non_residual_moe_theory_loops.ipynb` intentionally leaves the
checkpoint/residual family.  It screens MoE ideas from LLM, vision, retrieval,
and dynamic-compute papers: loss-free load balancing, BASE/Expert Choice global
assignment, GRIN gradient-informed routing, Mixture-of-Depths, Router-Tuning,
DeepSeekMoE fine-grained expert segmentation, CartesianMoE, and V-MoE.

The selected direction is written to
`output/07_non_residual_moe_theory_loops/stats/07_selected_non_residual_candidate.json`.

## 08 pseudoGT Router Recovery Fifteen-loop Screening

`notebooks/08_pseudogt_router_recovery_fifteen_loops.ipynb` focuses on beating
the previous Expert-Choice pseudoGT router.  The measured 05 failure mode was
not simply "MoE is bad"; it was that the router acted too much like a hard
filter and reduced night/hard pseudo boxes across rounds.

This notebook screens fifteen router recovery designs against the completed
03/04/05 evidence.  The highest-ranked direction is a domain-floor assignment
router: keep per-domain minimum exposure, especially for night, and assign
low-confidence-but-stable boxes with a weaker bbox loss instead of deleting
them.  The selected direction is written to
`output/08_pseudogt_router_recovery_fifteen_loops/stats/08_selected_router_candidate.json`.

## 09 Counterfactual View Expert Probe

`notebooks/09_counterfactual_view_expert_probe.ipynb` tests a more MoE-native
version of pseudoGT routing.  Instead of creating experts by domain floors, it
creates experts by the observation condition that made a pseudo box appear:
clean original boxes, illumination-rescued boxes, and cross-view bridge boxes.

The bounded probe samples 80 images per client, predicts original and
illumination-enhanced views, writes expert-specific pseudo datasets, and trains
a one-epoch neck/head `illumination_rescued` probe.  In the first run, the probe
recovered the 05 router failure: mAP50:95 was 0.201 versus 0.190 for 05 and
0.200 for the 03 BN-residual aggregate.  The report is written to
`output/09_counterfactual_view_expert_probe/09_counterfactual_view_expert_probe_report.md`.
