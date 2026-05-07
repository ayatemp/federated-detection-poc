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
