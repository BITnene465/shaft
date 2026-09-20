# SFT Loss Normalization

`train.loss_normalization` is an objective choice, not a kernel switch. Default:

```yaml
train:
  loss_normalization: global_token
```

Let S[r,m] be the sum of weighted shifted-token CE, N[r,m] its valid weight
mass, R the DDP world size, and M the actual number of microbatches in this
optimizer window (including incomplete final windows).

| Mode | Optimizer objective |
|---|---|
| global_token | sum(S) / sum(N), across ranks and the GA window |
| rank_token | mean_r(sum_m(S) / sum_m(N)) |
| microbatch_token | mean_r(mean_m(S / N)) |

`microbatch_token` reproduces the reduction of the audited LF multimodal
Trainer path. This is NOT a claim about every LF version or configuration.
At GA=1, rank_token and microbatch_token coincide; they need not coincide
with global_token when ranks have unequal supervision lengths.

Loss-scale weights contribute to BOTH numerator and denominator. Masked tokens
and shifted-away first tokens do not contribute. A zero-mass microbatch produces
zero loss, retaining its microbatch averaging slot. This explicit safe behavior
differs from implementations that return NaN on all-ignored labels.

Shaft owns GA scaling for all three modes; it must not also ask HF to divide
by GA. DDP averaging is compensated only for global_token. Local modes are
initially limited to built-in SFT CE on single-device/DDP. Other algorithms,
DataParallel, FSDP and DeepSpeed must not silently accept these local modes.
Auxiliary objectives keep their existing per-microbatch average semantics.

Evaluation loss remains globally token-normalized. Kernel choice (standard CE
or Liger fused linear CE) is independent of the reduction. Changing reduction
invalidates exact-resume compatibility; initialize a fresh run from base weights
for the LF comparison, rather than resuming an optimizer checkpoint.

The next controlled reproduction changes ONLY global_token to microbatch_token
relative to the 20260919 worker2 LF-kernel run. Retain its data, sampler, template,
seed, 8000 steps, batch4 x four GPUs, GA1, LR6e-5, warmup800, pixel budget
0.5M-2M, max_length15360 and save interval1000. Compare final checkpoint8000
using the same grounding-overlap exclusion list, ignore rules and saved LF
reference. This isolates reduction, not all remaining LF runtime differences.

## Acceptance (2026-09-20)

- Installed-source CPU regression: 445 passed, no skips or failures.
- Installed-source GPU/Liger gradient tests: 6 passed, no skips or failures.
- Two-process Gloo/DDP: all 6 mode/window combinations matched independent SGD
  updates, including incomplete accumulation windows and unequal rank token mass.
- Tiny real Qwen3.5 tests cover weighted/unweighted supervision, optimizer updates,
  evaluation normalization invariance, and exact-resume rejection on mode changes.
- Evidence: `temp/loss-normalization-20260920/installed-cpu.xml`,
  `installed-gpu.xml`, `ddp-final.log`. The first DDP test harness accidentally
  combined DistributedSampler with Accelerate sharding; corrected to unsharded
  SequentialSampler before acceptance. No production sampler changes were made.
- Next-run YAML: `temp/loss-normalization-20260920/lf-microbatch-repro.yaml`.
  Configuration loaded successfully and was structurally compared to the prior
  run; only normalization and experiment identity differ. Training not started.
- Historical configs retain global_token through the default; saved checkpoints
  and historical evaluation outputs were not edited.
