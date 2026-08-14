# Counterintuitive Data Rules

Read this before changing Shaft data. These rules capture cases where a seemingly harmless cleanup
or shortcut changes the training contract.

## 1. Active Raw Is Compact by Design

`data/raw/json` uses `size + layout[].type/bbox/parameters`. It is not an incomplete normalized
record. Do not manufacture `schema`, `annotation`, `instances`, or `extra` merely for uniformity.
Normalized imports may retain their documented `instances` contract; adaptation belongs at the
derived boundary.

## 2. Image-Only Inventory Is Not a Negative

An image without reviewed JSON says “unlabeled,” not “no targets.” Only a completed annotation
record can support a negative. Compact labeled rows with only `full_text` have no four-class
grounding target and keep one native empty row; they receive no train augmentation.

## 3. Full Text Is Raw Truth but Not a Grounding Label

`full_text` remains in compact human annotations. `grounding_layout` consumes only
`shape/icon/image/line` bboxes and must not delete `full_text` from raw or reinterpret it as
background.

## 4. Same Bbox Does Not Prove Duplicate Line Semantics

Two lines can share label and bbox while their ordered paths cross, reverse, or branch differently.
Compare `parameters.points`/`linestrip` before dedupe. The current raw set has known valid examples
of this case.

## 5. Boundary Coordinates Include Width and Height

`x2 == width` and `y2 == height` are valid image-boundary coordinates. Clip to
`[0,width] x [0,height]`, require positive area, then quantize to `0..999`. Clipping to
`width-1/height-1` can erase a right/bottom 1px line.

## 6. Split Before Augmentation

Use explicit train/val/test manifests and prove they are disjoint before generating views. Never
derive train rows from `vlm.test.json`. Empty v5.7 validation JSONL is intentional train-only
semantics, not a reason to invent validation or augment it.

## 7. Partial Crop Intersection Invalidates the Crop

For grounding, every retained bbox must be fully contained. A crop that cuts through any target
is rejected rather than treating the clipped fragment as a new object. Hard negatives require no
full target and no partial overlap, plus complete source annotation.

## 8. Derived Media Is Task-Local

Grounding and context structured/SFT rows reference task-owned images, not raw/source images.
This makes geometry, degradation, and snapshot identity explicit. Never overwrite original media
to make a derived crop or resize fit.

## 9. Prompt Text Is Runtime State

With prompt sampling enabled, SFT row `system_prompt` and `user_prompt` remain empty. The training
YAML maps every dataset name to one versioned pool. Empty row prompts are therefore correct; a
missing/incompatible pool is not.

## 10. Selection Is Identity, Not a Second Truth

V9 selections store source/instance identity and sampling strata. Shape/line attributes and
geometry are reread from `gt_standard` on every rebuild. Copying full targets into selection makes
later source fixes invisible and creates a competing truth.

## 11. Proposal Bbox Is Not Ground Truth

Context reconstruction uses a noisy `proposal_bbox_2d` to identify the intended target. Proposal
and output geometry share the entire contextual crop's Qwen `0..999` coordinate frame; target
geometry may extend outside the proposal. Do not normalize output against the proposal box or ask
the model to copy it.

## 12. Synthetic Reconstruction Has No Clean Twin

Every v5.7 synthetic shape/line crop uses deterministic `synthetic_realism_v1` with 1–3
size-preserving operations; extremely small targets get one mild operation. Store the exact plan
in derived `extra` and do not repeat it online. This differs intentionally from grounding, where
degraded rows have a clean geometry twin and exactly one degradation.

## 13. Real and Synthetic Line Points Share Schema, Not Pixel Policy

`line_context_points` keeps all 122,218 usable real paths and adds 15,000 audited v9 multi-branch
paths. Real crops stay clean; synthetic crops use the realism profile. Target keys are exactly
`is_single + points`. Do not add synthetic single paths, fabricate missing real points, reorder
segments, or add style fields.

## 14. Image Prompt v5.3 Is an Intentional v5.7 Exception

The reviewed real image task still uses its unchanged 13-class
`image_context_reconstruction.v5.3.yaml` pool. Do not rename it to v5.7 without a semantic change,
and do not map synthetic `image_type=N/A` to `other` to increase volume.

## 15. A Builder Existing Does Not Make a Dataset Active

Synthetic `grounding_layout_sync` has a builder but is not materialized or registered in the
formal v5.7 catalog. Background, shape weak attributes, region reconstruction, and legacy
arrow/point tasks are also excluded. The catalog and training YAML, not script presence, define
the active mix.

## 16. Config Validity Is Not Runtime Readiness

Strict config loading proves schema/path resolution for catalog and prompts. It does not prove
model weights, CUDA libraries, world size, or initialization checkpoints exist. In particular,
the v5.7-re config requires its declared `checkpoint-4000`; report it as structurally complete but
not runnable when that artifact is absent.

## 17. Do Not Fix Data Semantics in Training Code

Task field parsing, source adaptation, crop geometry, augmentation, and target construction belong
in raw/derived preparation. `ShaftDataCenter`, sampler, collator, trainer, and pipeline consume
generic validated records and must not reconstruct task truth or carry parallel special cases.

## 18. Audit the Published Bundle as One Contract

Before training or release, verify together:

- raw/v9 split membership and source validity;
- selection/source identity and exclusions;
- structured/SFT/media one-to-one alignment;
- exact target recomputation and prompt-argument schema;
- catalog names/weights/eval flags and prompt versions across all configs;
- model and checkpoint prerequisites.

Counts alone are insufficient: a dataset can have the expected number of rows while pairing the
wrong prompt, stale target schema, or missing continuation artifact.
