# Eval Bench UI and Icon Design

## Design Direction

Eval Bench is an engineering workstation, not a marketing surface. The UI should prioritize density,
scanability and stable work areas:

- Keep primary pages focused on one high-frequency workflow.
- Move low-frequency creation, import and registration flows into temporary dialogs.
- Use compact numeric controls for engineering settings; avoid wide presentation-style sliders.
- Keep diagnostics readable; do not hide image paths, labels, boxes, IoU or bbox data behind ellipses.
- Use branded raster icons only for product/domain concepts. Keep generic utility actions such as
  close, delete, search and archive on vector utility icons.

## Icon System

The icon library is generated from one image master sheet and sliced into transparent PNG assets.

- Source chroma sheet: `projects/eval_bench/frontend/assets/eval-bench-icons/source/icon-sheet-chroma.png`
- Transparent sheet: `projects/eval_bench/frontend/assets/eval-bench-icons/source/icon-sheet-transparent.png`
- Runtime PNG icons: `projects/eval_bench/frontend/public/icons/eval-bench/*.png`
- Runtime mapping: `projects/eval_bench/frontend/src/iconLibrary.tsx`

The generated icons are 256x256 PNGs with transparent corners. They are intentionally larger than
their displayed size so the same asset can be used at 17px buttons, 21px navigation and 26px metric
cards without maintaining parallel icon sizes.

There are two source sheets:

- Domain sheet: app mark, navigation, dashboard metrics and page-level actions.
- Action sheet: compact form actions such as restore, apply, preflight, enqueue, save and reset.

## Icon Inventory

| Icon | File | Primary Usage |
| --- | --- | --- |
| App mark | `app-mark.png` and `/logo.png` | Sidebar brand mark |
| Overview | `overview.png` | Overview navigation |
| Benchmark | `benchmark.png` | Benchmark navigation and benchmark metric |
| Service | `service.png` | Model service navigation |
| Eval job | `eval-job.png` | Jobs navigation |
| Run results | `run-results.png` | Runs navigation and run metric |
| Compare analysis | `compare-analysis.png` | Compare navigation |
| Workspace settings | `workspace-settings.png` | Settings navigation |
| Create eval | `create-eval.png` | New eval action |
| Create benchmark | `create-benchmark.png` | Create benchmark action |
| Import prediction | `import-prediction.png` | Import prediction action |
| Register service | `register-service.png` | Register service action |
| Samples | `samples.png` | Sample count metric |
| Predictions | `predictions.png` | Prediction count metric |
| Metrics | `metrics.png` | Reserved for metric-focused views |
| Diagnostics | `diagnostics.png` | Reserved for diagnostics and error views |
| Restore template | `restore-template.png` | Restore manifest template button |
| Apply prompt | `apply-prompt.png` | Apply/save prompt action |
| Preflight validate | `preflight-validate.png` | Validate manifest action |
| Enqueue job | `enqueue-job.png` | Submit eval job action |
| Submit create | `submit-create.png` | Create benchmark form action |
| Save service | `save-service.png` | Save model service form action |
| Reset settings | `reset-settings.png` | Reset setting group action |
| Clear rules | `clear-rules.png` | Clear label/rule action |

## Button Design

Buttons use a compact industrial hierarchy:

- Primary buttons use a dark cyan gradient with white text. Do not place low-contrast cyan text on
  dark cyan backgrounds.
- Secondary buttons use a white-to-light-cyan surface with dark text.
- Button text is always live HTML. Icons sit before text and are optional, but primary workflow
  actions should use an icon from `APP_ICON_PATHS`.
- Generated PNG icons are wrapped by button CSS with a small light backing tile, so the graphite
  portions remain visible on dark primary buttons.
- Page-level command rows must scope metadata text selectors to the header text only. Do not use a
  broad selector such as `.page-command-row span`, because it will override text inside buttons.
- Destructive, close and tiny table utility buttons stay on vector utility icons.
- Buttons must not rely on text truncation; use short labels or let the button width grow.

## Dialog Form Layout

Temporary dialogs should plan form width by field type instead of forcing every field into the same
column width:

- Default fields span 4 of 12 grid columns.
- Long path or URL fields use `.wide-field` and span 6 of 12 columns.
- Result and error messages use `.full-field` and span the full row.
- Submit buttons use `.form-submit-button` and keep a stable minimum width.
- At narrow widths every field spans the full row.
- Large structured inputs such as JSON manifests should use the dedicated manifest split layout,
  not a normal input grid.

## Application Rules

- Components should use `<AppIcon name="..." />` instead of hard-coded `/icons/...` paths.
- Add new icons by extending `APP_ICON_PATHS`; do not duplicate path strings in page components.
- Do not use the raster library for tiny destructive or utility controls. Lucide remains appropriate
  for close, delete, search, archive, inspect and similar one-off utility actions.
- Do not embed UI copy inside generated images. Text remains live HTML.
- The sidebar brand uses `/logo.png`, which is the generated app mark. Product naming remains text in
  the shell so the mark can stay compact.

## Generation Prompt

The current sheet was generated with the built-in `image_gen` tool as a 4x4 chroma-key icon sheet:

```text
Create a strict 4x4 icon sheet with 16 separate application icons for an industrial computer-vision
evaluation dashboard. No labels, no letters, no numbers, no UI text. Compact modern technical
icons, readable at 24px, dark graphite base with cyan and amber accents. Perfectly flat solid
#00ff00 chroma-key background. Row-major icons: app mark, overview, benchmark, service, eval job,
run results, compare analysis, workspace settings, create eval, create benchmark, import prediction,
register service, samples, predictions, metrics, diagnostics.
```

The current action sheet was generated as a 2x4 chroma-key icon sheet:

```text
Create a strict 2x4 icon sheet with 8 compact button action icons for an industrial computer-vision
evaluation dashboard. No labels, no letters, no numbers, no UI text. Match the dark graphite, cyan
and amber style. Row-major icons: restore template, apply prompt, preflight validate, enqueue job,
submit create, save service, reset settings, clear rules.
```

## Validation

After changing this system, run:

```bash
cd projects/eval_bench/frontend
npm run build
EVAL_BENCH_URL=http://127.0.0.1:8765/jobs npm run render-check
EVAL_BENCH_URL=http://127.0.0.1:8765 npm run test:dialogs
```
