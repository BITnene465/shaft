# Eval Bench Frontend Reference Study

本文件记录 Eval Bench 前端重构参考过的外部客户端源码。参考仓库放在
`temp/reference_sources/`，该目录由仓库 `.gitignore` 的 `/temp/` 规则忽略，不进入版本控制。

## Reference Checkouts

| Project | Local path | Commit |
| --- | --- | --- |
| VSCode | `temp/reference_sources/vscode` | `f0fad94e` |
| FiftyOne | `temp/reference_sources/fiftyone` | `a4bfd5b` |
| CVAT | `temp/reference_sources/cvat` | `ea569dd` |
| Codex | `temp/reference_sources/codex` | `95b332c` |

## VSCode

Read targets:

- `src/vs/workbench/browser/workbench.ts`
- `src/vs/workbench/browser/layout.ts`
- `src/vs/workbench/browser/parts/*`
- `src/vs/platform/actions/common/actions.ts`
- `src/vs/platform/keybinding/common/keybindingsRegistry.ts`
- `src/vs/platform/commands/common/commands.ts`

Design notes:

- Workbench startup, layout, actions, commands, menus and keybindings are separate platform services.
- `Action2` registration binds one action id to command execution, menus, command palette visibility and default keybindings.
- Keybindings are stored as command-oriented rules with `when` context expressions and weight, not as page-local key handlers.
- UI parts are composed by workbench/layout services; feature code contributes actions and views instead of editing the shell directly.

Eval Bench implication:

- Keep shortcut actions as an action registry, not hard-coded keys in page components.
- Page shell should compose feature modules; it should not own viewer geometry, service lifecycle rules or settings schema.

## FiftyOne

Read targets:

- `app/packages/app/src/index.tsx`
- `app/packages/state/src`
- `app/packages/looker/src/zoom.ts`
- `app/packages/looker/src/overlays`
- `app/packages/spaces/src/state.ts`
- `app/packages/spaces/src/SpaceTree.ts`
- `app/packages/spaces/src/components/Space.tsx`
- `app/packages/command-bus/src/dispatch/dispatcher.ts`

Design notes:

- App state is split into package-level domains such as state, looker, spaces, command bus and plugin panels.
- Looker keeps media/viewer math separate from panel layout and dataset application state.
- Spaces are represented as a serializable tree with independent panel state and persisted sizes.
- Command bus uses single-handler command dispatch for intentful operations instead of broad event broadcasts.

Eval Bench implication:

- Keep viewer math in pure modules and viewer rendering in a dedicated component.
- Split resizable/panel state and sample navigation from metric computation.
- Prefer intent/action functions for run/job/service operations instead of spreading boolean checks across tables.

## CVAT

Read targets:

- `cvat-ui/src/index.tsx`
- `cvat-ui/src/components/cvat-app.tsx`
- `cvat-ui/src/actions/shortcuts-actions.ts`
- `cvat-ui/src/reducers/shortcuts-reducer.ts`
- `cvat-ui/src/utils/mousetrap-react.tsx`
- `cvat-ui/src/components/header/settings-modal/shortcut-settings.tsx`
- `cvat-canvas/src/typescript/canvas.ts`
- `cvat-canvas/src/typescript/canvasModel.ts`
- `cvat-canvas/src/typescript/canvasController.ts`
- `cvat-canvas/src/typescript/canvasView.ts`
- `cvat-canvas/src/typescript/interactionHandler.ts`
- `cvat-canvas/src/scss/canvas.scss`

Design notes:

- CVAT separates canvas model, controller and view. Public canvas APIs call model/controller/view methods rather than embedding canvas details in React pages.
- Canvas geometry is a first-class object; transform and move operations update handlers and inverse-scale strokes/points.
- Shortcuts are registered per component into a global shortcut store, normalized for display, conflict-checked, and rendered in settings/help UI.
- Mousetrap is wrapped so shortcuts do not fire inside inputs and modal contexts can limit which shortcuts stay active.

Eval Bench implication:

- Keep `CanvasStage` and geometry calculations outside route/page components.
- Keep stroke widths, point sizes and label positions zoom-aware.
- Shortcut tests must cover all components that register global keyboard behavior, not only the top-level route file.

## Codex TUI

Read targets:

- `codex-rs/tui/src/app.rs`
- `codex-rs/tui/src/app_event.rs`
- `codex-rs/tui/src/app_event_sender.rs`
- `codex-rs/tui/src/app_command.rs`
- `codex-rs/tui/src/keymap.rs`
- `codex-rs/tui/src/key_hint.rs`
- `codex-rs/tui/src/chatwidget.rs`
- `codex-rs/tui/src/bottom_pane/*`
- `codex-rs/tui/src/history_cell.rs`

Design notes:

- TUI state is coordinated through typed `AppEvent` and `AppCommand` enums.
- `RuntimeKeymap` resolves config into concrete bindings with deterministic precedence and duplicate validation before handlers consume it.
- The top-level app loop routes terminal events; widgets emit typed events rather than reaching into app internals.
- Input handling is context-sensitive: modal/bottom-pane handlers get priority before main chat handlers.

Eval Bench implication:

- Treat shortcut bindings as a resolved runtime snapshot and pass action ids to focused surfaces.
- Keep UI events typed and explicit; avoid direct cross-component mutation.
- Modal or text-entry surfaces must have priority over global shortcuts.

## Current Eval Bench Boundary Targets

- `main.tsx`: route/page composition only.
- `dashboardState.ts`: shared dashboard state query hook.
- `workspaceSettings.ts`: browser setting schema, normalization and shortcut action registry.
- `workspaceLayout.tsx`: split pane layout, drag resize and persisted panel size.
- `jobsPage.tsx`: evaluation center, queue, manifest creation and runtime log panels.
- `runTables.tsx`: benchmark/run tables, run actions and filtering.
- `filterControls.tsx`: shared filter select controls.
- `statusModel.ts`: job/run/service display state and action permissions.
- `viewerGeometry.ts`: pure viewer math and color resolution.
- `viewerCanvas.tsx`: image stage, pan/zoom and SVG overlay rendering.
- `viewerPanels.tsx`: viewer controls, object list and visible metric panels.
- `settingsControls.tsx`: settings page editor controls.
- `servicesPage.tsx`: model service page API orchestration.
- `manifestTools.ts`: prompt/template/job manifest transformations.
- `sampleNavigation.ts`: sample URL navigation and paging helpers.
