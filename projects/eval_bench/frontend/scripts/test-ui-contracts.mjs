import { strict as assert } from "node:assert";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const srcRoot = path.join(root, "src");
const sourceFiles = await collectSourceFiles(srcRoot);

for (const filePath of sourceFiles) {
  const source = await readFile(filePath, "utf8");
  const relativePath = path.relative(root, filePath);
  assertNoBlockingBrowserDialogs(source, relativePath);
  assertNoBusinessDialogShell(source, relativePath);
  assertNoLegacySampleFilters(source, relativePath);
}

const jobsPage = await readSource("src/jobsPage.tsx");
assert(
  jobsPage.includes('<ActionButton variant="mini" onClick={() => onChange(labelOptions)}>'),
  "label subtask select-all action must use ActionButton",
);
assert(
  jobsPage.includes('<ActionButton variant="mini" onClick={() => onChange([])}>'),
  "label subtask default-policy action must use ActionButton",
);
assert(
  jobsPage.includes('<ActionButton variant="mini" type="submit">'),
  "label subtask custom-label submit must use ActionButton",
);
assert(
  !jobsPage.includes('<button type="submit">添加</button>'),
  "label subtask submit regressed to a raw button",
);
assert(
  jobsPage.includes("import { CompactSelectControl } from \"./controlPrimitives\";"),
  "manifest toolbar selects must use CompactSelectControl",
);
assert(
  (jobsPage.match(/<CompactSelectControl/g) ?? []).length >= 2,
  "manifest toolbar must render template and prompt through CompactSelectControl",
);
assert(
  !jobsPage.includes('className="filter-select compact"'),
  "jobs page must not create ad hoc compact filter selects outside filterControls",
);

const settingsControls = await readSource("src/settingsControls.tsx");
assert(
  settingsControls.includes('<ActionButton variant="mini" onClick={() => onReset(action.id)}>'),
  "shortcut reset action must use ActionButton",
);
assert(
  settingsControls.includes(
    '<ActionButton variant="secondary" className="settings-inline-action" onClick={onResetAll}>',
  ),
  "shortcut reset-all action must use ActionButton",
);

const settingsPage = await readSource("src/settingsPage.tsx");
assert(
  settingsPage.includes('import { CompactSelectControl, NumberSettingControl } from "./controlPrimitives";'),
  "settings page selects must use CompactSelectControl",
);
assert(
  /<CompactSelectControl\s+dense\s+label="预测线型"/.test(settingsPage),
  "settings prediction line style select must use CompactSelectControl",
);
assert(
  settingsPage.includes('className="settings-search-clear"'),
  "settings search clear action must use IconActionButton",
);
assert(
  !settingsPage.includes('className="compact-select dense"'),
  "settings page must not create ad hoc compact select shells",
);
assert(
  !/<button[^>]+className="settings-inline-action"/.test(settingsPage),
  "settings inline standard actions must use ActionButton",
);
assert(
  !/<button[^>]+removeLabelColor/.test(settingsPage),
  "settings label clear action must use ActionButton",
);
const overviewPage = await readSource("src/overviewPage.tsx");
assert(
  overviewPage.includes("export function OverviewPage()"),
  "overview page module must export OverviewPage",
);
const mainEntry = await readSource("src/main.tsx");
assert(
  mainEntry.includes('import { OverviewPage } from "./overviewPage";'),
  "main.tsx must route to the extracted OverviewPage module",
);
assert(
  !mainEntryHasOverviewImplementation(mainEntry),
  "main.tsx should only route to OverviewPage, not implement the overview workbench",
);
assert(
  !mainEntryHasSettingsImplementation(mainEntry),
  "main.tsx should only route to SettingsPage, not implement the settings workbench",
);

console.log("ui contract checks passed");

async function collectSourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectSourceFiles(entryPath)));
    } else if (/\.(ts|tsx)$/.test(entry.name)) {
      files.push(entryPath);
    }
  }
  return files;
}

async function readSource(relativePath) {
  return readFile(path.join(root, relativePath), "utf8");
}

function assertNoBlockingBrowserDialogs(source, relativePath) {
  const match = source.match(/\b(?:window\.)?(confirm|alert|prompt)\s*\(/);
  assert(!match, `${relativePath}: blocking browser dialog '${match?.[1]}' is not allowed`);
}

function assertNoBusinessDialogShell(source, relativePath) {
  if (relativePath === "src/ui.tsx") {
    return;
  }
  assert(
    !/className=\{?["'`][^"'`]*workspace-dialog/.test(source),
    `${relativePath}: dialog shell classes belong in WorkspaceDialog`,
  );
}

function assertNoLegacySampleFilters(source, relativePath) {
  assert(!source.includes("sample-filters"), `${relativePath}: legacy sample-filters are not allowed`);
}

function mainEntryHasSettingsImplementation(source) {
  return /function\s+SettingsPage\s*\(/.test(source) || source.includes("settings-workbench-shell");
}

function mainEntryHasOverviewImplementation(source) {
  return (
    /function\s+OverviewPage\s*\(/.test(source) ||
    source.includes("overview-console") ||
    source.includes("overview-chart-matrix")
  );
}
