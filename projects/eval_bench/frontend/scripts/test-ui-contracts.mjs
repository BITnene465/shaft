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
