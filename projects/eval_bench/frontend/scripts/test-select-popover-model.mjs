import { strict as assert } from "node:assert";

import {
  SELECT_VISIBLE_LIMIT,
  centeredSelectWindowStart,
  enabledIndexNear,
  firstEnabledIndex,
  nextEnabledIndex,
  pagedEnabledIndex,
  selectVisibleWindow,
  selectWindowStartForActiveIndex
} from "../src/selectPopoverModel.ts";

const options = Array.from({ length: 240 }, (_, index) => ({
  value: `item-${index}`,
  label: `Item ${index}`,
  disabled: index === 79 || index === 80 || index === 159
}));

assert.equal(SELECT_VISIBLE_LIMIT, 36, "select popover must keep a compact bounded render window");

const centeredStart = centeredSelectWindowStart(120, options.length);
assert.equal(centeredStart, 102, "centered window should place the active item in context");

const visibleWindow = selectVisibleWindow(options, centeredStart);
assert.equal(visibleWindow.options.length, SELECT_VISIBLE_LIMIT, "window must cap rendered options");
assert.equal(visibleWindow.start, 102, "window start should respect the requested stable start");
assert.equal(visibleWindow.hiddenBefore, 102, "window must report hidden items before it");
assert.equal(visibleWindow.hiddenAfter, 102, "window must report hidden items after it");

const shortOptions = options.slice(0, 12);
const shortWindow = selectVisibleWindow(shortOptions, 99);
assert.equal(shortWindow.options.length, 12, "short lists must not be cropped");
assert.equal(shortWindow.start, 0, "short lists must ignore stale window starts");
assert.equal(shortWindow.hiddenBefore, 0, "short lists must not report hidden leading items");
assert.equal(shortWindow.hiddenAfter, 0, "short lists must not report hidden trailing items");

const negativeStartWindow = selectVisibleWindow(options, -40);
assert.equal(negativeStartWindow.start, 0, "negative window starts must clamp to zero");
assert.equal(negativeStartWindow.hiddenAfter, 204, "clamped leading window must report hidden tail");

const oversizedStartWindow = selectVisibleWindow(options, 999);
assert.equal(oversizedStartWindow.start, 204, "oversized window starts must clamp to the final page");
assert.equal(oversizedStartWindow.hiddenBefore, 204, "final page must report hidden leading items");
assert.equal(oversizedStartWindow.hiddenAfter, 0, "final page must not report hidden trailing items");

assert.equal(
  selectWindowStartForActiveIndex(102, 120, options.length),
  102,
  "active item inside the window must not move the stable window"
);
assert.equal(
  selectWindowStartForActiveIndex(102, -1, options.length),
  102,
  "negative active item must keep the stable window"
);
assert.equal(
  selectWindowStartForActiveIndex(102, 101, options.length),
  101,
  "active item above the window should move the window just enough"
);
assert.equal(
  selectWindowStartForActiveIndex(102, 138, options.length),
  103,
  "active item below the window should move the window just enough"
);

assert.equal(firstEnabledIndex(options), 0, "first enabled option should skip no valid items");
assert.equal(nextEnabledIndex(options, 78, 1), 81, "arrow navigation should skip disabled options");
assert.equal(nextEnabledIndex(options, 81, -1), 78, "reverse navigation should skip disabled options");
assert.equal(
  enabledIndexNear(options, 79, 1),
  81,
  "paged navigation should search forward for a nearby enabled item"
);
assert.equal(
  enabledIndexNear(options, 80, -1),
  78,
  "paged navigation should search backward for a nearby enabled item"
);
assert.equal(
  pagedEnabledIndex(options, 0, 1),
  35,
  "PageDown should jump a render window and land on the next enabled item"
);
assert.equal(
  pagedEnabledIndex(options, 120, -1),
  85,
  "PageUp should jump a render window upward"
);
assert.equal(
  pagedEnabledIndex(options, -1, 1),
  35,
  "PageDown without an active item should start from the first enabled item"
);

const allDisabled = options.slice(0, 3).map((option) => ({ ...option, disabled: true }));
assert.equal(firstEnabledIndex(allDisabled), -1, "all-disabled lists must expose no active item");
assert.equal(nextEnabledIndex(allDisabled, 0, 1), -1, "all-disabled arrow navigation must stop");
assert.equal(enabledIndexNear(allDisabled, 1, 1), -1, "all-disabled paged navigation must stop");
assert.equal(pagedEnabledIndex(allDisabled, -1, 1), -1, "all-disabled PageDown must stop");

console.log("select popover model checks passed");
