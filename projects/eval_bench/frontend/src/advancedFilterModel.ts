import type { AdvancedFilterControl } from "./advancedFilterTypes";

export function defaultFilterValue(control: AdvancedFilterControl) {
  if (control.type === "select") {
    return control.values.includes("all") ? "all" : control.values[0] ?? "";
  }
  return "";
}

export function resetAdvancedFilter(control: AdvancedFilterControl) {
  control.onChange(defaultFilterValue(control));
}

export function applyAdvancedFilterValues(
  controls: AdvancedFilterControl[],
  values: Record<string, string>,
) {
  for (const control of controls) {
    const nextValue = values[control.id] ?? control.value;
    if (nextValue !== control.value) {
      control.onChange(nextValue);
    }
  }
}

export function displayFilterValue(control: AdvancedFilterControl) {
  if (control.type === "select") {
    return control.labels?.[control.value] ?? control.value;
  }
  return control.value;
}

export function groupAdvancedControls(controls: AdvancedFilterControl[]) {
  const groups = [
    { id: "search", title: "搜索", controls: [] as AdvancedFilterControl[] },
    { id: "scope", title: "范围目录", controls: [] as AdvancedFilterControl[] },
    { id: "tune", title: "阈值排序", controls: [] as AdvancedFilterControl[] },
  ];
  for (const control of controls) {
    const normalized = `${control.id} ${control.label}`.toLowerCase();
    if (control.type === "search") {
      groups[0].controls.push(control);
    } else if (
      control.type === "number" ||
      normalized.includes("sort") ||
      normalized.includes("排序") ||
      normalized.includes("最低") ||
      normalized.includes("order")
    ) {
      groups[2].controls.push(control);
    } else {
      groups[1].controls.push(control);
    }
  }
  return groups.filter((group) => group.controls.length > 0);
}

export function advancedFilterValues(controls: AdvancedFilterControl[]) {
  return Object.fromEntries(
    controls.map((control) => [control.id, control.value])
  ) as Record<string, string>;
}

export function advancedFilterDirtyControlIds(
  controls: AdvancedFilterControl[],
  values: Record<string, string>,
) {
  return new Set(
    controls
      .filter((control) => (values[control.id] ?? control.value) !== control.value)
      .map((control) => control.id)
  );
}

export function advancedFilterValuesKey(controls: AdvancedFilterControl[]) {
  return controls.map((control) => `${control.id}\u0000${control.value}`).join("\u0001");
}

export function syncDraftValuesWithApplied(
  controls: AdvancedFilterControl[],
  currentDraftValues: Record<string, string>,
  dirtyControlIds: ReadonlySet<string>,
) {
  return Object.fromEntries(
    controls.map((control) => {
      return [
        control.id,
        dirtyControlIds.has(control.id)
          ? currentDraftValues[control.id] ?? control.value
          : control.value,
      ];
    })
  );
}
