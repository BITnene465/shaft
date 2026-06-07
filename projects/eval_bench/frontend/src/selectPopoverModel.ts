export type SelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

export type VisibleSelectWindow = {
  options: ReadonlyArray<SelectOption>;
  start: number;
  hiddenBefore: number;
  hiddenAfter: number;
};

export const SELECT_VISIBLE_LIMIT = 80;

export function nextEnabledIndex(
  options: ReadonlyArray<SelectOption>,
  startIndex: number,
  step: 1 | -1
) {
  if (options.length === 0) {
    return -1;
  }
  let index = startIndex;
  for (let scanned = 0; scanned < options.length; scanned += 1) {
    index = (index + step + options.length) % options.length;
    if (!options[index]?.disabled) {
      return index;
    }
  }
  return -1;
}

export function firstEnabledIndex(options: ReadonlyArray<SelectOption>) {
  return options.findIndex((option) => !option.disabled);
}

export function enabledIndexNear(
  options: ReadonlyArray<SelectOption>,
  targetIndex: number,
  direction: 1 | -1
) {
  if (options.length === 0) {
    return -1;
  }
  const clampedIndex = Math.min(Math.max(0, targetIndex), options.length - 1);
  for (
    let index = clampedIndex;
    direction > 0 ? index < options.length : index >= 0;
    index += direction
  ) {
    if (!options[index]?.disabled) {
      return index;
    }
  }
  for (
    let index = clampedIndex - direction;
    direction > 0 ? index >= 0 : index < options.length;
    index -= direction
  ) {
    if (!options[index]?.disabled) {
      return index;
    }
  }
  return -1;
}

export function pagedEnabledIndex(
  options: ReadonlyArray<SelectOption>,
  currentIndex: number,
  direction: 1 | -1
) {
  const activeIndex = currentIndex >= 0 ? currentIndex : firstEnabledIndex(options);
  if (activeIndex < 0) {
    return -1;
  }
  return enabledIndexNear(options, activeIndex + direction * (SELECT_VISIBLE_LIMIT - 1), direction);
}

export function clampSelectWindowStart(start: number, itemCount: number) {
  return Math.min(Math.max(0, start), Math.max(0, itemCount - SELECT_VISIBLE_LIMIT));
}

export function centeredSelectWindowStart(anchorIndex: number, itemCount: number) {
  return clampSelectWindowStart(anchorIndex - Math.floor(SELECT_VISIBLE_LIMIT / 2), itemCount);
}

export function selectWindowStartForActiveIndex(
  currentStart: number,
  activeIndex: number,
  itemCount: number
) {
  if (activeIndex < 0) {
    return clampSelectWindowStart(currentStart, itemCount);
  }
  if (activeIndex < currentStart) {
    return clampSelectWindowStart(activeIndex, itemCount);
  }
  if (activeIndex >= currentStart + SELECT_VISIBLE_LIMIT) {
    return clampSelectWindowStart(activeIndex - SELECT_VISIBLE_LIMIT + 1, itemCount);
  }
  return clampSelectWindowStart(currentStart, itemCount);
}

export function selectVisibleWindow(
  options: ReadonlyArray<SelectOption>,
  windowStart: number
): VisibleSelectWindow {
  if (options.length <= SELECT_VISIBLE_LIMIT) {
    return {
      options,
      start: 0,
      hiddenBefore: 0,
      hiddenAfter: 0
    };
  }
  const start = clampSelectWindowStart(windowStart, options.length);
  const visibleOptions = options.slice(start, start + SELECT_VISIBLE_LIMIT);
  return {
    options: visibleOptions,
    start,
    hiddenBefore: start,
    hiddenAfter: Math.max(options.length - start - visibleOptions.length, 0)
  };
}
