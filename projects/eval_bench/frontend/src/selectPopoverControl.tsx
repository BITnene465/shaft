import {
  useDeferredValue,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown, Search, X } from "lucide-react";

import { ActionButton } from "./ui";
import {
  SELECT_VISIBLE_LIMIT,
  centeredSelectWindowStart,
  firstEnabledIndex,
  nextEnabledIndex,
  pagedEnabledIndex,
  selectVisibleWindow,
  selectWindowStartForActiveIndex
} from "./selectPopoverModel";
import type { SelectOption } from "./selectPopoverModel";
import "./selectPopover.css";

export type { SelectOption } from "./selectPopoverModel";

const SELECT_SEARCH_THRESHOLD = 8;
const SELECT_MENU_VIEWPORT_GAP = 8;
const SELECT_MENU_MAX_WIDTH = 340;
const SELECT_MENU_MIN_HEIGHT = 140;
const SELECT_MENU_MAX_HEIGHT = 300;
const SELECT_MENU_STYLE_KEYS = [
  "--select-menu-left",
  "--select-menu-top",
  "--select-menu-width",
  "--select-menu-max-height"
] as const;
const SELECT_WINDOW_SCROLL_LISTENER_OPTIONS: AddEventListenerOptions = {
  capture: true,
  passive: true
};
const SELECT_PASSIVE_LISTENER_OPTIONS: AddEventListenerOptions = { passive: true };

type SelectPopoverKind = "form" | "compact" | "filter";
type SelectMenuPlacement = "top" | "bottom";

type SelectPopoverControlProps = {
  label: string;
  value: string;
  options: ReadonlyArray<SelectOption>;
  disabled?: boolean;
  required?: boolean;
  className?: string;
  hideLabel?: boolean;
  dense?: boolean;
  compact?: boolean;
  kind: SelectPopoverKind;
  onChange: (value: string) => void;
};

function normalizedSelectText(value: string) {
  return value.trim().toLocaleLowerCase();
}

function selectMenuStyleChanged(current: Record<string, string>, next: Record<string, string>) {
  return SELECT_MENU_STYLE_KEYS.some((key) => current[key] !== next[key]);
}

function SelectPopoverControl({
  label,
  value,
  options,
  disabled = false,
  required = false,
  className,
  hideLabel = false,
  dense = false,
  compact = false,
  kind,
  onChange
}: SelectPopoverControlProps) {
  const id = useId();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const [activeOptionIndex, setActiveOptionIndex] = useState(-1);
  const [windowStart, setWindowStart] = useState(0);
  const [menuStyle, setMenuStyle] = useState<Record<string, string>>({});
  const [menuPlacement, setMenuPlacement] = useState<SelectMenuPlacement>("bottom");
  const menuStyleRef = useRef<Record<string, string>>({});
  const menuPlacementRef = useRef<SelectMenuPlacement>("bottom");
  const searchable = options.length >= SELECT_SEARCH_THRESHOLD;
  const selectedOption = useMemo(
    () => options.find((option) => option.value === value),
    [options, value]
  );
  const indexedOptions = useMemo(
    () => {
      if (!open || !searchable) {
        return [];
      }
      return options.map((option) => ({
        option,
        searchText: `${normalizedSelectText(option.label)} ${normalizedSelectText(option.value)}`
      }));
    },
    [open, options, searchable]
  );
  const filteredOptions = useMemo(() => {
    const normalizedQuery = normalizedSelectText(deferredQuery);
    if (!open || !searchable || !normalizedQuery) {
      return options;
    }
    return indexedOptions
      .filter((item) => item.searchText.includes(normalizedQuery))
      .map((item) => item.option);
  }, [deferredQuery, indexedOptions, open, options, searchable]);
  const visibleWindow = useMemo(
    () => selectVisibleWindow(filteredOptions, windowStart),
    [filteredOptions, windowStart]
  );
  const visibleOptions = visibleWindow.options;
  const hiddenResultCount = visibleWindow.hiddenBefore + visibleWindow.hiddenAfter;
  const selectLabelId = `${id}-label`;
  const listboxId = `${id}-listbox`;
  const activeOptionInVisibleWindow =
    activeOptionIndex >= visibleWindow.start &&
    activeOptionIndex < visibleWindow.start + visibleOptions.length;
  const activeOptionId =
    open && activeOptionIndex >= 0 && activeOptionIndex < filteredOptions.length && activeOptionInVisibleWindow
      ? `${listboxId}-option-${activeOptionIndex + 1}`
      : undefined;
  const selectedLabel = selectedOption?.label ?? value;
  const controlClassName = [
    "select-popover-control",
    `select-popover-${kind}`,
    dense ? "dense" : "",
    compact ? "compact" : "",
    hideLabel ? "select-control-label-hidden" : "",
    className
  ]
    .filter(Boolean)
    .join(" ");

  useEffect(() => {
    if (!open) {
      return;
    }

    function closeFromDocument(event: PointerEvent) {
      const target = event.target as Node;
      if (!rootRef.current?.contains(target) && !menuRef.current?.contains(target)) {
        closePopover();
      }
    }

    document.addEventListener("pointerdown", closeFromDocument);
    return () => document.removeEventListener("pointerdown", closeFromDocument);
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const selectedIndex = filteredOptions.findIndex((option) => option.value === value);
    const nextActiveIndex = selectedIndex >= 0 ? selectedIndex : firstEnabledIndex(filteredOptions);
    setActiveOptionIndex(nextActiveIndex);
    setWindowStart(centeredSelectWindowStart(Math.max(nextActiveIndex, 0), filteredOptions.length));
  }, [filteredOptions, open, value]);

  useEffect(() => {
    if (open && searchable) {
      window.requestAnimationFrame(() => searchRef.current?.focus());
    }
  }, [open, searchable]);

  useLayoutEffect(() => {
    if (!open) {
      return;
    }
    let positionFrame = 0;
    function updateMenuPosition() {
      const rect = rootRef.current?.getBoundingClientRect();
      if (!rect) {
        return;
      }
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const boundaryRect = rootRef.current
        ?.closest('[role="dialog"], .settings-drawer-scroll')
        ?.getBoundingClientRect();
      const safeTop = Math.max(SELECT_MENU_VIEWPORT_GAP, boundaryRect?.top ?? SELECT_MENU_VIEWPORT_GAP);
      const safeBottom = Math.min(
        viewportHeight - SELECT_MENU_VIEWPORT_GAP,
        boundaryRect?.bottom ?? viewportHeight - SELECT_MENU_VIEWPORT_GAP
      );
      const availableBelow = Math.max(0, safeBottom - rect.bottom - 2);
      const availableAbove = Math.max(0, rect.top - safeTop - 2);
      const openUp = availableBelow < SELECT_MENU_MIN_HEIGHT && availableAbove > availableBelow;
      const availableSpace = openUp ? availableAbove : availableBelow;
      const menuMaxHeight = Math.max(1, Math.min(availableSpace, SELECT_MENU_MAX_HEIGHT));
      const menuWidth = Math.min(
        Math.max(rect.width, Math.min(300, viewportWidth - SELECT_MENU_VIEWPORT_GAP * 2)),
        SELECT_MENU_MAX_WIDTH,
        viewportWidth - SELECT_MENU_VIEWPORT_GAP * 2
      );
      const menuLeft = Math.min(
        Math.max(SELECT_MENU_VIEWPORT_GAP, rect.left),
        viewportWidth - menuWidth - SELECT_MENU_VIEWPORT_GAP
      );
      const menuTop = openUp
        ? Math.max(safeTop, rect.top - menuMaxHeight - 2)
        : Math.min(rect.bottom + 2, safeBottom - menuMaxHeight);
      const nextPlacement = openUp ? "top" : "bottom";
      const nextMenuStyle = {
        "--select-menu-left": `${menuLeft}px`,
        "--select-menu-top": `${Math.max(safeTop, menuTop)}px`,
        "--select-menu-width": `${menuWidth}px`,
        "--select-menu-max-height": `${menuMaxHeight}px`
      };
      if (menuPlacementRef.current !== nextPlacement) {
        menuPlacementRef.current = nextPlacement;
        setMenuPlacement(nextPlacement);
      }
      if (selectMenuStyleChanged(menuStyleRef.current, nextMenuStyle)) {
        menuStyleRef.current = nextMenuStyle;
        setMenuStyle(nextMenuStyle);
      }
    }
    function scheduleMenuPositionUpdate() {
      if (positionFrame !== 0) {
        return;
      }
      positionFrame = window.requestAnimationFrame(() => {
        positionFrame = 0;
        updateMenuPosition();
      });
    }
    updateMenuPosition();
    window.addEventListener("resize", scheduleMenuPositionUpdate);
    window.addEventListener(
      "scroll",
      scheduleMenuPositionUpdate,
      SELECT_WINDOW_SCROLL_LISTENER_OPTIONS
    );
    window.visualViewport?.addEventListener("resize", scheduleMenuPositionUpdate);
    window.visualViewport?.addEventListener(
      "scroll",
      scheduleMenuPositionUpdate,
      SELECT_PASSIVE_LISTENER_OPTIONS
    );
    return () => {
      if (positionFrame !== 0) {
        window.cancelAnimationFrame(positionFrame);
      }
      window.removeEventListener("resize", scheduleMenuPositionUpdate);
      window.removeEventListener(
        "scroll",
        scheduleMenuPositionUpdate,
        SELECT_WINDOW_SCROLL_LISTENER_OPTIONS
      );
      window.visualViewport?.removeEventListener("resize", scheduleMenuPositionUpdate);
      window.visualViewport?.removeEventListener(
        "scroll",
        scheduleMenuPositionUpdate,
        SELECT_PASSIVE_LISTENER_OPTIONS
      );
    };
  }, [open]);

  function focusTrigger() {
    window.requestAnimationFrame(() => {
      rootRef.current
        ?.querySelector<HTMLButtonElement>(".select-popover-trigger")
        ?.focus({ preventScroll: true });
    });
  }

  function focusSearchInput() {
    window.requestAnimationFrame(() => searchRef.current?.focus({ preventScroll: true }));
  }

  function openPopover() {
    setOpen(true);
  }

  function closePopover({ restoreFocus = false }: { restoreFocus?: boolean } = {}) {
    setOpen(false);
    setQuery("");
    if (restoreFocus) {
      focusTrigger();
    }
  }

  function togglePopover() {
    if (open) {
      closePopover({ restoreFocus: false });
      return;
    }
    openPopover();
  }

  function clearSearch() {
    setQuery("");
    focusSearchInput();
  }

  function selectOption(option: SelectOption) {
    if (disabled || option.disabled) {
      return;
    }
    onChange(option.value);
    closePopover({ restoreFocus: true });
  }

  function moveActive(step: 1 | -1) {
    const nextIndex = nextEnabledIndex(filteredOptions, activeOptionIndex, step);
    if (nextIndex >= 0) {
      setActiveOptionIndex(nextIndex);
      setWindowStart((currentStart) =>
        selectWindowStartForActiveIndex(currentStart, nextIndex, filteredOptions.length)
      );
    }
  }

  function pageActive(direction: 1 | -1) {
    const nextIndex = pagedEnabledIndex(filteredOptions, activeOptionIndex, direction);
    if (nextIndex >= 0) {
      setActiveOptionIndex(nextIndex);
      setWindowStart((currentStart) =>
        selectWindowStartForActiveIndex(currentStart, nextIndex, filteredOptions.length)
      );
    }
  }

  function isSearchInputEvent(event: ReactKeyboardEvent<HTMLDivElement>) {
    return event.target === searchRef.current;
  }

  function handleKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (disabled) {
      return;
    }
    if (!open && (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      openPopover();
      return;
    }
    if (!open) {
      return;
    }
    if (isSearchInputEvent(event) && (event.key === "Home" || event.key === "End")) {
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      event.nativeEvent.stopImmediatePropagation?.();
      closePopover({ restoreFocus: true });
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveActive(1);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      moveActive(-1);
      return;
    }
    if (event.key === "PageDown") {
      event.preventDefault();
      pageActive(1);
      return;
    }
    if (event.key === "PageUp") {
      event.preventDefault();
      pageActive(-1);
      return;
    }
    if (event.key === "Home") {
      event.preventDefault();
      const nextIndex = firstEnabledIndex(filteredOptions);
      setActiveOptionIndex(nextIndex);
      setWindowStart(selectWindowStartForActiveIndex(0, nextIndex, filteredOptions.length));
      return;
    }
    if (event.key === "End") {
      event.preventDefault();
      for (let index = filteredOptions.length - 1; index >= 0; index -= 1) {
        if (!filteredOptions[index]?.disabled) {
          setActiveOptionIndex(index);
          setWindowStart(selectWindowStartForActiveIndex(windowStart, index, filteredOptions.length));
          break;
        }
      }
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const activeOption = filteredOptions[activeOptionIndex];
      if (activeOption) {
        selectOption(activeOption);
      }
    }
  }

  return (
    <div
      ref={rootRef}
      className={controlClassName}
      onKeyDown={handleKeyDown}
      data-select-open={open ? "true" : undefined}
    >
      <span id={selectLabelId} className="select-popover-label">
        {label}
      </span>
      <ActionButton
        variant="secondary"
        className="select-popover-trigger"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listboxId : undefined}
        aria-labelledby={`${selectLabelId} ${id}-value`}
        aria-required={required}
        title={label}
        data-select-value={value}
        onClick={togglePopover}
      >
        <span id={`${id}-value`} className="select-popover-value">
          {selectedLabel || "未选择"}
        </span>
        <ChevronDown size={14} aria-hidden="true" />
      </ActionButton>
      {open ? createPortal(
        <div
          ref={menuRef}
          className="select-popover-menu"
          data-select-popover-menu="true"
          data-placement={menuPlacement}
          style={menuStyle}
        >
          {searchable ? (
            <div className="select-popover-search">
              <Search size={14} aria-hidden="true" />
              <input
                ref={searchRef}
                type="search"
                value={query}
                placeholder="搜索选项"
                aria-label={`搜索${label}`}
                aria-autocomplete="list"
                aria-controls={listboxId}
                aria-activedescendant={activeOptionId}
                onChange={(event) => setQuery(event.target.value)}
              />
              {query ? (
                <ActionButton
                  variant="mini"
                  compact
                  className="select-popover-clear"
                  aria-label="清空搜索"
                  title="清空搜索"
                  icon={<X size={13} />}
                  onClick={clearSearch}
                />
              ) : null}
            </div>
          ) : null}
          {searchable ? (
            <div className="select-popover-meta">
              <span>
                {filteredOptions.length} / {options.length}
              </span>
              {hiddenResultCount > 0 ? (
                <strong className="select-popover-window-note">
                  {visibleWindow.hiddenBefore > 0 ? `上 ${visibleWindow.hiddenBefore}` : null}
                  {visibleWindow.hiddenBefore > 0 && visibleWindow.hiddenAfter > 0 ? " / " : null}
                  {visibleWindow.hiddenAfter > 0 ? `下 ${visibleWindow.hiddenAfter}` : null}
                </strong>
              ) : null}
            </div>
          ) : null}
          <div
            id={listboxId}
            className="select-popover-list"
            role="listbox"
            aria-labelledby={selectLabelId}
            aria-activedescendant={activeOptionId}
            data-select-visible-limit={SELECT_VISIBLE_LIMIT}
            data-select-window-start={visibleWindow.start}
          >
            {visibleOptions.length === 0 ? (
              <div className="select-popover-empty">没有匹配项</div>
            ) : (
              visibleOptions.map((option, index) => {
                const selected = option.value === value;
                const absoluteIndex = visibleWindow.start + index;
                const active = absoluteIndex === activeOptionIndex;
                return (
                  <ActionButton
                    id={`${listboxId}-option-${absoluteIndex + 1}`}
                    key={option.value}
                    variant="mini"
                    compact
                    className={[
                      "select-popover-option",
                      selected ? "selected" : "",
                      active ? "active" : ""
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    disabled={option.disabled}
                    role="option"
                    aria-selected={selected}
                    aria-posinset={absoluteIndex + 1}
                    aria-setsize={filteredOptions.length}
                    tabIndex={-1}
                    data-select-value={option.value}
                    data-select-window-index={absoluteIndex + 1}
                    onMouseEnter={() => {
                      setActiveOptionIndex(absoluteIndex);
                      setWindowStart((currentStart) =>
                        selectWindowStartForActiveIndex(
                          currentStart,
                          absoluteIndex,
                          filteredOptions.length
                        )
                      );
                    }}
                    onClick={() => selectOption(option)}
                  >
                    <span>{option.label}</span>
                    {selected ? <Check size={14} aria-hidden="true" /> : null}
                  </ActionButton>
                );
              })
            )}
          </div>
        </div>,
        document.body
      ) : null}
    </div>
  );
}

export function FormSelectControl({
  label,
  value,
  options,
  disabled = false,
  required = false,
  className,
  hideLabel = false,
  onChange
}: {
  label: string;
  value: string;
  options: ReadonlyArray<SelectOption>;
  disabled?: boolean;
  required?: boolean;
  className?: string;
  hideLabel?: boolean;
  onChange: (value: string) => void;
}) {
  const labelClassName = [className, hideLabel ? "select-control-label-hidden" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <SelectPopoverControl
      label={label}
      value={value}
      options={options}
      disabled={disabled}
      required={required}
      className={labelClassName || undefined}
      hideLabel={hideLabel}
      kind="form"
      onChange={onChange}
    />
  );
}

export function CompactSelectControl({
  label,
  value,
  options,
  disabled = false,
  dense = false,
  onChange
}: {
  label: string;
  value: string;
  options: ReadonlyArray<{ value: string; label: string }>;
  disabled?: boolean;
  dense?: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <SelectPopoverControl
      label={label}
      value={value}
      options={options}
      disabled={disabled}
      dense={dense}
      kind="compact"
      onChange={onChange}
    />
  );
}

export function FilterSelectControl({
  label,
  value,
  values,
  labels,
  compact = false,
  onChange
}: {
  label: string;
  value: string;
  values: string[];
  labels?: Record<string, string>;
  compact?: boolean;
  onChange: (value: string) => void;
}) {
  const options = useMemo(
    () =>
      values.map((item) => ({
        value: item,
        label: labels?.[item] ?? item
      })),
    [labels, values]
  );
  return (
    <SelectPopoverControl
      label={label}
      value={value}
      options={options}
      compact={compact}
      kind="filter"
      onChange={onChange}
    />
  );
}
