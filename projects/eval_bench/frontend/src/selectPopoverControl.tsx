import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent } from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown, Search, X } from "lucide-react";

import { ActionButton } from "./ui";
import "./selectPopover.css";

export type SelectOption = {
  value: string;
  label: string;
  disabled?: boolean;
};

const SELECT_SEARCH_THRESHOLD = 8;
const SELECT_VISIBLE_LIMIT = 80;
const SELECT_MENU_VIEWPORT_GAP = 8;
const SELECT_MENU_MAX_WIDTH = 340;
const SELECT_MENU_MIN_HEIGHT = 140;
const SELECT_MENU_MAX_HEIGHT = 300;

type SelectPopoverKind = "form" | "compact" | "filter";

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

function selectOptionMatches(option: SelectOption, query: string) {
  const normalizedQuery = normalizedSelectText(query);
  if (!normalizedQuery) {
    return true;
  }
  return (
    normalizedSelectText(option.label).includes(normalizedQuery) ||
    normalizedSelectText(option.value).includes(normalizedQuery)
  );
}

function nextEnabledIndex(options: ReadonlyArray<SelectOption>, startIndex: number, step: 1 | -1) {
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

function firstEnabledIndex(options: ReadonlyArray<SelectOption>) {
  return options.findIndex((option) => !option.disabled);
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
  const [activeIndex, setActiveIndex] = useState(-1);
  const [menuStyle, setMenuStyle] = useState<Record<string, string>>({});
  const [menuPlacement, setMenuPlacement] = useState<"top" | "bottom">("bottom");
  const selectedOption = useMemo(
    () => options.find((option) => option.value === value),
    [options, value]
  );
  const filteredOptions = useMemo(
    () => options.filter((option) => selectOptionMatches(option, query)),
    [options, query]
  );
  const visibleOptions = useMemo(
    () => filteredOptions.slice(0, SELECT_VISIBLE_LIMIT),
    [filteredOptions]
  );
  const hiddenResultCount = Math.max(filteredOptions.length - visibleOptions.length, 0);
  const searchable = options.length >= SELECT_SEARCH_THRESHOLD;
  const selectLabelId = `${id}-label`;
  const listboxId = `${id}-listbox`;
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
        setOpen(false);
        setQuery("");
      }
    }

    document.addEventListener("pointerdown", closeFromDocument);
    return () => document.removeEventListener("pointerdown", closeFromDocument);
  }, [open]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const selectedIndex = visibleOptions.findIndex((option) => option.value === value);
    setActiveIndex(selectedIndex >= 0 ? selectedIndex : firstEnabledIndex(visibleOptions));
  }, [open, query, value, visibleOptions]);

  useEffect(() => {
    if (open && searchable) {
      window.requestAnimationFrame(() => searchRef.current?.focus());
    }
  }, [open, searchable]);

  useLayoutEffect(() => {
    if (!open) {
      return;
    }
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
      setMenuPlacement(openUp ? "top" : "bottom");
      setMenuStyle({
        "--select-menu-left": `${menuLeft}px`,
        "--select-menu-top": `${Math.max(safeTop, menuTop)}px`,
        "--select-menu-width": `${menuWidth}px`,
        "--select-menu-max-height": `${menuMaxHeight}px`
      });
    }
    updateMenuPosition();
    window.addEventListener("resize", updateMenuPosition);
    window.addEventListener("scroll", updateMenuPosition, true);
    window.visualViewport?.addEventListener("resize", updateMenuPosition);
    window.visualViewport?.addEventListener("scroll", updateMenuPosition);
    return () => {
      window.removeEventListener("resize", updateMenuPosition);
      window.removeEventListener("scroll", updateMenuPosition, true);
      window.visualViewport?.removeEventListener("resize", updateMenuPosition);
      window.visualViewport?.removeEventListener("scroll", updateMenuPosition);
    };
  }, [open]);

  function closePopover() {
    setOpen(false);
    setQuery("");
  }

  function selectOption(option: SelectOption) {
    if (disabled || option.disabled) {
      return;
    }
    onChange(option.value);
    closePopover();
  }

  function moveActive(step: 1 | -1) {
    const nextIndex = nextEnabledIndex(visibleOptions, activeIndex, step);
    if (nextIndex >= 0) {
      setActiveIndex(nextIndex);
    }
  }

  function handleKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (disabled) {
      return;
    }
    if (!open && (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      setOpen(true);
      return;
    }
    if (!open) {
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      closePopover();
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
    if (event.key === "Home") {
      event.preventDefault();
      setActiveIndex(firstEnabledIndex(visibleOptions));
      return;
    }
    if (event.key === "End") {
      event.preventDefault();
      for (let index = visibleOptions.length - 1; index >= 0; index -= 1) {
        if (!visibleOptions[index]?.disabled) {
          setActiveIndex(index);
          break;
        }
      }
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const activeOption = visibleOptions[activeIndex];
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
        onClick={() => setOpen((current) => !current)}
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
                  onClick={() => setQuery("")}
                />
              ) : null}
            </div>
          ) : null}
          {searchable ? (
            <div className="select-popover-meta">
              <span>
                {filteredOptions.length} / {options.length}
              </span>
              {hiddenResultCount > 0 ? <strong>前 {SELECT_VISIBLE_LIMIT} 项</strong> : null}
            </div>
          ) : null}
          <div id={listboxId} className="select-popover-list" role="listbox" aria-labelledby={selectLabelId}>
            {visibleOptions.length === 0 ? (
              <div className="select-popover-empty">没有匹配项</div>
            ) : (
              visibleOptions.map((option, index) => {
                const selected = option.value === value;
                const active = index === activeIndex;
                return (
                  <ActionButton
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
                    tabIndex={-1}
                    data-select-value={option.value}
                    onMouseEnter={() => setActiveIndex(index)}
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
  const options = values.map((item) => ({
    value: item,
    label: labels?.[item] ?? item
  }));
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
