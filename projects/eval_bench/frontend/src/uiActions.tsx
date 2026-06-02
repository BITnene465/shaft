import { Link } from "@tanstack/react-router";
import { createElement } from "react";
import type {
  AnchorHTMLAttributes,
  ButtonHTMLAttributes,
  ElementType,
  HTMLAttributes,
  ReactNode
} from "react";

export type ButtonVariant = "primary" | "secondary" | "mini";

export function joinClassNames(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

export function ActionButton({
  variant = "secondary",
  icon,
  compact,
  className,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  icon?: ReactNode;
  compact?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      className={joinClassNames(
        `${variant}-button`,
        compact && "compact",
        className,
      )}
    >
      {icon}
      {children}
    </button>
  );
}

export function CommandButton({
  variant = "primary",
  icon,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary";
  icon?: ReactNode;
}) {
  return (
    <ActionButton {...props} variant={variant} className="command-button" icon={icon}>
      <span>{children}</span>
    </ActionButton>
  );
}

export function IconActionButton({
  icon,
  title,
  dense = true,
  danger,
  className,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  icon: ReactNode;
  title: string;
  dense?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      title={title}
      aria-label={props["aria-label"] ?? title}
      className={joinClassNames(
        "icon-button",
        dense && "dense",
        danger && "danger",
        className,
      )}
    >
      {icon}
    </button>
  );
}

export function IconNavLink({
  icon,
  title,
  dense = true,
  className,
  ...props
}: {
  icon: ReactNode;
  title: string;
  dense?: boolean;
  className?: string;
  [key: string]: unknown;
}) {
  return createElement(
    Link as ElementType,
    {
      ...props,
      title,
      "aria-label": (props["aria-label"] as string | undefined) ?? title,
      className: joinClassNames("icon-button", dense && "dense", className)
    },
    icon,
  );
}

export function InlineNavLink({
  icon,
  children,
  className,
  ...props
}: {
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
  [key: string]: unknown;
}) {
  return createElement(
    Link as ElementType,
    {
      ...props,
      className: joinClassNames("mini-link", className)
    },
    <>
      {icon}
      {children}
    </>,
  );
}

export function InlineAnchor({
  icon,
  children,
  className,
  ...props
}: AnchorHTMLAttributes<HTMLAnchorElement> & {
  icon?: ReactNode;
}) {
  return (
    <a {...props} className={joinClassNames("mini-link", className)}>
      {icon}
      {children}
    </a>
  );
}

export function NavigationCardAnchor({
  children,
  className,
  ...props
}: AnchorHTMLAttributes<HTMLAnchorElement>) {
  return (
    <a {...props} className={joinClassNames("navigation-card-anchor", className)}>
      {children}
    </a>
  );
}

export function NavigationCardFrame({
  children,
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div {...props} className={joinClassNames("navigation-card-frame", className)}>
      {children}
    </div>
  );
}

export function PanelToggleButton({
  active,
  className,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      aria-expanded={active ?? props["aria-expanded"]}
      className={joinClassNames("panel-toggle-button", active && "active", className)}
    >
      {children}
    </button>
  );
}

export function SelectableRowButton({
  selected,
  className,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  selected?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      aria-current={selected ? "true" : props["aria-current"]}
      className={joinClassNames("sample-row", selected && "selected", className)}
    >
      {children}
    </button>
  );
}

export function SelectableTableRow({
  selected,
  className,
  children,
  ...props
}: HTMLAttributes<HTMLTableRowElement> & {
  selected?: boolean;
}) {
  return (
    <tr
      {...props}
      aria-current={selected ? "true" : props["aria-current"]}
      className={joinClassNames("selectable-row", selected && "selected", className)}
    >
      {children}
    </tr>
  );
}

export function OptionChipButton({
  active,
  className,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      aria-pressed={active ?? props["aria-pressed"]}
      className={joinClassNames("query-chip", active && "active", className)}
    >
      {children}
    </button>
  );
}

export function SelectableCardButton({
  active,
  className,
  children,
  type = "button",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  active?: boolean;
}) {
  return (
    <button
      {...props}
      type={type}
      aria-pressed={active ?? props["aria-pressed"]}
      className={joinClassNames("selectable-card-button", active && "active", className)}
    >
      {children}
    </button>
  );
}

