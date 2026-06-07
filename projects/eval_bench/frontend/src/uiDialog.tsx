import { useEffect, useId, useRef } from "react";
import type { ReactNode } from "react";
import { AlertTriangle, X } from "lucide-react";

import { ActionButton, IconActionButton } from "./uiActions";
import "./workspaceDialog.css";

export const DIALOG_FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "[tabindex]:not([tabindex='-1'])"
].join(",");

export function WorkspaceDialog({
  open,
  title,
  meta,
  wide,
  onClose,
  children
}: {
  open: boolean;
  title: string;
  meta?: string;
  wide?: boolean;
  onClose: () => void;
  children: ReactNode;
}) {
  const titleId = useId();
  const metaId = useId();
  const dialogRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (!open) {
      return;
    }
    const previouslyFocused = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousBodyOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const focusTarget = dialogRef.current?.querySelector<HTMLElement>(DIALOG_FOCUSABLE_SELECTOR);
    (focusTarget ?? dialogRef.current)?.focus();
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const focusable = Array.from(
        dialogRef.current?.querySelectorAll<HTMLElement>(DIALOG_FOCUSABLE_SELECTOR) ?? []
      ).filter((element) => element.offsetParent !== null || element === document.activeElement);
      if (focusable.length === 0) {
        event.preventDefault();
        dialogRef.current?.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousBodyOverflow;
      previouslyFocused?.focus();
    };
  }, [onClose, open]);
  if (!open) {
    return null;
  }
  return (
    <div
      className="workspace-dialog-backdrop"
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) {
          onClose();
        }
      }}
    >
      <section
        ref={dialogRef}
        className={wide ? "workspace-dialog wide" : "workspace-dialog"}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={meta ? metaId : undefined}
        tabIndex={-1}
      >
        <header className="workspace-dialog-head">
          <div>
            <strong id={titleId}>{title}</strong>
            {meta ? <span id={metaId}>{meta}</span> : null}
          </div>
          <IconActionButton icon={<X size={14} />} title="关闭" onClick={onClose} />
        </header>
        <div className="workspace-dialog-body">{children}</div>
      </section>
    </div>
  );
}

export function DangerConfirmDialog({
  open,
  title,
  subject,
  description,
  confirmLabel = "确认删除",
  pending,
  onCancel,
  onConfirm
}: {
  open: boolean;
  title: string;
  subject: string;
  description: string;
  confirmLabel?: string;
  pending?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <WorkspaceDialog
      open={open}
      title={title}
      meta="危险操作确认"
      onClose={pending ? () => {} : onCancel}
    >
      <div className="danger-confirm-panel">
        <div className="danger-confirm-copy">
          <div className="danger-confirm-mark">
            <AlertTriangle size={22} />
          </div>
          <div>
            <span>目标对象</span>
            <strong title={subject}>{subject}</strong>
            <p>{description}</p>
          </div>
        </div>
        <div className="danger-confirm-actions">
          <ActionButton variant="secondary" disabled={pending} onClick={onCancel}>
            取消
          </ActionButton>
          <ActionButton
            variant="primary"
            className="danger-action-button"
            disabled={pending}
            onClick={onConfirm}
          >
            {pending ? "处理中" : confirmLabel}
          </ActionButton>
        </div>
      </div>
    </WorkspaceDialog>
  );
}
