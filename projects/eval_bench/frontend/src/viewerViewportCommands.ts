export const VIEWPORT_RESET_COMMAND = "eval-bench-viewer-reset";

export type ViewportResetCommandDetail = {
  viewportSyncKey?: string | null;
};

export function requestViewportReset(viewportSyncKey?: string | null) {
  window.dispatchEvent(
    new CustomEvent<ViewportResetCommandDetail>(VIEWPORT_RESET_COMMAND, {
      detail: { viewportSyncKey }
    })
  );
}

export function viewportResetCommandDetail(event: Event): ViewportResetCommandDetail | null {
  if (!(event instanceof CustomEvent) || event.type !== VIEWPORT_RESET_COMMAND) {
    return null;
  }
  const detail = event.detail as Partial<ViewportResetCommandDetail> | undefined;
  return {
    viewportSyncKey:
      typeof detail?.viewportSyncKey === "string" ? detail.viewportSyncKey : null
  };
}
