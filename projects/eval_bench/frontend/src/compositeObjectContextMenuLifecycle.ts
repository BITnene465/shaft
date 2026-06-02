import { useEffect } from "react";

import type { CompositeObjectMenuRequest } from "./compositeObjectContextMenu";

export function useCompositeObjectContextMenuLifecycle({
  contextMenu,
  closeContextMenu
}: {
  contextMenu: CompositeObjectMenuRequest | null;
  closeContextMenu: () => void;
}) {
  useEffect(() => {
    if (!contextMenu) {
      return undefined;
    }
    function closeMenu() {
      closeContextMenu();
    }
    function closeMenuFromKey(event: KeyboardEvent) {
      if (event.key === "Escape") {
        closeContextMenu();
      }
    }
    window.addEventListener("pointerdown", closeMenu);
    window.addEventListener("keydown", closeMenuFromKey);
    return () => {
      window.removeEventListener("pointerdown", closeMenu);
      window.removeEventListener("keydown", closeMenuFromKey);
    };
  }, [closeContextMenu, contextMenu]);
}
