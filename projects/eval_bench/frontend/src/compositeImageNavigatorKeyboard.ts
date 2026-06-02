import { useEffect } from "react";
import type { RefObject } from "react";

import { isEditableTarget } from "./keyboardTargets";

export function useCompositeImageKeyboard({
  rootRef,
  imageCount,
  imageIndex,
  setSearchOpen,
  jumpTo
}: {
  rootRef: RefObject<HTMLDivElement | null>;
  imageCount: number;
  imageIndex: number;
  setSearchOpen: (open: boolean) => void;
  jumpTo: (index: number) => void;
}) {
  useEffect(() => {
    function handleDocumentKeyDown(event: KeyboardEvent) {
      if (isEditableTarget(event.target)) {
        if (event.key === "Escape") {
          setSearchOpen(false);
        }
        return;
      }
      if (event.key === "/") {
        event.preventDefault();
        setSearchOpen(true);
        focusCompositeImageSearchInput(rootRef.current);
        return;
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        jumpTo(imageIndex - 1);
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        jumpTo(imageIndex + 1);
        return;
      }
      if (event.key === "PageUp") {
        event.preventDefault();
        jumpTo(imageIndex - 10);
        return;
      }
      if (event.key === "PageDown") {
        event.preventDefault();
        jumpTo(imageIndex + 10);
        return;
      }
      if (event.key === "Home") {
        event.preventDefault();
        jumpTo(0);
        return;
      }
      if (event.key === "End") {
        event.preventDefault();
        jumpTo(imageCount - 1);
      }
    }
    document.addEventListener("keydown", handleDocumentKeyDown);
    return () => document.removeEventListener("keydown", handleDocumentKeyDown);
  }, [imageCount, imageIndex, jumpTo, rootRef, setSearchOpen]);
}

export function focusCompositeImageSearchInput(root: HTMLDivElement | null) {
  root?.querySelector<HTMLInputElement>(".image-navigator-search input")?.focus();
}
