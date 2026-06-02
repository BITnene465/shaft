import { useEffect } from "react";

import { isEditableTarget } from "./keyboardTargets";

export function useCompositeObjectKeyboardNavigation({
  navigateObject
}: {
  navigateObject: (direction: -1 | 1) => void;
}) {
  useEffect(() => {
    function navigateObjectFromKey(event: KeyboardEvent) {
      if (isEditableTarget(event.target) || event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }
      const direction = objectNavigationDirection(event.key);
      if (!direction) {
        return;
      }
      event.preventDefault();
      navigateObject(direction);
    }
    window.addEventListener("keydown", navigateObjectFromKey);
    return () => window.removeEventListener("keydown", navigateObjectFromKey);
  }, [navigateObject]);
}

export function objectNavigationDirection(key: string): -1 | 1 | null {
  if (key === "[") {
    return -1;
  }
  if (key === "]") {
    return 1;
  }
  return null;
}
