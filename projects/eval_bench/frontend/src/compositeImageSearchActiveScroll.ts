import { useEffect } from "react";
import type { RefObject } from "react";

export function useImageSearchActiveScroll<T extends HTMLElement>({
  elementRef,
  activeResultIndex,
  resultCount
}: {
  elementRef: RefObject<T | null>;
  activeResultIndex: number;
  resultCount: number;
}) {
  useEffect(() => {
    scrollActiveImageResultIntoView(elementRef.current, activeResultIndex);
  }, [activeResultIndex, elementRef, resultCount]);
}

function scrollActiveImageResultIntoView(root: HTMLElement | null, activeResultIndex: number) {
  root
    ?.querySelector<HTMLElement>(`[data-result-window-index="${activeResultIndex}"]`)
    ?.scrollIntoView({ block: "nearest" });
}
