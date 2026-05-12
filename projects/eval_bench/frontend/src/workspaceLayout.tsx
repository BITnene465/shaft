import { useEffect, useRef, useState } from "react";
import type { PointerEvent, ReactNode } from "react";

import { clampNumber } from "./viewerGeometry";
import { loadSplitSize } from "./workspaceSettings";

export function ResizableSplit({
  className,
  storageKey,
  fixedPane = "first",
  defaultSize,
  minSize,
  maxSize,
  first,
  second
}: {
  className: string;
  storageKey: string;
  fixedPane?: "first" | "second";
  defaultSize: number;
  minSize: number;
  maxSize: number;
  first: ReactNode;
  second: ReactNode;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState(() => loadSplitSize(storageKey, defaultSize, minSize, maxSize));
  const [containerWidth, setContainerWidth] = useState(0);
  const dragRef = useRef<{ pointerId: number; startX: number; startSize: number } | null>(null);
  const effectiveMaxSize = Math.max(
    minSize,
    Math.min(maxSize, containerWidth > 0 ? containerWidth - minSize - 8 : maxSize)
  );

  useEffect(() => {
    localStorage.setItem(storageKey, String(size));
  }, [size, storageKey]);

  useEffect(() => {
    const node = rootRef.current;
    if (!node) {
      return undefined;
    }
    function updateWidth() {
      setContainerWidth(Math.max(0, node?.getBoundingClientRect().width ?? 0));
    }
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setSize((current) => clampNumber(current, minSize, effectiveMaxSize));
  }, [effectiveMaxSize, minSize]);

  function startResize(event: PointerEvent<HTMLDivElement>) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startSize: size
    };
  }

  function moveResize(event: PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    const delta = event.clientX - drag.startX;
    const signedDelta = fixedPane === "first" ? delta : -delta;
    setSize(clampNumber(drag.startSize + signedDelta, minSize, effectiveMaxSize));
  }

  function endResize(event: PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <div
      ref={rootRef}
      className={`${className} resizable-split ${fixedPane === "first" ? "fixed-first" : "fixed-second"}`}
      style={{
        gridTemplateColumns:
          fixedPane === "first"
            ? `${size}px 8px minmax(0, 1fr)`
            : `minmax(0, 1fr) 8px ${size}px`
      }}
    >
      {first}
      <div
        className="split-resizer"
        role="separator"
        aria-orientation="vertical"
        tabIndex={0}
        onPointerDown={startResize}
        onPointerMove={moveResize}
        onPointerUp={endResize}
        onPointerCancel={endResize}
      />
      {second}
    </div>
  );
}
