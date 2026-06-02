import { CompositeMicroMeter } from "./compositeMicroMeter";

import "./compositeImageSearchStatus.css";

export function CompositeImageSearchStatus({
  hiddenBeforeCount,
  hiddenAfterCount,
  hiddenCount,
  dragging,
  dragTargetLabel
}: {
  hiddenBeforeCount: number;
  hiddenAfterCount: number;
  hiddenCount: number;
  dragging: boolean;
  dragTargetLabel: string;
}) {
  const hasClippedResults = hiddenCount > 0;
  const beforeProgress = hasClippedResults ? hiddenBeforeCount / hiddenCount : 0;
  const afterProgress = hasClippedResults ? hiddenAfterCount / hiddenCount : 0;
  return (
    <div
      className="image-jump-search-status"
      data-clipped={hasClippedResults ? "true" : undefined}
      data-dragging={dragging ? "true" : undefined}
    >
      {hasClippedResults ? (
        <div className="image-jump-search-window" aria-label="当前图片搜索结果窗口">
          <CompositeMicroMeter
            className="image-jump-search-window-meter"
            label="before"
            value={hiddenBeforeCount.toLocaleString()}
            meta="clipped"
            progress={beforeProgress}
            idle={hiddenBeforeCount <= 0}
            ariaLabel={`${hiddenBeforeCount.toLocaleString()} hidden results before current window`}
          />
          <CompositeMicroMeter
            className="image-jump-search-window-meter after"
            label="after"
            value={hiddenAfterCount.toLocaleString()}
            meta="clipped"
            progress={afterProgress}
            idle={hiddenAfterCount <= 0}
            ariaLabel={`${hiddenAfterCount.toLocaleString()} hidden results after current window`}
          />
        </div>
      ) : null}
      <div className="image-jump-search-gesture">
        <span>{dragging ? "释放鼠标跳转到预览图片" : "按住结果拖动，连续扫选预览"}</span>
        {dragTargetLabel ? <strong>{dragTargetLabel}</strong> : null}
      </div>
    </div>
  );
}

export function CompositeImageSearchMore({ hiddenCount }: { hiddenCount: number }) {
  if (hiddenCount <= 0) {
    return null;
  }
  return (
    <div className="image-jump-search-more">
      窗口外还有 {hiddenCount.toLocaleString()} 项，继续输入或用 ↑↓ 滑动窗口
    </div>
  );
}
