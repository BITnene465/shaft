import {
  ChevronLeft,
  ChevronRight,
  MousePointer2,
  Move,
  RotateCcw,
  Search
} from "lucide-react";

import { IconActionButton } from "./ui";

import "./compositeInteractionPalette.css";

export function CompositeInteractionPalette({
  canPrevious,
  canNext,
  onPrevious,
  onNext,
  onSearch,
  onResetViewport
}: {
  canPrevious: boolean;
  canNext: boolean;
  onPrevious: () => void;
  onNext: () => void;
  onSearch: () => void;
  onResetViewport: () => void;
}) {
  return (
    <div className="composite-interaction-palette" role="toolbar" aria-label="组合视图快捷工具">
      <span className="interaction-palette-anchor" title="组合视图支持鼠标、键盘和对象交互">
        <MousePointer2 size={13} />
      </span>
      <IconActionButton
        className="interaction-palette-tool"
        data-tool="previous"
        icon={<ChevronLeft size={13} />}
        title="上一张图片"
        disabled={!canPrevious}
        onClick={onPrevious}
      />
      <IconActionButton
        className="interaction-palette-tool"
        data-tool="next"
        icon={<ChevronRight size={13} />}
        title="下一张图片"
        disabled={!canNext}
        onClick={onNext}
      />
      <IconActionButton
        className="interaction-palette-tool"
        data-tool="search"
        icon={<Search size={13} />}
        title="打开图片搜索"
        onClick={onSearch}
      />
      <IconActionButton
        className="interaction-palette-tool"
        data-tool="reset"
        icon={<RotateCcw size={13} />}
        title="复位当前组合视口"
        onClick={onResetViewport}
      />
      <span className="interaction-palette-tool passive" data-tool="pan" title="拖拽画布空白区域平移">
        <Move size={13} />
      </span>
    </div>
  );
}
