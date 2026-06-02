import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight
} from "lucide-react";

import { TextInputControl } from "./controlPrimitives";
import { IconActionButton } from "./ui";

import "./compositeImageJumpControl.css";

export function CompositeImageJumpControl({
  imageCount,
  jumpDraft,
  canPrevious,
  canNext,
  onJumpDraftChange,
  onSubmitJump,
  onJump,
  onStep
}: {
  imageCount: number;
  jumpDraft: string;
  canPrevious: boolean;
  canNext: boolean;
  onJumpDraftChange: (value: string) => void;
  onSubmitJump: () => void;
  onJump: (index: number) => void;
  onStep: (delta: -1 | 1) => void;
}) {
  return (
    <div className="image-jump-control" aria-label="图片跳转控制">
      <div className="image-jump-step-group" aria-label="向前跳转">
        <IconActionButton
          className="image-jump-step edge"
          title="第一张"
          icon={<ChevronsLeft size={14} />}
          disabled={!canPrevious}
          onClick={() => onJump(0)}
        />
        <IconActionButton
          className="image-jump-step"
          title="上一张"
          icon={<ChevronLeft size={14} />}
          disabled={!canPrevious}
          onClick={() => onStep(-1)}
        />
      </div>
      <div className="image-jump-field">
        <TextInputControl
          className="image-jump-input"
          label="Jump"
          type="number"
          min={1}
          max={Math.max(1, imageCount)}
          value={jumpDraft}
          onChange={onJumpDraftChange}
          onBlur={onSubmitJump}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              onSubmitJump();
            }
          }}
        />
        <strong>/ {imageCount.toLocaleString()}</strong>
      </div>
      <div className="image-jump-step-group" aria-label="向后跳转">
        <IconActionButton
          className="image-jump-step"
          title="下一张"
          icon={<ChevronRight size={14} />}
          disabled={!canNext}
          onClick={() => onStep(1)}
        />
        <IconActionButton
          className="image-jump-step edge"
          title="最后一张"
          icon={<ChevronsRight size={14} />}
          disabled={!canNext}
          onClick={() => onJump(imageCount - 1)}
        />
      </div>
    </div>
  );
}
