import type { CompositeSampleView } from "./api";
import { CompositeImageJumpControl } from "./compositeImageJumpControl";
import { basename } from "./formatters";

export function CompositeImageNavigatorPrimary({
  composite,
  jumpDraft,
  canPrevious,
  canNext,
  onJumpDraftChange,
  onSubmitJump,
  onJump,
  onStep
}: {
  composite: CompositeSampleView;
  jumpDraft: string;
  canPrevious: boolean;
  canNext: boolean;
  onJumpDraftChange: (value: string) => void;
  onSubmitJump: () => void;
  onJump: (index: number) => void;
  onStep: (delta: -1 | 1) => void;
}) {
  return (
    <div className="image-navigator-primary">
      <div className="image-navigator-copy">
        <span>Image Union</span>
        <strong title={composite.image_key}>{basename(composite.image_key)}</strong>
        <em title={composite.image_key}>{composite.image_key}</em>
      </div>
      <div className="image-navigator-actions">
        <CompositeImageJumpControl
          imageCount={composite.image_count}
          jumpDraft={jumpDraft}
          canPrevious={canPrevious}
          canNext={canNext}
          onJumpDraftChange={onJumpDraftChange}
          onSubmitJump={onSubmitJump}
          onJump={onJump}
          onStep={onStep}
        />
      </div>
    </div>
  );
}
