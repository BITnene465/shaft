import type { ImageJumpItem } from "./compositeImageNavigationModel";
import { CompositeImageJumpSummary } from "./compositeImageJumpItem";
import { ActionButton } from "./ui";

import "./compositeImageSearchPreview.css";

export function CompositeImageSearchPreview({
  activeResult,
  onJump,
  onClose
}: {
  activeResult: ImageJumpItem | null;
  onJump: (index: number) => void;
  onClose: () => void;
}) {
  if (!activeResult) {
    return null;
  }
  return (
    <ActionButton
      variant="mini"
      compact
      className="image-jump-active-preview"
      title={`${activeResult.index + 1}. ${activeResult.image}`}
      onClick={() => {
        onJump(activeResult.index);
        onClose();
      }}
    >
      <CompositeImageJumpSummary item={activeResult} badge="Active" compact />
      <i>Enter / Click</i>
    </ActionButton>
  );
}
