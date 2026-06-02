import type { ImageMapBin } from "./compositeImageNavigationModel";
import { CompositeImageAtlas } from "./compositeImageAtlas";
import { useCompositeImageAtlasController } from "./compositeImageAtlasController";
import { CompositeImagePanelHeader } from "./compositeImagePanel";

export function CompositeImageAtlasPanel({
  imageIndex,
  imageKey,
  filteredCount,
  imageMapBins,
  onJump
}: {
  imageIndex: number;
  imageKey: string;
  filteredCount: number;
  imageMapBins: ImageMapBin[];
  onJump: (index: number) => void;
}) {
  const atlas = useCompositeImageAtlasController({
    imageMapBins,
    onJump
  });

  return (
    <section className="image-jump-atlas-panel" aria-label="图片空间索引">
      <CompositeImagePanelHeader
        title="Image Atlas"
        meta={`${(imageIndex + 1).toLocaleString()} / ${filteredCount.toLocaleString()} matches`}
      />
      <CompositeImageAtlas
        imageIndex={imageIndex}
        imageKey={imageKey}
        filteredCount={filteredCount}
        imageMapBins={imageMapBins}
        {...atlas}
      />
    </section>
  );
}
