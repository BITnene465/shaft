import { imageProgressPercent } from "./compositeImageNavigationModel";
import { CompositeMicroMeter } from "./compositeMicroMeter";

import "./compositeImageIndexMeter.css";

export function CompositeImageIndexMeter({
  imageIndex,
  imageCount
}: {
  imageIndex: number;
  imageCount: number;
}) {
  const progress = imageProgressPercent(imageIndex, imageCount) / 100;
  return (
    <CompositeMicroMeter
      className="image-index-meter"
      label="Image"
      value={(imageIndex + 1).toLocaleString()}
      meta={`/ ${imageCount.toLocaleString()}`}
      progress={progress}
      ariaLabel={`当前组合图片 ${imageIndex + 1} / ${imageCount}`}
    />
  );
}
