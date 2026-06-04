import type { CompositeSampleView } from "./api";
import { CompositeImageNavigator } from "./compositeImageNavigator";
import type { ActiveLayerConfig } from "./compositeReportModel";
import { useCompositeReportStageController } from "./compositeReportStageController";
import { CompositeStageWorkbench } from "./compositeStageWorkbench";
import { errorMessage } from "./formatters";

import "./compositeReportStage.css";

export function CompositeStage({
  composite,
  layerConfigs,
  loading,
  refreshing,
  error,
  enabled,
  onImageIndexChange,
  activeSlotCount,
  readyLayerCount,
  missingLayerCount,
  focusedLayerKey,
  onFocusedLayerChange
}: {
  composite?: CompositeSampleView;
  layerConfigs: ActiveLayerConfig[];
  loading: boolean;
  refreshing: boolean;
  error: unknown;
  enabled: boolean;
  onImageIndexChange: (index: number) => void;
  activeSlotCount: number;
  readyLayerCount: number;
  missingLayerCount: number;
  focusedLayerKey: string | null;
  onFocusedLayerChange: (layer: string | null) => void;
}) {
  const stage = useCompositeReportStageController({
    composite,
    focusedLayerKey,
    onFocusedLayerChange
  });

  return (
    <section className="composite-report-stage-card">
      {!enabled ? (
        <div className="composite-empty-state">至少选择两个可见 run 图层。</div>
      ) : loading ? (
        <div className="composite-empty-state">正在加载 sample。</div>
      ) : error ? (
        <div className="composite-empty-state danger-text">{errorMessage(error)}</div>
      ) : !composite ? (
        <div className="composite-empty-state">暂无组合视图。</div>
      ) : (
        <>
          {stage.layers.length === 0 ? (
            <div className="composite-empty-state">当前图片没有任何可渲染预测图层。</div>
          ) : (
            <>
              <CompositeStageWorkbench
                stage={stage}
                layerConfigs={layerConfigs}
                composite={composite}
                activeSlotCount={activeSlotCount}
                readyLayerCount={readyLayerCount}
                missingLayerCount={missingLayerCount}
                refreshing={refreshing}
                onFocusedLayerChange={onFocusedLayerChange}
                navigator={<CompositeImageNavigator composite={composite} onImageIndexChange={onImageIndexChange} />}
              />
            </>
          )}
        </>
      )}
    </section>
  );
}
