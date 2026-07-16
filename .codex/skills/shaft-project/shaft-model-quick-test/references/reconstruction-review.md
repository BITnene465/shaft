# Reconstruction Review Reference

仅在用户要求临时 review reconstruction 推理结果、重建 render/overlay、生成浏览页面时读取。

## 边界

- 不把 reconstruction review 的 render、overlay、HTML 生成逻辑长期维护在
  `src/shaft`、`scripts/tasks`、正式 CLI 或测试里。
- 每次 review 时可以在 shell 里临时写一次性 Python 代码，输出到 `temp/<run>_review/`
  或当前 review artifact 目录。
- skill/reference 只维护规范与注意事项，不维护可执行脚本。
- 不为了临时 review 增加单测；只有沉入正式框架能力时才补测试。

## 推荐 Artifact

- `source/<task>/<safe_id>.*`
- `crop/<task>/<safe_id>.*`
- `overlay/<task>/<safe_id>.*`
- `render/<task>/<safe_id>.png`
- `json/<task>/<safe_id>.json`：完整记录，保留为链接。
- `prediction_json/<task>/<safe_id>.json`：页面主视图只展示 `prediction` 对象。
- `*_index.html`：临时浏览页，支持图片放大、拖拽、滚轮缩放。

## 推理请求契约

- Qwen3VL 的训练、eval 和运行时推理统一使用 image-first。用户消息中的图片必须在文本指令
  之前。
- 本地 HF/chat template 形态使用：`[{"type": "image"}, {"type": "text", "text": prompt}]`。
- OpenAI/vLLM 兼容 API 形态使用：
  `[{"type": "image_url", "image_url": {"url": ...}}, {"type": "text", "text": prompt}]`。
- 临时 eval/review 脚本不得改成 text-first。若新建 summary 或 manifest，记录
  `message_order: image_first`，方便之后排查 run 之间的请求契约差异。
- 对比不同 checkpoint 或不同 run 前，先确认 prompt、pixel budget、generation 参数、parser
  口径以及 `message_order` 一致。

## 页面展示

- 单条样本优先使用：
  - 左侧四图：`source / crop / overlay / render`
  - 右侧：只展示 `prediction`，不要把 `source_ann/raw_text/latency/artifacts` 等完整
    JSON 噪声塞进主视图。
- 完整 JSON 只作为 header 链接保留。
- 四图区域用稳定 2x2 网格；render 使用棋盘格背景以检查透明 PNG。
- 图片应可点击放大，放大后支持移动和缩放。

## Render 规则

- render 输出优先使用透明背景 PNG。
- review render 不必沿用原 crop 的低分辨率；应按 crop 尺寸等比放大到适合肉眼检查
  的分辨率，通常长边约 1000-1400px，极小图至少放大到短边约 220-300px。
- 只根据模型预测字段渲染，不从 GT 或 relax crop 中补几何。
- border、fill 颜色按预测忠实渲染；缺失字段才用清晰 fallback。
- P0 几何必须尽量完整支持：
  - shape: `rectangle / oval / triangle / trapezoid / parallelogram / diamond / step /
    regular_pentagon / regular_hexagon / arrow_pentagon / other_polygon / callout`。
  - shape 几何优先使用预测里的 `corners / body_corners / body_bbox / tail.points`。
  - `other` 只输出空透明图或明确 unsupported，不从 crop bbox 虚构形状。
  - oval 只有在预测中存在明确几何字段时才按该几何渲染；普通 oval 可按完整
    normalized crop box作为该 DSL 的隐式主体，但不要用 relax 后额外区域扩大 overlay。
- P1 风格也需要服务日常 review：
  - shape border `style=solid|dash|dot` 都要可见。
  - shape fill `solid / linear_gradient / radial_gradient / none / complex` 都要有明确处理；
    `complex` 可以用可识别 fallback，但不能静默当成普通 solid。
  - `effect=shadow|glow` 要渲染近似效果或显示明确 warning。
  - line `dash_style=dash` 要按路径虚线渲染。
  - line shape 的两色 `fill_color=["#...","#..."]` 要用渐变近似，不要只取第一个颜色。
  - line border `border_style=solid|dash` 要影响外轮廓。
- 圆角必须按圆弧/曲线渲染，不要用直角或采样点替代视觉语义。
- line reconstruction 中，`line_type=curved` 的点是曲线上的采样点，不是 Bezier
  控制点；4 点通常按 `t=0, 1/3, 2/3, 1` 的三等分采样点拟合穿点曲线。
- line reconstruction 中，`corner_style=round` 必须显式影响 straight polyline：
  对内部顶点做圆角化；`corner_style=sharp` 才保留折线尖角。
- line endpoint arrow 必须覆盖 `none / line / stealth / triangle / pointy / tee / circle`；
  具体比例可以是 review 近似，但不能全部退化成同一种三角箭头。
- 多段 line (`is_single=false`) 要逐段渲染，不要只取第一段。

### P1 风格字段清单

临时 renderer 写之前先从当前 JSON 统计字段值域，再按下面入口消费。没有出现在当前
run 的字段可以不实现，但 reference 中已有字段不能静默忽略。

- shape `parameters.border`
  - `type=none`：不画边框。
  - `type=uniform`：读取 `color / style`。
  - `style=solid`：连续描边。
  - `style=dash`：沿 shape path 虚线描边。
  - `style=dot`：沿 shape path 点状描边。
  - 其他值：在 review 页面或 render fallback 中显式可见，不能伪装成 solid。
- shape `parameters.fill`
  - `type=none`：透明主体。
  - `type=solid`：读取 `color`。
  - `type=linear_gradient`：读取 `colors / direction`；方向可以近似，但颜色顺序要保留。
  - `type=radial_gradient`：读取 `colors / direction`；`center_to_edge` 至少要表现中心到边缘渐变。
  - `type=complex`：使用可识别 fallback，例如斜纹/网纹；不要假装为单色。
- shape `parameters.effect`
  - `type=none`：不额外渲染。
  - `type=shadow`：用半透明偏移阴影近似。
  - `type=glow`：用主体 mask 的模糊外发光近似。
- line `parameters`
  - `dash_style=dash`：路径或 line shape 都要体现虚线。
  - `fill_color` 为字符串：用该颜色填充线体。
  - `fill_color` 为两色数组：用两色渐变近似线体，不只取第一个颜色。
  - `has_border=true`：读取 `border_color / border_style` 并画外轮廓。
  - `border_style=dash`：边框虚线化；`solid` 连续描边。
  - `begin_arrow / end_arrow`：端点类型影响形状，不只影响长度。
  - `corner_style=round`：straight polyline 内部顶点圆角化。

P1 风格的目标是让 reviewer 能判断模型是否预测对了风格类别，不追求和 Office/Canva
完全像素一致；但颜色、虚实、渐变方向、发光/阴影、有无边框这些语义必须可见。

## Overlay 规则

- overlay 用于检查预测几何和原 crop 的相对关系，不是标注真源。
- 只可视化显式控制点，不显示曲线采样点。
- 圆角有三个显式点时必须都显示：`start / mid / end`。
- 不要把文字标签压在图上；优先用颜色和 marker 形状区分点类型，并在页面放 legend。
- 如果确实需要编号，放在图外或留足 padding，不能遮挡图像内容或被裁切。
- relax crop 只影响输入/可视范围，不应让 overlay 几何变大；尤其不能把无控制点
  oval 画成被 relax 后的外接椭圆。

## 临时实现提醒

- 先读当前完整 JSON，确认预测字段值域，再写临时 renderer。
- 坐标系要先验证：常见为 0-1000 归一化坐标映射到 crop 像素。
- 重建 HTML 时给新 overlay/render URL 加 cache-busting query，避免浏览器缓存旧图。
- 一次性代码可以直接在当前 turn 执行，不保留到长期脚本目录。
