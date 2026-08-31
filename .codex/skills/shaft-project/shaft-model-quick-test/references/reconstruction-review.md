# Layout Inference and Reconstruction Review Reference

在用户要求 task-local 真实模型推理、layout recognition detection/reconstruction 评测、结果 review、重建
render/overlay 或浏览页面时读取。推理前门禁同时约束 detection 和 reconstruction，不允许只对齐后者。

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

### 推理前确认门禁

每次真实推理都必须先向用户展示本次完整合同并获得确认，包括 canary；历史 run、训练配置或本 reference 的
推荐值都不能代替本次确认。确认前只能做只读检查、配置准备和 dry-run，不启动 vLLM generation。

必须逐项对齐：

- 模型族、精确 checkpoint、finetune/merge 状态、served-model 名称。
- 数据集名称与 revision、样本范围、是否使用 GT；检测和 reconstruction proposal 的来源。
- prompt pool 的版本、formulation、variant 和最终 `prompt_id`；v5.7 checkpoint 不得静默使用 v5.8 prompt，
  反之亦然。
- thinking/reasoning：`enable_thinking`、适用时的 `preserve_thinking` 与 `reasoning_effort`。
- detection/reconstruction 分阶段的 `min_pixels/max_pixels`、smart-resize 真源、是否要求原生分辨率。
- crop 的 `padding_ratio`、`minimum_crop_size`、坐标空间和回映射方式。
- generation：`do_sample`、`temperature`、`top_p/top_k`、`max_tokens`、stop、重复抑制、结构化/受约束
  decoding。
- retry/fallback、失败 attempt 的保留方式、resume/force/overwrite、输出目录和上传范围。
- GPU、tensor parallel/replica 方式、canary 样本与正式扩量门禁。
- 多 replica 调度方式：正式批次默认使用共享动态 endpoint-slot 队列，明确每 endpoint 的最大 in-flight；固定
  sample-ID 路由只允许用于要求严格 endpoint 复现的诊断，不作为高吞吐默认值。

对齐时给出“推荐值 + 理由 + 与训练合同的差异”，等待用户确认；不要只问一个笼统的“是否开始”。

### Layout recognition 推荐提案

历史 real_v2 两阶段成功 run 的 detection/reconstruction 都使用了 1M–4M；后续 real_v1/real_v2 detection
像素预算实验覆盖了 0.5M–2M、0.5M–4M、1M–4M。用户最终确定未来默认提案为 detection 0.5M–4M、
reconstruction 0.5M–4M，并优先采用训练更充分、invalid 更稳定的 v5.8 27B ckpt8000。两个阶段即使当前
数值相同，也必须分别列出和确认，不能用一个范围隐去阶段合同；历史 run 仍按其真实配置报告。

- detection：Shaft 统一 Qwen smart resize，默认提案为 `500,000–4,000,000` pixels。real_v1/real_v2
  对照支持保留 4M 上限；0.5M 与 1M 下限的差异未形成跨 checkpoint 一致显著性，最终按用户决策采用
  与训练下限一致的 0.5M，并优先使用训练更充分的 ckpt8000。
- reconstruction：默认提案为 `500,000–4,000,000` pixels；`padding_ratio=0.65`、
  `minimum_crop_size=256`。
- Qwen smart resize 只执行一次；vLLM 不再私自解释 pixel budget，也不做第二次业务 resize。
- 结构化确定性评测的推荐提案是关闭 thinking，`do_sample=false`、`temperature=0`、`top_p=1`、
  `max_tokens=8000`，并使用 backend 的中性 generation config；这些仍需逐次确认。
- prompt 必须与 checkpoint 版本匹配，并把最终 `prompt_id` 写进每条 raw artifact 和 summary。
- 只有 `finish_reason=stop`、完整 JSON、task schema 和跨字段 contract 均通过时才安装 parsed 结果。
- fallback 前保留原始失败 attempt；summary 分别报告 generation error、parser contract violation、geometry
  violation 和 fallback，不能合并成“解析成功”。
- 用户确认 invalid 直接跳过后，失败项只保存 error artifact，resume 复用该失败状态，不再对确定性长输出反复
  重试。评测允许 prediction 缺失，但必须把它计为 `parse_ok=false` 和全量 GT 的 FN，同时在 provenance 保存
  missing count/stems；不得写一个伪造的空 prediction JSON 来冒充模型正常返回。
- canary 通过后才扩到全集；正式结果逐请求审计 pixel budget、prompt、generation、GT 隔离、ID 集和输出 hash。
- 多单卡 vLLM replica 不是一个跨进程 scheduler。批量 runner 必须维护共享动态 slot 队列：每个 endpoint 可按
  `max_num_seqs` 暴露若干 in-flight slot，请求完成即归还 slot，等待任务由先空闲的 slot 领取。禁止用 request ID
  哈希固定绑定 replica，否则长尾请求会造成“一卡满载、其余卡空闲”。
- 发布只上传约定的 prediction JSON；score/method metadata 是否上传必须在本次合同中确认。

对 reconstruction，0.5M 相比无下限/processor 65K 是关键修复；从 0.5M 增到 1M 属于额外质量余量。若怀疑
tiny shape 退化，先按目标尺寸分桶做 0.5M/1M canary，再决定是否提高 reconstruction 下限。该结论不能外推到
detection；detection 也独立采用 0.5M 下限，但两个阶段的默认值必须分别维护和确认。

- Qwen3VL 的训练、eval 和运行时推理统一使用 image-first。用户消息中的图片必须在文本指令
  之前。
- 本地 HF/chat template 形态使用：`[{"type": "image"}, {"type": "text", "text": prompt}]`。
- OpenAI/vLLM 兼容 API 形态使用：
  `[{"type": "image_url", "image_url": {"url": ...}}, {"type": "text", "text": prompt}]`。
- 临时 eval/review 脚本不得改成 text-first。若新建 summary 或 manifest，记录
  `message_order: image_first`，方便之后排查 run 之间的请求契约差异。
- 对比不同 checkpoint 或不同 run 前，先确认 prompt、pixel budget、generation 参数、thinking、parser
  口径以及 `message_order` 一致；将差异写进 comparison summary，不能只凭目录名推断。

## 页面展示

- 单条样本优先使用：
  - 左侧四图：`source / crop / overlay / render`
  - 右侧：只展示 `prediction`，不要把 `source_ann/raw_text/latency/artifacts` 等完整
    JSON 噪声塞进主视图。
- 完整 JSON 只作为 header 链接保留。
- 四图区域用稳定 2x2 固定网格；source/crop/overlay/render 必须等比 `contain` 在窗格内，不能因原图过大产生
  页面或图片区域滚动。右侧完整 JSON 可以独立滚动。
- 如需细看，可点击进入独立 modal 放大、移动和缩放；modal 不能改变主 review 页固定布局。

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
  - 单对象 reconstruction review 中，`other` 只输出空透明图或明确 unsupported，不从 crop bbox 虚构形状。
    端到端最终合成的目标不同：为避免已检测对象静默消失，`shape_type=other` 必须像 icon/image 一样直接粘贴
    预测 bbox 对应的原图 crop，并在 summary 记录 `shape_other crop fallback` 数量。该 fallback 只消费模型
    detection bbox 和原图，不补 GT、不虚构几何，也不得写回 prediction parameters。
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

### Line endpoint marker 可迁移合同

本节是 reconstruction review 的规范真源。临时 renderer 可以删除或重写，后续实现仍必须仅凭本节恢复相同
视觉语义，不能依赖某个 `temp/` 脚本、历史 HTML、固定仓库外路径或未追踪的生产 bundle。

实现前按以下顺序确认合同：

1. 读取本次 checkpoint 对应的 line prompt、parser/validator 和当前完整 prediction JSON。
2. 统计 `line_style × begin_arrow × end_arrow` 值域及非法组合数量，不能只看 schema 枚举。
3. 如果当前生产编辑器有独立视觉实现，用它做交叉核验；冲突时先确认 schema 版本，不把生产代码路径写成
   renderer 的运行时依赖。

端点使用 path 的有向顺序。begin 的 `tip=p[0], neighbor=p[1]`，end 的
`tip=p[-1], neighbor=p[-2]`。令单位方向 `u=(tip-neighbor)/|tip-neighbor|`，法向量
`n=(-u_y,u_x)`。各枚举必须产生可区分的形状：

| marker | 合法 line_style | 可迁移视觉语义 |
| --- | --- | --- |
| `none` | path / shape | 不画端点。 |
| `line` | path | 以 tip 为尖端的开口 V；只描边，不闭合、不填充。 |
| `triangle` | path / shape | 以 tip 为尖端、后缘平直的实心三角形。 |
| `stealth` | path / shape | `tip → left → rear-notch → right` 的实心凹尾箭头；不得退化成 triangle。 |
| `pointy` | shape | filled shape 的锥形端点；它表示主体逐渐收窄到 tip，不是 path marker。其横向半宽应明显窄于同尺度 triangle，可用 `0.72 × triangle_half_span` 近似。 |
| `tee` | path / shape | 以 tip 为中心、沿法向量 `n` 的垂直端帽。 |
| `circle` | path / shape | 以 tip 为圆心的空心圆；主体路径应到达圆心。 |

filled polygon marker（`triangle / stealth / pointy`）会取代路径末端的一段主体。绘制主体时先沿 neighbor 方向
把端点内缩，推荐内缩约 `0.72 × marker_length`，再拼接 marker；否则粗 body 会穿到 tip，pointy 看起来仍是
平头。`line / tee / circle` 以标注端点为中心，主体不内缩。shape 的 border 必须同时覆盖 body 与 marker；
两色 fill 的 begin/end marker 分别使用对应端颜色。

marker 尺寸优先读取显式 body width、marker width/length。字段不存在时才允许从中心线和轴对齐 bbox 近似：

- 设 bbox 宽高为 `W,H`，中心线首尾横纵跨度为 `dx,dy`，`L=hypot(dx,dy)`。
- 当 `|dy|/L > ε` 时得到候选横向厚度 `T_x=(W-dx)/(|dy|/L)`；当 `|dx|/L > ε` 时得到
  `T_y=(H-dy)/(|dx|/L)`。
- 只保留正候选并取较小值；无正候选时才 fallback 到 `min(W,H)`。body width 与 marker 总横宽要分别估计，
  不得相等复用。
- 禁止直接把轴对齐 bbox 四角投影到法向量后当横向厚度；斜线的路径长度会混入该投影，使箭头头部异常放大。
- path marker 尺寸只跟 stroke width 和可见性下限相关，不使用整个 bbox 的法向跨度。

非法组合必须显式处理，不能静默伪装成模型预测正确：

- `path + pointy`：标为 contract violation；如页面要与生产编辑器预览一致，可仅在视觉层归一为
  `triangle`。
- `shape + line`：标为 contract violation；可仅在视觉层归一为 `triangle`。
- raw prediction、右侧 JSON 和统计必须保留原值，视觉归一不得写回 prediction 或改变指标。
- 未知 marker 使用明确 unsupported/warning，不默认画成 triangle。

全量重建前必须生成或抽取以下 canary：

- 合法 marker matrix：七种 marker 至少覆盖 begin/end；`line` 覆盖 path，`pointy` 覆盖 shape，其余覆盖实际
  出现的 style。
- 旋转矩阵：至少检查 `0° / 30° / 45° / 90°` 的 triangle、stealth、pointy，防止 bbox 估宽只在水平线正确。
- straight / curved / rounded polyline / multi-segment 各一例，确认切线方向与端点顺序。
- 有 border、dash、两色 fill 的 shape arrow 各一例，确认 marker 与 body 的 paint 一致。
- 浏览器 canvas 和静态导出若同时存在，必须对同一 canary 使用相同枚举语义、非法组合策略和尺寸公式。

验收记录至少包含值域统计、非法组合计数、canary 列表、人工抽查结论、图片解码检查和页面脚本检查。图片能
生成、JPEG 能解码或没有 JavaScript 异常，只能证明程序可运行，不能证明 marker 语义正确。

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
- 一次性代码可以直接在当前 turn 执行，不保留到长期脚本目录；可迁移规则必须回写本 reference，不能只留在
  临时脚本或聊天记录中。
