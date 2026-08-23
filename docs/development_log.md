# Shaft 开发日志

本文档记录已经暴露过的工程问题、语义偏差和后续防线。每条记录至少包含现象、根因、影响范围、
修复方式、回归测试和后续防线；结论不能只保留在聊天、临时日志或本机环境中。

历史开发日志在 squash 前的仓库历史中仍可审计；本文件从当前 HF-first 主线继续维护。

## 2026-07-16：本地环境掩盖 clean-runner 测试依赖与布尔契约缺口

### 现象

最终 squash commit 在开发机上通过 framework、smoke 和 distributed 回归，但首次推送 `main` 后：

- `framework-ci` 有 7 个失败；
- `framework-runtime` 有 4 个 distributed 失败；
- 两个公开运行时边界允许非布尔 truthy 值进入安全相关能力判断。

### 根因

测试隔离问题有三类：

1. 部分测试写死 `models/Qwen3-VL-4B-Instruct`、`models/Qwen3.6-27B`。开发机存在未跟踪模型目录，
   clean runner 不存在时，相对路径被当作 Hub repo 并触发远程 config 解析。
2. Qwen3.5/Qwen3.6 varlen 正路径测试依赖开发机已安装的 CUDA isolation kernels；GRPO vLLM 测试使用
   `dict.get(key, distribution_version(key))`，其默认参数会被 eager 求值，导致 clean runner 查询未安装的
   vLLM distribution。
3. distributed support 脚本以文件路径交给 `torchrun` 后，子进程的 `sys.path` 只保证脚本目录，未保证
   仓库根目录；开发机 editable 环境掩盖了 `tests.support` 无法导入的问题。

另外两个生产边界使用了 Python truthiness：

- `allow_unverified_base_model="false"` 会被当作真值，跳过 adapter/base provenance 验证；
- `ShaftInferAdapterCapabilities` 接受字符串形式的 capability，可能绕过 execution-control fail-closed。

### 影响范围

- 7+4 个 CI 失败属于测试环境契约缺失，不表示 Qwen、FSDP、GRPO 或 DDP 主链本身失效。
- provenance override 与 infer capability 属于真实运行时边界缺口；配置对象或第三方调用方传入非布尔值时
  可能错误放宽安全约束。
- 本次问题不涉及模型能力，也不属于 eval、codec、metric 或 data 误判。

### 修复方式

- 测试统一用 `tmp_path/config.json` 生成最小本地 HF descriptor，不依赖未跟踪模型资产或网络。
- 可选 CUDA kernel 与 vLLM distribution 在测试中通过已有 seam 显式模拟；生产运行时仍保持缺依赖即拒绝。
- distributed 测试启动器集中构造子进程环境，把仓库根目录前置到 `PYTHONPATH`，同时保留调用方原值，
  并继续隐藏 CUDA。
- adapter provenance override 和 infer adapter capability 使用 exact-bool 校验；字符串、整数和 `None`
  均在执行工作前失败。

### 回归测试

- 原 7 个 framework 失败用例与新增缺 kernel 负例：全部通过，并在
  `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 下复验。
- 原 4 个 distributed 失败用例：全部通过。
- `ruff check src/shaft tests`：通过。
- `python -m compileall -q src/shaft tests`：通过。
- `pytest -q tests --suite framework`：通过。
- `pytest -q tests --suite smoke`：通过。
- `pytest -q tests --suite distributed`：通过。

### 后续防线

- framework/smoke 测试不得依赖未跟踪的 `models/`、`configs/`、缓存目录或外网可达性。
- 测试可选后端时应模拟 package metadata，并保留独立的缺依赖 fail-closed 负例；真实 kernel correctness
  只在对应 GPU suite 验证。
- 任何 `torchrun` 测试 helper 必须显式建立 support-module import contract，不能依赖 editable install 的偶然路径。
- 安全豁免、能力声明、checkpoint/provenance override 等布尔边界必须使用 exact-bool 校验，禁止用
  `bool(value)` 解释外部输入。
- 最终候选必须在 clean runner 上通过 required CI；本机全绿不能替代远端门禁。

## 2026-07-17：v5.3 context reconstruction 与派生数据安全收口

### 现象

- v5.3 需要从既有 selection 恢复 shape/line/image 上下文重建样本，同时模拟有界 detector proposal
  偏差，并让 proposal 与 target geometry 共用当前 crop 的 Qwen `0..999` 坐标系。
- 真实 shape attribute weak label 中曾出现 prompt 合同外的嵌套字段；只检查 required value 会把
  `border.color2` 等 teacher extra 原样带入监督目标。
- grounding padding 在原图尺寸未按 processor factor 对齐、但又无需降采样时，metadata 使用了 floor
  后的 content size，实际像素却保持原尺寸，导致图像与 bbox 错位。现有 v5.3 派生数据中约576条
  padded row 命中该分支。
- 部分 task 测试曾导入 ignored `subTasks/` 脚本或依赖本地 ignored prompt pool；开发机可通过，clean
  checkout 无法复现。若 converter 在加载 prompt 前执行 `--clean`，缺文件还会先删除旧 SFT。

### 根因

- selection identity、source truth、weak-label truth 和 derived audit metadata 的边界此前没有统一到一个
  fail-closed builder 合同。
- padding 的 factor 对齐同时改变了声明 content geometry，却只在超预算分支执行真实 resize。
- prompt 版本通过脚本默认值静默切换，输入/split/path 校验发生在 destructive clean 之后。
- bbox minimum extent 最初由 task 脚本手工修补，未复用共享 coordinate codec。

### 影响范围

- raw 图片和人工 JSON 未被本轮代码修改；错位影响只在可重建的 grounding derived PNG/structured/SFT。
  受影响 padded rows 在继续训练前必须 clean rebuild，不能沿用旧派生结果。
- `image_context_reconstruction` 当前 selection 还有75个 source JSON 不在现行 `data/raw/json`，影响447行。
  builder 会明确失败并保留旧输出；这是本地数据版本可重建性问题，不允许静默回退 archive 真值。
- weak labels 只用于 train-only 辅助任务，不成为正式 eval truth。

### 修复方式

- 新增维护入口 `scripts/tasks/build_context_reconstruction_sft.py`：按 source 分组解码，重载 source truth，
  生成确定性 context crop/proposal、task-local media/selection/structured/SFT，并通过同盘 staging 原子发布。
- 新增共享 `shaft.data.context_attribute_contract.validate_shape_parameters`；weak sidecar consumer 执行 exact
  nested-key 与 API provenance gate，拒绝空 selection、非正 `--limit` 和缺少 provenance 的 weak rows。
- padding 仅在实际超预算时调整 content size并执行 resize；原生预算可容纳时保持原图像素和 bbox 尺寸。
- SFT converter 恢复 tracked v5.0 默认值，v5.3 使用显式 `--prompt-config TASK=PATH`；prompt 和全部
  structured split 在 `--clean` 前预检。v5.3 prompt pools继续按项目策略保持 local/ignored。
- grounding builder 在 clean 前检查 train/val overlap、缺失 GT、workers 和 raw/output path；synthetic sync
  builder拒绝输入输出路径重叠并记录真实 split provenance。
- `quantize_qwen_bbox` 增加可选 `minimum_extent_bins`，grounding/context bbox 使用共享1-bin最小尺寸；像素级
  zero-area bbox直接拒绝，量化碰撞再按 `label+bbox_2d` 稳定去重。
- tracked task tests全部使用测试内最小 prompt fixture，不再读取 ignored配置或本地子项目脚本。

### 回归测试

- `ruff check src/shaft scripts/tasks tests`：通过。
- `python -m compileall -q src/shaft scripts/tasks tests`：通过。
- `pytest -q tests --suite task`：通过。
- `pytest -q tests --suite framework`：通过。
- focused 回归覆盖非对齐 native padding、低预算 resize、clean 前置预检、split/path 重叠、zero/empty guard、
  weak exact-schema/provenance、共享 codec edge extent 和 worker failure旧输出保留。
- 真实 clean weak sidecar 19,709/19,709 行通过新增 provenance gate。

### 后续防线

- grounding v5.3 必须先重建受影响的 padded/structured/SFT 后再训练，并重新核对图片引用、bbox、row count、
  test overlap 和 stale files；raw 不需要回写。
- image context 重建前必须恢复或版本化缺失的75个 source JSON；不得因为当前 raw 缺失而自动把 archive
  当作真值。
- prompt pool可以是本地运行资产，但 tracked tests只能使用最小 fixture；任何 destructive clean 都必须在
  所有输入、schema、split 和路径预检完成后执行。
- context builder 当前会物化 selections/work items；269,904 shape rows实测增加约214MB RSS，属于后续流式化
  的 P2 优化，不阻断当前离线构建。

## 2026-07-17：真实 shape 属性扩量与非法 teacher 输出清洗

### 现象

- Opus 4.8 对30,000个独立真实 shape context crop 批量标注时，部分响应包含合同外字段、缺失条目、
  非法颜色或无法解析的文本；Bedrock 长尾期间还出现 API read timeout。
- 若只检查必填字段，`border.color2`、`border.fill`、`fill.effect` 等 teacher extra 会进入训练目标；若为
  缺失颜色或结构补默认值，又会把解析策略误当成视觉真值。

### 根因

- 外部 teacher 只能提供弱监督，不能保证严格遵循 nested-key schema；并发请求还受到区域服务容量和
  请求尺寸影响。
- API schema-valid、人工语义正确和可进入 SFT 是三个不同门槛，不能用成功 HTTP 响应替代训练合同验证。

### 影响范围

- 新一轮30,000条 prediction 中，29,148条通过 exact-schema gate；716条 schema 非法、136条 API error。
- 非法与超时记录只存在于临时 prediction audit，不进入 canonical weak-label sidecar、selection、
  structured 或 SFT。人工 raw JSON 和正式 eval truth 未被改写。

### 修复方式

- 使用共享 `validate_shape_parameters` 对每条结果执行 exact nested-key 校验；非法行直接删除，不移动字段、
  不猜测缺失值、不填默认颜色。
- 将29,148条有效增量与原19,709条 clean sidecar 按 `sample_id` 零重合合并，形成48,857条 canonical
  weak labels，并保留合并前备份和独立增量快照。
- 派生任务保留全部11,029条非矩形，确定性采样11,029条矩形，使 rectangle严格占50%；clean rebuild
  生成22,058条 train-only `shape_context_attributes` SFT，并把媒体快照升级为
  `banana-v5.3-media-v7`。

### 回归测试

- 30,000个 prediction 文件全部可解析且 ID 属于冻结 selection；有效行再次通过共享合同，合同错误为0。
- merged sidecar 共48,857个唯一 `sample_id`，旧/新增 overlap为0；所有 source JSON/image/index/bbox
  可追溯。
- selection、structured、SFT与task-local PNG均为22,058，validation为空；target不含任何 geometry字段。

### 后续防线

- API 成功只能进入候选池；任何 weak-label promotion 必须重新执行共享 exact-schema gate。
- `other` 只能包含 `shape_type`；非 `other` 只允许合同规定的 `border/fill/effect`，`callout` 才允许
  `body_type`。控制点、bbox和 tail/body geometry 不得进入本属性子任务。
- teacher API 的 rejected audit 与训练数据分离；不得为了提高保留率对非法输出做猜测性修补。

## 2026-07-18：历史主任务评测入口依赖漂移与 v5.3 复评收口

### 现象

- v5.3 最终权重准备复用历史正式 layout runner 时，入口在发出任何推理请求前因已删除的
  `projects/eval_bench` 模块导入失败；历史脚本已无法在当前 `src/shaft` 主链直接运行。
- 这属于评测基础设施失效，不是模型输出错误，也不代表旧指标本身发生变化。

### 根因

- 历史 runner 把像素预算、HTTP 请求适配和输出解析绑定到了已归档/删除的项目目录，没有把共享
  `codec` 与当前 Qwen3VL pixel-budget 实现作为稳定依赖。
- 仓库重构后缺少一条冻结 raw output 的 parity 回归，因而入口依赖漂移直到下一次正式复评才暴露。

### 影响范围

- 只影响历史 runner 在当前代码树的可执行性；失败发生在模型服务请求前，没有产生半套 v5.3 指标。
- v5.3 正式评测仍使用冻结的175张 testset、相同 IoU/去重口径和 image-first 请求；未修改 GT、预测或
  checkpoint。

### 修复方式

- runner 增加当前 Shaft Qwen3VL pixel-budget 工具、image-first HTTP adapter 和共享 JSON/coordinate
  codec 的兼容路径，保留历史 label normalization、IoU@0.5 matching 与 IoU@0.95 prediction dedupe 口径。
- 对 v5.3 1M/2M/4M 共525份 prediction 另写独立 matcher，从原始 GT 复算 TP/FP/FN、precision、recall、
  F1、sample F1 与 matched-box mIoU，不直接复用 runner 汇总。

### 回归测试

- 用 v5.2 checkpoint-12000 的2M历史 raw output 做 parity 回放：all/line/non-line/shape/icon/image 六组
  TP、FP、FN、F1、mIoU 和 parse 统计与历史 summary 完全一致。
- v5.3 三档独立复算与 runner 汇总在 `1e-12` 容差内一致；每档175个 sample ID 与冻结 testset一致，
  request error为0，pixel-budget越界为0。
- `uv run pytest -q tests/test_pixel_budget.py tests/test_codec.py tests/test_infer_engine.py -q`：37项通过。

### 后续防线

- 应把固定 benchmark runner 收口到维护中的 `scripts/tasks` 与共享 `infer/codec`，禁止正式入口继续依赖
  ignored、archive 或已删除模块。
- 为正式 benchmark 保留小规模冻结 raw-output parity fixture；框架重构涉及 infer/codec/pixel budget 时，
  必须同时验证“入口可运行”和“历史 metric 逐项不漂移”。
- 报告必须明确区分模型能力与 eval 基础设施：本次兼容修复不构成模型提分，v5.3 的指标结论来自修复后
  正式推理及独立复算。

## 2026-07-18：v5.3 全权重选优与 line `is_single` 语义契约补全

### 现象

- 只评测最终 checkpoint-12000 时会把它误当成默认部署权重；补齐2k–12k全 sweep 后，面向2M–4M的
  最优点实际是 checkpoint-8000，checkpoint-12000仅排第三。
- 剩余任务首次汇总虽然 JSON 与字段合同通过率接近100%，抽检却发现个别 line 输出声明
  `is_single=false` 但 `points` 只有一个 segment；旧检查器把它误标成 contract valid。

### 根因

- 训练关闭 validation 且 `best/` 只是最终权重副本；logged loss 继续下降不能替代冻结主任务上的
  checkpoint selection。
- 临时 line 输出检查器只验证了 `is_single` 的布尔类型与 points 坐标合法性，没有落实 prompt/训练合同中
  “true恰好一个segment、false必须多个segment”的跨字段约束。

### 影响范围

- checkpoint-12000 的历史预测与指标本身没有计算错误，但“它是 v5.3 最佳部署点”的结论失效；2M–4M
  排序改为 checkpoint-8000、10000、12000。
- line reconstruction/points 的请求、JSON 解析和预测内容没有改变；只修正 contract-health 标记。修正后
  checkpoint-8000 两个 line 任务通过率为99.516%/99.692%，checkpoint-10000为99.692%/99.736%。

### 修复方式

- 对 checkpoint-2000/4000/6000/8000/10000/12000 全部执行1M/2M/4M同口径主任务评测；以2M与4M
  all-label F1非加权均值作为部署排序真源，并保留 minimum F1、mean mIoU 作为稳定性字段。
- line 临时评测 validator 增加 `is_single` 与 `points` segment 数量的一致性检查；不修补模型输出，只从
  原始 prediction 重新计算 contract errors/summary，并重建 paired review 状态。

### 回归测试

- 18组主任务结果均为175/175请求成功、预算不越界；独立 matcher 从3,150份 prediction 与原始GT复算，
  所有汇总与 runner 在 `1e-12` 内一致。
- 两个入选权重各11,253个剩余任务请求均无请求错误；revalidation后六个任务的 result count、summary与
  review逐项一致。paired review共11,253组、380个HTML入口/分页，本地引用缺失为0。
- 抽检 `is_single=false + one segment` 样本已显示为 contract invalid；renderer仍原样展示 prediction，
  没有用规则或GT改写输出。

### 后续防线

- 无 validation 的长训练不得用最终 checkpoint 或 `best/` 名称代替选优；正式训练报告必须在部署预算上
  sweep全部保存点，或明确说明没有完成 checkpoint selection。
- 结构化输出合同必须覆盖字段间语义关系，而不只做单字段schema检查；`is_single`、segment数量等约束应
  进入后续共享 codec/contract，而不是长期停留在临时评测脚本。
- contract valid只能表示输出满足机器合同，不能表示 reconstruction语义正确；缺少canonical DSL GT时，
  报告必须把自动健康统计与人工prediction-only review明确分开。

## 2026-07-19：line 预标注的 source-label 先验与病态 points 输出防线

### 现象

- 对 `json_20260706` 的150,503个line/arrow实例做双任务预标注时，少量 source `arrow` 的视觉推理结果
  两端均为 `none`，与原始类别“至少存在一个head”的数据集先验冲突；source `line` 则不应被模型误加head。
- 输出虽通过旧 points schema，但有长尾生成退化：最坏单segment含569点，其中563个是相邻重复点；另一个
  样本生成154个segment，其中大量segment完全相同。

### 根因

- reconstruction属性识别与points几何识别由两路prompt独立完成，head在低清、小目标或邻近干扰下可能
  无法仅凭crop稳定恢复；旧融合合同没有把原始 `line/arrow` 类别先验作为最终硬约束。
- 旧points合同只检查坐标范围、每段至少两个不同点和 `is_single` 关系，没有限制相邻重复点、重复segment
  或病态密集轨迹；因此“结构合法”被误当成“可安全写入训练标注”。

### 影响范围

- 原始raw JSON未被改写；问题仅影响本轮临时模型预标注候选。150,503条中，15条points几何被标记warning，
  9条最终需要bbox主轴高风险fallback，16条arrow在多级视觉head恢复后仍需数据集先验兜底。
- 病态轨迹会放大JSON、污染points监督并在overlay中形成虚假密集路径；它不是模型整体能力或主任务指标问题。

### 修复方式

- 融合阶段落实 source-label 硬约束：source `line` 强制两端 `none`；source `arrow` 强制至少一端non-none。
  arrow依次使用crop head recovery、full-image recovery、forced non-none endpoint type；全部失败时才使用数据集
  经验MAP `end_arrow=stealth`，并以独立 provenance 标记，禁止伪装成视觉判断。
- 属性合同与几何合同解耦：reconstruction几何失效时仍可保留合同有效的样式/颜色属性；几何优先使用
  `line_context_points`，其次reconstruction，最后才用bbox主轴并标高风险。
- finalize在不改写raw model audit的前提下去除非法点、相邻重复点、完全重复segment；straight路径用RDP
  保留弯折并限制单段最多64个代表点，curved仅在多于4点时按弧长采4点。每条清理统计写入派生JSON provenance。

### 回归测试

- focused测试覆盖line/arrow先验、crop/full-image/forced head recovery、属性与几何解耦、bbox高风险fallback、
  相邻重复点清理、curved四点采样和重复segment清理；`tests/test_prelabel_line_reconstruction.py` 18项通过。
- 全量推理结果150,503/150,503成功；source line 23,298条均无head，source arrow 127,205条均至少一端有head。
  finalize覆盖9,197 JSON，1,502条发生几何清理、共移除13,717点；原始response仍完整保留用于追溯。

### 后续防线

- line/arrow source先验必须进入共享合同或数据promotion gate，不能只写在prompt里；任何规则兜底都必须记录
  `arrow_head_source`，并进入人工review筛选。
- points validator需同时检查坐标/拓扑与生成退化：相邻重复点、重复segment、异常segment数和异常点密度。
  schema-valid只代表可解析，不代表几何可信。
- bbox轴fallback、经验head fallback和大幅几何清理样本不得无差别promotion；正式训练前必须先审阅对应筛选页。

## 2026-07-19：预标注交付误用了 raw schema，而非 `gt_standard`

### 现象

- line预标注首次finalize输出根层 `image_width/image_height/instances`，实例使用
  `label/bbox/extra.parameters`，还把模型、prompt、融合和清理审计写入 `extra.line_prelabel`。
- 这些字段适合raw维护或prelabel audit，但不符合用户指定的真实 `gt_standard` 示例；后者根层严格是
  `size/background/layout`，实例严格是 `type/bbox/parameters?`。

### 根因

- 实现时只对齐了line reconstruction的参数DSL，没有先从100,500份真实 `gt_standard` 统计完整容器schema，
  错把“参数内容正确”等同于“交付格式正确”。
- prelabel skill允许临时JSON携带丰富审计，但本任务要求的是可直接使用的 `gt_standard` 交付；两层用途没有
  在finalize边界明确分离。

### 影响范围

- 模型推理、150,503条line参数和原始raw都未受影响；错误只存在于首次临时交付JSON的包装层。
- 首版约522MB且混入不需要的审计字段，不能直接当作 `gt_standard` 消费。

### 修复方式

- finalize直接输出纯净schema：`size=[width,height]`，raw `background=true`映射为 `image`、缺失/false映射
  为 `none`，所有实例进入 `layout`；source line/arrow均写为 `type=line`，其他类型保持原类别并把已有
  `extra.parameters`提升为实例顶层 `parameters`。
- bbox使用覆盖式整数化，line points使用最近整数并在最终像素坐标再次去重；审计不进入annotation，原始
  response/retry继续留在外部 `results/`，聚合信息留在summary/review。
- 原子替换旧 `json/` 后删除内部backup，避免新旧两份交付并存。

### 回归测试

- 9,197/9,197份输出根键精确为 `size/background/layout`；480,416个实例只允许
  `type/bbox/parameters?`，全目录搜索不到 `extra/line_prelabel/instances/label/image_width/image_height`。
- 150,503个目标均为 `type=line`，line/arrow hard prior、整数points范围、segment拓扑、非目标参数保留全部
  通过；21条focused tests通过。

### 后续防线

- 用户提到某个具名格式时，必须先读取真实样例并同时验证根schema、实例schema和坐标类型，不能只对齐
  内层DSL。
- prelabel audit与最终annotation是两个产物：审计放外部sidecar/results，最终交付不得为了追溯方便混入
  `extra`。validator必须使用exact key sets，而不是只检查必填字段。

## 2026-07-21：shape 缺失 corners 时的 source-bbox 渲染 fallback

### 现象

- v5.3 两阶段临时review中，`test1 #59 shape` 的模型响应是完整合法JSON，但rectangle只输出
  `shape_type/border/fill/effect`，漏掉复原几何必需的 `corners`；contract正确标为
  `corners:must_be_nonempty`，旧renderer因此保持透明，看起来像“没有解析出来”。

### 根因

- 一阶段框覆盖了包含多个子对象的整块区域并接近crop右下边界，模型退化为只给属性、不输出几何。
- 旧renderer只有“使用预测corners”与“透明”两种路径，没有在已有请求source bbox时提供明确、可审计的
  降级渲染。这是模型输出缺字段叠加review能力缺口，不是codec JSON解析错误。

### 影响范围

- raw response、decoded prediction及contract统计都没有错误；问题只影响prediction-only review和全局
  复原图的可见性。bbox fallback只能补齐近似外接矩形，不能证明模型恢复了真实corners，也不能纠正
  border/fill等属性错误。

### 修复方式

- shape review renderer在polygon-like shape没有有效corners、但record带source bbox时，生成一个
  `bbox_rectangle`渲染几何，并返回 `reason/source/geometry/shape_type` fallback metadata。
- fallback严格停留在renderer：不写回prediction JSON、不改变contract状态；review同时显示
  `CONTRACT INVALID`和`render: source bbox fallback`。全局复原按shape bbox面积从大到小绘制，使大容器
  位于小对象下层。

### 回归测试

- 几何单测覆盖missing corners触发bbox path、有效corners不触发、shape other保持透明及alpha范围。
- 从既有结果重建checkpoint-8000/2M与checkpoint-10000/4M两套review；各验证2个fallback、HTML缺失链接0，
  两套均为`ready_to_review`。重建前后prediction/raw/decode指纹完全一致，证明没有污染模型结果。

### 后续防线

- review必须把“模型原生预测”与“展示fallback”分栏或显式标记；不得用fallback后可绘制冒充contract valid。
- 任何基于source bbox的几何都只能用于可视化/部署降级，自动指标仍使用原始prediction；若进入正式导出，必须
  单独记录provenance并允许调用方禁用。

## 2026-08-04：compact 人工标注到 grounding structured 的派生边界修复

### 现象

- 新的 `data/raw/json` 使用人工精简的 `size + layout[]` schema，旧 grounding builder 只读取
  `instances[].label`；直接重建会把有效 bbox 当作空目标。
- 旧 train split 仍指向上一批 17,065 个 JSON，无法覆盖当前 20,060 个清洗后 source。少量 JPEG 还存在
  EXIF 方向与磁盘解码尺寸不同，若统一转置或统一不转置都会让一部分 bbox 落入错误坐标系。

### 根因

- 派生脚本把 legacy normalized raw contract 当成唯一输入，没有在 raw/derived 边界显式适配 compact
  `layout[].type`，也没有区分 compact 中合法但不属于四类检测的 `full_text`。
- split 是旧数据快照的路径清单；图像加载只信任磁盘解码尺寸，没有使用 compact 顶层 `size` 验证坐标空间。

### 影响范围

- 影响当前 20,060 个 raw JSON 的 `grounding_layout` structured 重建，不涉及模型训练、推理、codec 或 metric。
- 19,953 个 source 含 `shape/icon/image/line` 目标；107 个 source 只有 `full_text`。三张 EXIF 图需要按声明
  `size` 条件转置。line 的人工 points 和所有 reconstruction parameters 均不属于本次检测 target。

### 修复方式

- builder 在同一入口兼容 normalized `instances[].label` 与 compact `size + layout[].type`，只派生四类
  `label + bbox`；排除 `full_text`，忽略 line points，不改写 raw 文件。
- compact 图像先比较磁盘尺寸与声明 `size`，仅在 EXIF transpose 后精确匹配时转置，否则 fail closed。
- 备份旧 train manifest 后，按当前 `json/` 重建 20,060 条 train split，并继续排除 canonical test id；
  无四类目标的 source 只保留一个 native empty row，不进入 resize、padding、degradation 或 crop 配额。
- 使用 40 进程生成 58,440 条 structured 数据：20,060 native、17,882 clean resize、1,995 padded、
  14,965 degraded、2,993 density crop、545 hard negative；本轮不生成 SFT。

### 回归测试

- `tests/test_build_grounding_structured.py` 12 项通过，覆盖两种 raw schema 的完整 CLI 路径、compact
  `full_text` 排除、条件 EXIF transpose、零目标 source 和多尺度计划边界；ruff 检查通过。
- 全量审计确认 58,440 个 JSONL row 与 58,440 张任务图一一对应，sample id 唯一，所有 bbox/label/图像
  引用合法，生成式 resize/padded/degraded view 均满足 2M 与 factor-32 约束；val 为 0，SFT 目录不存在。

### 后续防线

- 人工 compact schema 是真源，不得把字段精简误判成数据缺失，也不得为了兼容 builder 补造 rich fields。
- raw schema 适配只放在 derived builder 边界；split 每次随 active raw 快照重建并执行 test overlap 检查。
- compact `size` 是 bbox 坐标空间真源。EXIF 只能在尺寸精确证明时条件应用，不能全局猜测方向。

## 2026-08-04：v5.7 grounding prompt 与新版人工标注规范对齐

### 现象

- 新版 `data/标注格式.pdf` 明确了四类 detection 的可见边界、line 完整结构以及 raw `layout` 的底层优先
  语义；tracked v5.0 prompt 只列出标签和 JSON schema，未说明文本排除、类别边界或完整 line bbox。
- 本地 v5.3 prompt 已覆盖连通多分支 line，但仍未说明纯文本、page background、shape/image border 和
  card 内部分割线的排除边界。

### 根因

- 旧 prompt 以最小 Qwen detection schema 为主，人工规范后来扩展了可编辑对象边界；两者没有随 PDF
  更新一起版本化。
- PDF 的 raw 层级顺序与模型 SFT 的 canonical output order 属于不同合同，不能直接把前者抄入 prompt。

### 影响范围

- 只影响后续 v5.7 `grounding_layout` 的 train/eval prompt，不改变四类标签、bbox 输出 schema、structured
  数据或既有 v5.3 实验。
- 若继续使用 v5.0，模型可能把 text-only region、对象边框或 card divider 误判成 detection target，也可能
  拆分连通多叉 line。

### 修复方式

- 新增 `grounding_layout.v5.7.yaml`，五个等价 variant 统一要求四类语义、完整可见 bbox、整图整数
  `0..999` 坐标、背景/纯文本排除、所属边框与 card divider 不重复成 line，以及连通多分支 line 合并。
- 不在 prompt 中要求底层优先。当前 raw 抽样 2,000 文件时，外层 shape 位于内层 shape 之前仅 53.096%，
  不能据此替换已经过 GT 分析的 row-major SFT 顺序。

### 回归测试

- 使用 `shaft.prompting.load_prompt_pool` 解析 v5.7 pool，校验 metadata、main variant、五个非空 variant、
  权重和静态 schema；逐 variant 检查四类、`0..999`、text exclusion 与 connected/disconnected line 合同。

### 后续防线

- PDF 更新后分别审查 raw schema、derived target 和 model prompt，不把三层坐标或排序语义混为一谈。
- prompt variant 只改变表述，不得改变标签、schema、空结果、对象边界或 line connectivity 语义。

## 2026-08-04：grounding SFT 贴边 bbox 退化修复与 v5.7 数据生成

### 现象

- 从 58,440 条 grounding structured 数据生成 SFT 时，`prod_032526` 中贴图像底边的
  `[0,385,832,386]` line bbox 被判定为退化框，40 进程转换在写出前失败。

### 根因

- structured bbox 使用图像边界坐标，合法右/下边界可等于 `width/height`；SFT converter 却先裁到
  `width-1/height-1`，把贴边 1px bbox 的两个边界压到同一坐标。这是 derived coordinate contract
  错误，不是人工标注问题。

### 影响范围

- 影响所有 `x2=width` 或 `y2=height` 的 grounding SFT 派生，尤其是贴右/贴底的细 line。首次失败发生在
  原子写出前，没有留下半成品 train JSONL；structured 与 raw 数据不受影响。

### 修复方式

- `_clip_bbox` 改为在 `[0,width] x [0,height]` 上裁剪，再调用共享 Qwen coordinate codec 映射到
  `0..999`；排序、同 label 同量化 bbox 去重和 minimum-one-bin 合同保持不变。
- 使用 v5.7 prompt override 和 40 进程重新生成 train 58,440 条、val 0 条 SFT；row-level prompt 继续为空，
  runtime prompt pool 是训练提示词真源。

### 回归测试

- 新增右边界和底边界 1px line 用例；`tests/test_build_sft_from_structured.py` 与
  `tests/test_qwen_coordinates.py` 共 11 项通过，ruff 与 diff check 通过。
- 全量逐行复算 58,440 条 SFT：sample id 全部唯一，图片引用全部存在，target 与 structured 重新量化结果
  完全一致，坐标范围为 0..999，问题计数为 0。3,181,407 个 structured instance 量化后得到
  3,181,278 个 target，129 个同 label 同 bbox 合并；652 个空 target 均来自预期空样本。

### 后续防线

- 像素 bbox 与像素索引不是同一合同；任何派生脚本都必须明确右/下边界是否允许等于图像尺寸。
- SFT 全量交付不能只检查行数，必须复算 target 并检查 prompt id、坐标、排序、去重和图片引用。

## 2026-08-04：SFT 严格长度、有序多图与 FP16 输入合同收口

### 现象

- `data.max_length` 过去只限制 target budget；processor 后的 prefix 自身超限时，总序列仍可能超过配置上限。
  直接从 prefix 尾部截断虽然能压到上限，却会破坏 chat generation suffix，并可能删掉视觉占位 token。
- 数据、collator 和推理接口以单个 `image_path` 为主；多图记录可能在边界处丢失顺序、静默退化为单图，或让
  消息中的 image placeholder 数与实际媒体数不一致。字符串形式的 message content 也会被错误地按 iterable
  content block 处理。
- 训练配置只有 BF16 开关，模型加载 dtype 与 AMP 执行精度缺少独立 FP16 合同。真实 2B FP16 单步 canary
  虽正常退出，但 GradScaler 首步溢出，`update_applied_steps=0`，证明只检查退出码会产生假阳性。
- 真实 4B BF16 单步 canary 虽有有限梯度和一次 optimizer attempt，保存后的 LoRA-B 仍全零：短 schedule
  的 warmup 首步学习率为零，`update_applied_steps` 本身也不能证明参数已经变化。

### 根因

- 模板层没有声明“允许被删除的消息正文区间”，collator 和 cost estimator 因而无法在保护模板结构、special
  token、媒体 token 和 assistant generation suffix 的前提下执行统一截断。
- 单图字段在 source、record、dataset、processor adapter、infer request 和 online eval 间形成了隐式真源；
  placeholder 计数也在多层分别实现，字符串/dict/list content 的语义不一致。
- `model.torch_dtype` 曾被当作训练精度的替代信息；同时，FP16 的短跑验收没有区分 optimizer attempt 与
  GradScaler 实际应用的 update，BF16 短跑也没有越过 warmup 零学习率阶段。

### 影响范围

- 影响 padded SFT/DPO 的 processor 输入、长度估算、监督 label，以及本地/vLLM 推理和在线评估的媒体顺序。
  GRPO 仍保持显式单图合同；varlen/packing 多图尚未开放并继续 fail closed。
- 影响 CUDA DDP/FSDP 的 FP16 配置、sequence/resume contract 与训练验收。DeepSpeed preset 仍只支持 BF16；
  Qwen3.5/3.6 MoE 只保留内部 descriptor/tiny 骨架，不属于正式训练能力。

### 修复方式

- 模板 plan 增加精确的 truncatable message-body spans。严格截断按较早正文优先删除，只在这些 span 内取舍，
  保留 chat envelope、generation suffix、special/media token，并为需要监督的样本至少保留一个 target token；
  无法安全容纳时提前报错。runtime 与 cost estimator 复用同一选择算法，未发生截断时走零额外分配快路径。
- 统一以有序 `image_paths` 为内部真源：JSONL 接受 `images`，旧 `image_path/image` 仅兼容单图；padded
  SFT/DPO、本地推理、vLLM 和 online eval 全链按输入顺序传递。共享 message-content helper 统一处理
  string/dict/list 并校验 placeholder 数；缺省 `user_prompt` 固定为 `""`。
- 增加互斥的 `train.fp16`，与 `model.torch_dtype` 独立进入装配和 exact-resume contract；拒绝 CPU FP16、
  BF16+FP16、DeepSpeed FP16，以及 trainable FP16 参数再叠加 GradScaler 的不安全组合。真实训练 gate 额外要求
  `update_applied_steps > 0`，LoRA gate 还检查保存后的 LoRA-B 至少一个非零，避免把全部溢出跳步或零学习率
  attempt 误判为训练成功。

### 回归测试

- focused data/collator/template/infer/online-eval/config/checkpoint/SFT/RLHF 测试、framework、smoke、distributed
  suite 全部通过；ruff、compileall 和 `git diff --check` 通过。真实 Qwen3-VL processor 验证 304-token prefix
  在 `max_length=288` 时精确输出 288 token，同时保留 generation suffix、64 个媒体 token，且估算与运行一致。
- CUDA 0、1 上完成真实两卡 DDP：Qwen3-VL-2B-Instruct 以 FP16 AMP、FP32 加载、LoRA 跑 8 steps，最终
  loss `5.8584`、有限 grad norm `27.0551`、实际更新 2 次、useful tokens/s `499.02`；Qwen3-VL-4B-Instruct
  以 BF16、LoRA 跑 2 steps，loss `6.0951`、grad norm `75.4138`、实际更新 2 次、useful tokens/s
  `268.36`。两者 padding
  fraction 均为 0，双图/单图各一条，PEFT adapter 均通过标准导出校验且 safetensors 可读。全程只设置
  `CUDA_VISIBLE_DEVICES=0,1`，未向 `gpu-holder` 发送信号或修改其进程。

### 后续防线

- 任何长度优化必须以模板声明的结构边界为依据，不能对已渲染多模态 prefix 做普通左/右切片；cost estimator
  与 runtime 不得各维护一套截断语义。
- 新媒体接口以复数、有序字段为真源；单数字段只能是“恰好一张图”的兼容视图。多图 varlen/packing 在完成
  media/segment/position/attention isolation 专项验证前保持拒绝。
- FP16 canary 至少运行到一次实际 optimizer update，并检查有限 loss/grad norm、efficiency summary、adapter
  可读与标准导出；任何短 canary 还必须越过 warmup 并证明权重发生变化，单步退出码不能作为数值稳定性
  结论。FSDP FP16 仍需专项 CUDA canary，DeepSpeed 保持
  BF16-only，MoE 训练继续列入 TODO。

## 2026-08-04：Qwen3.5/3.6 MoE objective、fused LoRA 与分片后端边界收口

### 现象

- Qwen3.5/3.6 MoE descriptor 能被识别，但 routed experts 是 fused 3-D parameters，普通
  `target_modules=all-linear` 不会完整覆盖；router auxiliary loss 也没有进入 Shaft 的 SFT objective。
- 初版 eval 若直接平均每个 batch 的上游 router aux，会随 eval batch size/rank partition 改变；同时内部
  `compute_loss()` 一度在 eval 模式返回 CE + batch-local aux，即使最终 `eval_loss` 再被覆盖，外部
  `predict()`/直接调用仍可能得到错误语义。
- 两卡 ZeRO-3 gate 首先在 Shaft 形状校验看到 `shape=(0,)`，识别 `ds_shape` 后又在 PEFT 0.18.1
  `ParamWrapper` 内部因直接读取 empty shard 的 `param.shape` 崩溃。此前只有“目录存在/export 可加载”检查，
  不能证明 backend-native checkpoint 的 optimizer、RNG 和 shard 真正 exact resume。
- fixed-batch checkpoint resume 的最终 `samples_per_second/steps_per_second` 曾沿用总计划量，夸大本次恢复进程
  实际执行的工作量。

### 根因

- 模型注册只声明了 dense-style PEFT module policy，没有 direct-parameter target、训练 objective 和
  dataset-global eval statistic 的统一模型接口。
- router load-balancing loss 是 expert assignment 与 router probability 的乘积，不是可按样本/token 直接
  加和的 scalar；per-batch scalar 事后加权无法恢复 dataset-global 值。
- Transformers 在 `TrainingArguments` 建立 ZeRO-3 runtime 后于模型构造期分区参数；Shaft 可以用
  `ds_shape` 做 plan，但当前 PEFT parameter wrapper 不支持在该状态下注入。这是上游能力边界，不能靠吞掉
  错误或在 pipeline 猴子补丁修复。
- resume throughput 没有以 Trainer 实际 fetch 的 local samples 和 `initial_global_step` 为计数真源。

### 影响范围

- 影响 Qwen3.5/3.6 MoE SFT 的 full/LoRA 训练目标、eval 可比性、adapter init/resume/export，以及
  FSDP/DeepSpeed 的能力声明。dense Qwen 与普通 next-token CE 不改变。
- `eval_loss` 现在明确只代表 token-normalized CE；router balance 是独立诊断指标，不属于模型能力分数，
  也不是 eval/codec/metric/data 对模型输出的误判。
- `ZeRO-3 + PEFT target_parameters` 当前不可用；ZeRO-3 full 与 FSDP fused-parameter LoRA 可用。tiny gate
  不能推出真实 35B 权重的显存、吞吐或收敛结论。

### 修复方式

- `ModelGroup -> ShaftModelAdapter` 增加模型拥有的 `TrainingObjectivePolicy`：训练返回 batch-local scalar
  auxiliary term；eval 返回 batch-first expert counts、router probability sums、valid routed tokens，Trainer
  经 `gather_for_metrics` 去除尾部副本后再生成 dataset-global router balance。eval 的直接 loss 和
  `eval_loss` 均保持 CE-only。
- Qwen3.5/3.6 MoE PEFT policy 为 routed expert `gate_up_proj/down_proj` 与 router `gate.weight` 声明
  `target_parameters`；resolved 参数与 target modules/modules-to-save 一起绑定 adapter signature。
  parameter-target LoRA 要求 PEFT 0.18.1、dropout=0、非 DoRA；MoE QLoRA和 dense/MoE 预量化 FP8 训练
  fail closed。
- plan 使用 `ds_shape` 识别 ZeRO 参数真实维度；组合校验对 ZeRO-3 + target_parameters 提前给出 FSDP LoRA
  或 ZeRO-3 full 的可操作建议。Qwen3.5/3.6 FSDP activation checkpoint wrapper 也由 model policy 禁止。
- backend release gate 分别检查 FSDP optimizer state 与 DeepSpeed 每 rank model/optimizer shard，并比较
  fresh/resumed 最终权重、scheduler、RNG、Trainer state、adapter/full export 与真实非零更新。
- Trainer 记录实际 fetch 的 local sample 数；fixed resume 的 loss、step/s 和 sample/s 只按本次执行窗口计算。

### 回归测试

- objective/Trainer、PEFT/config/model meta、optimizer、pipeline focused tests 与 W2/GA2 Gloo 全局归约测试通过；
  3 条 eval 数据在 2 rank 下验证 sampler 尾部副本不会污染 router metric。
- CUDA 0/1 tiny upstream Qwen3.5-MoE：FSDP fused-parameter LoRA 与 DeepSpeed ZeRO-3 full 均完成 2-step
  fresh、checkpoint-1→step-2 resume、非零 router/expert 更新和标准 HF/PEFT reload；backend-native 最终状态
  精确等价。真实 Qwen3VL 2B/4B 与最终全套 suite 结果以本轮最终验收记录为准。

### 后续防线

- 新模型的非 CE objective 必须进入 model-owned policy；Trainer 不识别模型字段名，pipeline 不实现模型分支。
- 非线性 auxiliary metric 必须先定义可加和充分统计量，再做 distributed gather/finalize；禁止平均 batch scalar。
- 分片后端“能启动”不能作为 resume 结论；至少比较权重、optimizer、scheduler、每 rank RNG、Trainer state、
  backend shards 和 reload forward。tiny 架构证据必须与真实发布权重容量证据分开标注。
- 对上游不支持的组合应早期、可操作地拒绝；不得复制 PEFT 内部注入器或用路径名猜测兼容性。

## 2026-08-04：SFT 最终 review 暴露的 FP16 checkpoint 与证据矩阵问题

### 现象

- `model.torch_dtype=auto + train.fp16=true` 能通过字符串级配置校验；若 checkpoint 自身声明 FP16，HF 会把
  trainable parameters 实际加载成 `torch.float16`，直到 GradScaler unscale 才失败。
- 真实 Qwen3VL-2B FP16 首轮通常由 GradScaler 跳过 overflow，但 HF 同时把 `grad_norm=NaN` 写入
  `trainer_state.json`；Shaft 的严格 JSON checkpoint commit 因而正确拒绝非有限常量，训练无法保存。
- MoE eval statistic 把 accumulator 放在第一层 router-logit device，却没有把其它层 logits 迁到该设备；
  `device_map` 跨设备放置会发生 device mismatch。
- 初版 release gate 只验证 fresh export，FSDP native model state、DeepSpeed shard 数量/语义和分布式 eval 的
  rank-1 实际贡献证据不足；部分文档还把 tiny/allowlisted 能力写成真实规模或跨精度验收。

### 根因

- 配置字符串只能表达用户请求，不能替代模型装配后的实际参数事实；训练精度防线缺少 post-model 校验。
- FP16 overflow 是 scaler 的正常控制流，但非有限 grad norm 只是诊断值，不应进入要求标准 JSON 和 exact
  resume 的持久化状态。
- dataset-global router metric 的计算设备没有像上游 Qwen 实现一样统一所有 layer logits。
- 测试曾把“文件存在/顶层权重相同/运行时 allowlist”混同为 backend-native resume 或正式模型验收。

### 影响范围

- 影响 SFT FP16 的首次 checkpoint、`torch_dtype=auto` 安全边界和 exact resume；BF16/FP32 objective 不变。
- 影响使用跨设备 model placement 的 Qwen3.5/3.6 MoE eval；当前 DDP 每 rank 单设备训练不受影响。
- 影响支持声明的可信度：Qwen3VL 2B/4B 有真实权重证据，Qwen3.5/3.6 dense/MoE 仍只有 tiny upstream，
  Qwen3VL-32B、真实 27B/35B、varlen FP16 和 FSDP FP16 均不能外推。

### 修复方式

- SFT/RLHF pipeline 在模型与 finetune plan 装配后统一扫描 trainable floating parameters；FP16 AMP 发现实际
  `torch.float16` trainable 参数立即报出示例名称。这样 `torch_dtype=auto` 不能绕过防线，冻结的 FP16 base
  仍可由未来独立验证的 adapter 合同接入。
- SFT 日志在 FP16 下把非有限 `grad_norm` 原位替换为 `grad_norm_overflow=1`，保留 scaler 跳步事实，不伪造
  数值，也不放宽严格 JSON parser；有限 grad norm 仍按原字段记录。
- Qwen3.5 MoE eval 将每层 router logits 显式迁到统一 accumulator device 后再 softmax/top-k。
- release gate 比较 scaler、FSDP DTensor local state/native optimizer、DeepSpeed 每 rank model/optimizer shard
  语义、恢复后的 telemetry 与 fresh/resumed 双侧 export；两 rank eval 使用 batch size 1 并要求每个 rank
  都得到 dataset-global 指标。
- 文档按“正式真实权重 / tiny validated / runtime allowlisted / rejected”拆分，删除 32B、FP16 varlen和真实
  Qwen3.5/3.6 的过度声明。

### 回归测试

- 最终 framework 全量回归、smoke suite、distributed suite、MoE CPU fresh/resume/HF reload、W2/GA2 全局
  loss/eval aux probe 全部通过；Ruff、compileall 与 diff check 通过。
- CUDA 0/1（未操作 `gpu-holder`）：真实 Qwen3VL-2B 完成 FP16 AMP + FP32-load padded LoRA 8-step fresh、
  checkpoint-4→step-8 exact resume、GradScaler state、非零 LoRA 更新和双侧 export/reload；真实 Qwen3VL-4B
  完成 BF16 greedy-varlen LoRA fresh/resume/export/reload。
- 同一最终源码态下，tiny upstream Qwen3.5/3.6 MoE DDP padded LoRA、DDP varlen full、FSDP fused-parameter
  LoRA 与 ZeRO-3 full 的 fresh/resume/backend-native state/双侧 export gates 全部通过。

### 后续防线

- 任何 AMP 安全判断都必须同时校验请求配置和装配后的实际 trainable dtype；不能只相信 artifact config 或
  YAML 字符串。
- overflow 可作为有限离散事件记录，NaN/Infinity 不得进入 checkpoint JSON；loss 等主目标非有限仍应失败，
  不能被日志清洗掩盖。
- 分片 resume gate 必须检查后端真正读取的 state，且测试本身要证明所有 rank 的贡献；文件存在和字节哈希
  只能作为补充。
- 新的模型规模、precision、layout 或 topology 必须分别验收；注册兼容性、tiny gate 和 runtime allowlist
  都不能替代真实 checkpoint/hardware 证据。

## 2026-08-05：训练 `device_map` 晚失败与 distributed eval 证据碰撞

### 现象

- 训练 schema 接受非空 `model.device_map`，Qwen loader 会原样传给 HF；torchrun/FSDP/DeepSpeed 要到
  Accelerate prepare 阶段才拒绝 `device_map=auto`，静态映射还可能让多个 rank 先把完整模型装到 GPU0。
- 两 rank eval probe 虽设置了每卡 batch size 1，但三条样本的统计值为 `6/6/9`；rank 0 首条与 rank 1
  独占样本碰巧同值，错误地重复 rank 0 contribution 仍可能得到正确均值。

### 根因

- 推理装载语义与训练分布式 placement 共用了 `ModelConfig.device_map`，训练入口缺少 pre-model 边界校验。
- 测试只检查最终全局数值，没有保证每个 rank 的原始 contribution 可辨识且局部均值不同于全局均值。

### 影响范围

- 影响所有 Shaft 训练算法的启动成本与显存安全，真实 Qwen3.5/3.6 35B 风险最高；正常由 DDP/FSDP/
  DeepSpeed 管理 placement 的配置不受影响。HF 本地推理继续支持 `device_map`。
- 不改变 MoE eval 聚合实现，但旧 probe 对 rank-1 contribution 的证明不足，不能作为完整 release evidence。

### 修复方式

- `build_hf_training_args()` 在任何模型权重加载前拒绝非空 `model.device_map`，训练 placement 只允许由
  `train.distributed.strategy` 负责；推理配置和 engine 保持原语义。
- eval probe 改用互不相同的 `1/10/23` 样本统计值，收集并 all-gather 每 rank reduction 前的原始值；
  两个 rank 的局部均值都必须不同于 dataset-global 均值，且最终每个 rank 必须得到同一全局结果。

### 回归测试

- 新增的 training-args 负向配置用例覆盖 `device_map=auto` 与显式 dict，两者均在模型装载前失败。
- 两 rank Gloo global weighted loss/eval aux probe 通过，并显式观测 rank-local `[[1,10],[1,23]]` contribution。

### 后续防线

- 任何只适合 inference 的模型装载参数进入共享 schema 时，训练入口都必须在大 artifact 物化前 fail closed。
- distributed metric 测试必须让各 rank contribution 可辨识；“每个 rank 最终值相同”不能单独证明所有 rank
  都参与了 reduction。

## 2026-08-05：本地大模型不可变身份的分布式重复全量读取

### 现象

- checkpointable 本地 HF 模型在 baseline 与 post-load closure 各做一次完整 SHA-256；原实现由每个 rank
  独立执行。8-rank 训练会把身份校验的逻辑读取量放大为 `16 × artifact bytes`，真实 27B/35B 权重在共享
  存储上会显著拉长启动并制造 I/O 峰值。
- 初次拆分优化还存在 branch 风险：是否物化身份由 rank-local checkpointing intent 决定，若 rank 配置漂移，
  一部分进程可能进入专用 collective，另一部分跳过。

### 根因

- immutable identity 的内容证据按进程重复建立，没有利用标准 torchrun 同节点子进程共享 mount namespace
  的事实，也没有 node leader、跨 node manifest consensus 与独立挂载 fallback 的分层协议。
- `save_strategy/resume` 决定 collective schedule，但最初未进入调用前的 rank-consensus fingerprint。

### 影响范围

- 影响所有从本地 HF 目录启动且需要 checkpoint/save/resume 的 SFT/RLHF，模型越大、rank 越多、共享盘越慢
  越明显；`save_strategy=no`、Hub immutable revision 和 built-in smoke model 不承担同样的本地全量读取成本。
- 不改变模型字节身份、plan fingerprint 或 exact-resume 轨迹语义；独立挂载仍必须自行完整 hash。

### 修复方式

- baseline 与 post-load closure 各由每 node 的 `LOCAL_RANK=0` 完整 hash；所有 node leader 对 content
  fingerprint/file manifest 做全局一致性校验。同 node 非 leader 比较完整 stat manifest，无法证明共享同一
  文件身份时自行 fallback hash。
- 所有本地准备、hash、fallback 和最终 stat closure 异常先转为 status，再通过独立的长超时 Gloo group
  收敛；大模型 hash 不再受默认 NCCL collective timeout 约束。
- SFT/RLHF 的 `model-plan-local` 在调用该协调器前同时收敛 model-plan 与 checkpointing intent，禁止 rank
  进入不同 collective 分支。

### 回归测试

- 单机两 rank Gloo probe 覆盖共享 stat（两阶段 hash 计数 `[4,0]`）与 rank-1 独立 stat fallback
  （`[4,4]`）；两种情况下各 rank content fingerprint/file manifest 完全一致。
- SFT/RLHF pipeline 测试显式检查 `model-plan-local` 包含 checkpointing fingerprint，且仅在需要保存/恢复时
  物化 immutable identity；focused model/pipeline、Ruff、compileall 与 diff check 通过。

### 后续防线

- 任何决定 collective schedule 的条件必须在分支前做 all-rank consensus；不能依赖“正常配置应该一样”。
- stat manifest 只用于证明标准同节点 torchrun 进程看到同一文件身份，不能作为内容 cache；每次 launch 的
  baseline 与 post-load closure 都必须保留。
- 当前证据不外推到真实双主机、NCCL default group + Gloo subgroup、多容器 mount namespace或多 GiB 吞吐。
  对外声明真实多机前必须在冻结 SHA 上补对应 canary 和启动 I/O benchmark。

## 2026-08-05：`half` dtype alias 在加载器与 varlen capability gate 间不一致

### 现象

- Qwen loader 接受 `model.torch_dtype=half` 并按 FP16 加载，但 Qwen3VL 与 Qwen3.5/3.6 的 CUDA varlen gate
  只接受 `fp16/float16`，同一有效配置在 padded 与 varlen 间产生无意义差异。

### 根因

- dtype alias 在 loader 与 sequence execution policy 各自维护，varlen allowlist 漏掉了 loader 已公开接受的
  `half`。

### 影响范围

- 仅影响显式使用 `half` alias 的 CUDA varlen SFT；`float16/fp16/bfloat16/bf16` 与 padded 路径不受影响。

### 修复方式

- Qwen3VL 与 Qwen3.5/3.6 varlen policy 统一接受 `half`，contract 仍保存规范化后的小写请求值并进入
  fingerprint。

### 回归测试

- 两个模型族 policy 均以 `half + flash_attention_2 + DDP` 成功建立 varlen contract。

### 后续防线

- loader、precision validator 与 sequence policy 新增 dtype alias 时必须共享同一接受集合，或用跨层契约测试
  锁住等价 alias；不得让布局切换改变 dtype 拼写语义。

## 2026-08-05：SFT exact resume 丢失尚未 flush 的 reporting window

### 现象

- 当 `save_steps` 小于或不整除 `logging_steps` 时，checkpoint 可能落在两个 logging event 中间。模型、
  optimizer、scheduler 与 RNG 恢复后仍能得到相同最终权重，但下一次 `loss` 和 model-owned
  `aux/*` 日志只统计恢复后的半个窗口；`trainer_state.log_history` 与 `on_log` 事件因而和 uninterrupted
  轨迹不一致。
- 原有 fresh/resume gate 常令 `save_steps == logging_steps` 或每步都记录日志，恰好在保存前清空 accumulator，
  因而没有发现该差异。这不是模型能力或 eval/codec/metric/data 的误判，而是 checkpoint reporting state
  不完整。

### 根因

- Transformers 的 reporting loss、`_total_loss_scalar` 与 `_globalstep_last_logged` 是 Trainer 进程内的临时状态，
  标准 checkpoint 不会自动持久化；Shaft 的 MoE auxiliary raw/weighted/count accumulator 同样只存在内存中。
- resume 初始化会重新建立并清空上述窗口。只比较最终权重和 optimizer state，无法证明可观察训练日志也
  exact。
- 初版修复依赖 Transformers 5.10 的 `_init_training_state/self._tr_loss` 私有形态，但 `pyproject.toml` 仍错误
  声明 `transformers>=4.57.6`；4.57 的 loss 是 inner-loop 局部变量，且仓库其它 checkpoint API 也已不兼容。
  HF callback replacement 开关还会创建未 bind 的 reporting callback。
- 初版恢复把累计 total 除以本次 executed steps，导致 resumed root `train_loss` 翻倍；DDP 下用 rank-local
  pending 作为 baseline 还会引入 `P_mean-P_rank0`。GA>1 若到 optimizer boundary 才注入 pending，会改变
  FP32 加法顺序；no-op resume 和 NaN/Inf filter 也会静默丢窗口。

### 影响范围

- 影响所有 SFT exact resume 的区间 loss 日志；使用 model-owned auxiliary objective 的 Qwen3.5/3.6 MoE
  还会影响 `aux/*`。最终参数轨迹、optimizer update 和 eval 指标本身不受影响，但终态 `train_loss` 会误导
  训练监控。
- 新协议要求 checkpoint 含完整 per-rank reporting snapshot；修复前生成的旧 checkpoint 缺少该 state，不能
  按当前协议 exact resume，只能用 `init_from_checkpoint` 载入权重后启动新 schedule。

### 修复方式

- 新增 stateful `ShaftSFTReportingStateCallback`。每次 checkpoint 在 HF 写 `trainer_state.json` 前，所有
  rank 收集本地未 flush 的 reporting loss、`_total_loss_scalar/_globalstep_last_logged` 与 auxiliary accumulator，
  保存 version/step/world/rank 完整的 snapshot。
- resume 严格解析 callback schema、有限数值、rank 顺序、global step、world size与 auxiliary term 集合；
  HF 完成 reporting state 初始化后，在 `on_train_begin` 恢复本 rank 窗口。任一 rank 构造 snapshot 失败时先
  汇总 status，再一致失败，避免其它 rank 卡在 collective。
- auxiliary 日志消费也先汇总每个 rank 的 term-name 集合；名字漂移时所有 rank 在进入逐项 tensor gather 前
  一致失败。
- checkpoint 捕获使用 HF 4.57/5.x 共有的 `_maybe_log_save_evaluate(tr_loss, ...)` 边界，不再覆写
  `_init_training_state`；仓库基础依赖同时收紧为 `transformers>=5.10.1,<6`，与实际 checkpoint/Qwen3.5 API
  对齐。`restore_callback_states_from_checkpoint=True` 明确拒绝，callback list 非 canonical state 也拒绝。
- pending loss 在恢复后的首个 `training_step` 已完成 backward 后注入，保持 GA 微步顺序；nonfinite filter
  复现 HF 对旧窗口的有限替代。snapshot 全 rank 的 `total+pending` 均值形成 resume baseline，最终 rank-local
  remainder 再做全局平均，`train_loss` 只统计本次执行窗口。no-op resume 保留累计/global-step 终值。
- run-root `save_state()` 移除 stale reporting callback；它只保留终态摘要，resume 仍只接受正式 checkpoint。

### 回归测试

- 新增 Trainer 级回归：fresh 4 step、step 2 保存、step 4 才 logging；checkpoint-2→4 恢复后的 loss、
  auxiliary 日志、`log_history/on_log` 与 fresh 完全一致。测试在修复前稳定复现 resumed loss
  `3.14017`、fresh loss `3.39312`，修复后通过。
- tiny upstream Qwen3.5/3.6 MoE 两 rank CPU Gloo gate 改为 `save_steps=1/logging_steps=2`，当前源码下
  fresh/resume、committed checkpoint、HF reload 全部通过；rank pending 明确不同，root resumed
  `train_loss` 与两个 checkpoint 的全局累计差值一致。
- focused 回归覆盖 GA=2 加法顺序、独立 `on_log` capture、callback replacement、rank schema、no-op resume、
  NaN/Inf filter、final execution-window loss 和 direct `training_step`。当前源码还通过 CUDA0/1 的 tiny MoE
  DDP/FSDP/ZeRO-3、真实 Qwen3VL-2B FP16 padded LoRA 与真实 Qwen3VL-4B BF16 greedy-varlen gates；远端
  CI 和真实双主机仍需按最终提交 SHA 独立验收。

### 后续防线

- exact resume 的定义同时覆盖未来 optimizer trajectory 和已公开的可观察 reporting trajectory；测试不能
  永远把 save/log 边界对齐。
- 新增训练期 accumulator 时，必须明确它是 checkpoint state、可重算 state 还是纯 wall-clock telemetry；
  前两者不得只留在 Trainer 内存。
- distributed checkpoint snapshot 必须先收敛局部异常和字段集合，再进入顺序相关 collective；不能让单 rank
  提前抛错造成 peer hang。

## 2026-08-05：ZeRO-3 保存后 `total_flos` 口径漂移

### 现象

- tiny Qwen3.5 MoE 的 DeepSpeed ZeRO-3、GA=2 exact-resume gate 中，连续训练与从 checkpoint-1 恢复到
  checkpoint-2 的模型权重、optimizer shard、scheduler 和每 rank RNG 均完全一致，但 `trainer_state.json`
  的 `total_flos` 分别为 `1364505984` 与 `1063951872`。
- 该差异不是模型能力，也不是 eval/codec/metric/data 误判，而是训练遥测口径随 ZeRO-3 运行时参数视图变化。

### 根因

- Transformers 默认在每个 microbatch 通过 live
  `model.num_parameters(exclude_embeddings=True)` 计算 FLOPs。ZeRO-3 的 checkpoint gather/partition 会改变
  参数的即时 `numel()` 视图，导致同一个连续进程在保存前后使用不同模型参数量；恢复进程没有继承这段临时
  gather 状态，因此无法复现错误的计数轨迹。
- 仅在 Trainer 构造时缓存 HF `num_parameters()` 也不充分：ZeRO-3-aware `from_pretrained` 可能已经把参数构造
  为 `numel()==0` placeholder。第一次门禁只比较 fresh/resume 相等，曾让 `total_flos=0 == 0` 假绿。

### 影响范围

- 影响使用参数分片且中途保存 checkpoint 的 SFT `total_flos` 和由其派生的观测结果；不会改变 forward、
  backward、optimizer update、最终参数或 eval 指标。
- 若把 `total_flos` 作为 fresh/resume 状态的一部分，旧实现会产生假失败；若直接从比较中排除，又会掩盖真实
  遥测漂移。

### 修复方式

- `ShaftSFTTrainer` 在构造时保存 HF-compatible 的 non-embedding logical parameter count 与 main input
  name；普通参数使用 `numel()`，已由 ZeRO 构造为 placeholder 的参数使用 `ds_numel/ds_shape`。后续 FLOPs
  只使用这一稳定参数量和当前输入元素数。
- 不修改 DeepSpeed/PEFT 内部状态，也不放宽 checkpoint state 等价标准。

### 回归测试

- Trainer 单测主动改变构造后的 live logical parameter metadata，并单独构造 `numel()==0` 的
  `ds_numel/ds_shape` placeholder（含 embedding exclusion），确认 FLOPs 始终使用构造期逻辑参数量。
- CUDA0/1 tiny Qwen3.5 MoE sharded release gate 以 GA=2 重新通过 FSDP fused-parameter LoRA 与 ZeRO-3
  full-finetune 的 fresh/resume、backend-native state 和双侧 export 验证；ZeRO-3 gate 额外要求
  `total_flos > 0`，本次 fresh/resume 均为 `149655830528`。全程未操作 `gpu-holder`。
- 冻结源码后，focused SFT/pipeline/finetune/optimizer 回归、framework、smoke、distributed suites，Ruff、
  compileall、lock 与 diff check 全部通过；CUDA0/1 又通过 tiny MoE DDP padded/varlen、真实 Qwen3VL-2B
  FP16 padded LoRA 与真实 Qwen3VL-4B BF16 greedy-varlen 的 fresh/resume/export/reload gates。

### 后续防线

- 与模型静态结构相关的训练遥测必须在 wrapper/sharder 接管参数前建立真源；不能在热路径从 live shard
  tensor 反推全模型结构。
- exact-resume gate 应继续比较 `total_flos` 等确定性累计状态，只排除 wall-clock、吞吐和显存峰值等真正的
  环境遥测。

## 2026-08-05：MoE router auxiliary coefficient 缺少安全的 run-level 覆写接口

### 现象

- Qwen3.5/3.6 MoE SFT 已能从 HF model config 读取 `router_aux_loss_coef`，但训练 YAML 无法为单次实验覆写；
  唯一办法是修改模型 artifact 的 `config.json` 或另存一份 checkpoint，容易污染模型真源，也无法在 Shaft
  exact-resume contract 中清楚表达训练目标差异。
- `algorithm.params` 已进入 resume contract，却没有内置消费者；拼错的 SFT param 会保持在配置中但不改变
  训练。该问题不是模型能力或 eval/codec/metric/data 误判，而是训练 objective 配置与模型 policy 的接口缺口。

### 根因

- `TrainingObjectivePolicy` 只返回运行时 auxiliary term/default coefficient，没有声明稳定、可配置的 term
  names；`SFTAlgorithm.prepare_trainer()` 也明确忽略 `AlgorithmContext`。
- eval raw metric 名 `router_global_balance` 与训练 term 名 `router_aux_loss` 不同，若通用 Trainer 按 metric
  名猜关联，或在 model finalizer 前替换 coefficient，都会破坏跨模型泛化并可能污染 raw eval 指标。

### 影响范围

- 新接口只影响显式配置 `algorithm.params.auxiliary_loss_weights` 的 SFT。默认省略时，Qwen3.5/3.6 MoE
  继续使用 checkpoint 中的 `router_aux_loss_coef`；dense/Qwen3VL 等未声明 auxiliary term 的 profile 会在
  加载数据和权重前拒绝相关 override。
- override 改变训练总 loss、`aux/*_weighted` 与对应 `eval_aux/*_weighted`；raw auxiliary 指标和 CE-only
  `eval_loss` 不变，也不会修改 HF model config。权重设为 0 仍保留 router logits 与 raw 观测开销。

### 修复方式

- 内置 SFT params 严格只接受 `auxiliary_loss_weights`；term name 规范化为小写，值必须是非布尔的有限
  非负数。空 map 归一为省略，未知 params/term fail closed。
- `TrainingObjectivePolicy.auxiliary_loss_names()` 声明稳定 term names；pipeline 在 model plan 解析后、任何
  immutable artifact hash 前用 resolved adapter 校验，并在 pre-model 阶段防御复核；`SFTAlgorithm` 与
  Trainer 也保留边界校验。config-preflight 将 canonical weights 纳入跨 rank fingerprint，runtime 同时拒绝
  未声明或重复 emitted terms。
- `ShaftEvalAuxiliaryStatistic` 和 `ShaftEvalAuxiliaryMetric` 用必填 `coefficient_key` 显式传播关联。model
  finalizer 始终看到默认 coefficient 并先生成 raw metric，Trainer 最后才对 `*_weighted` 应用 effective
  coefficient。
- normalized `algorithm.params` 已由统一 training resume contract 绑定；不新增第二份 objective fingerprint。

### 回归测试

- config 回归覆盖 canonicalization、空/未知/非 mapping、负数、布尔值和非 SFT 使用；pipeline smoke 证明
  override 被传到 Trainer，并证明不支持的模型在 model loader 前失败。
- Trainer 回归同时验证默认与 override 的训练 loss/gradient/logging、raw eval coefficient 不受覆写、weighted
  eval 使用 effective coefficient，以及未知 term 拒绝；Qwen3.5 MoE policy 回归锁定声明名与默认 coefficient。
- focused config/model/trainer/pipeline tests 已通过；CPU-only tiny Qwen3.5/3.6 MoE 端到端门禁也已通过
  fresh train、checkpoint-1 精确恢复到 step 2、router/expert 参数更新、auxiliary loss 日志、full HF 保存与
  reload。该门禁使用官方 Transformers 类随机初始化微型 MoE checkpoint，只证明框架接线与状态语义；本地
  没有真实 Qwen3.5/3.6 MoE 权重，因此不能外推 35B 规模的显存、吞吐、分布式稳定性或长程收敛。

### 后续防线

- 新模型 auxiliary objective 必须同时声明 canonical term name、模型默认 coefficient 和 eval
  `coefficient_key`；禁止在 Trainer 按模型族硬编码或按 metric 名推导。
- 新增内置 `algorithm.params` 必须有严格 schema、真实 consumer、pre-expensive-start validation、resume
  contract 和训练行为测试；不能再保留静默无效字段。
- 任何 coefficient override 都只能改变 weighted contribution，不能提前进入 model-owned raw metric
  finalizer。

## 2026-08-05：v5.7 reconstruction prompt 与新版标注协议不一致

### 现象

- `grounding_layout.v5.7` 已按新版人工标注指南更新，但 shape/line context reconstruction 仍使用 v5.3
  协议：shape 缺少 `card/splits`，line 仍输出 `fill_color` 和旧边框字段，line points 未明确新版曲线和圆角
  取点规则。
- 这不是模型能力问题，也不是 eval/codec/metric 误判，而是 prompt、data target 与新版标注协议不同步。

### 根因

- 旧 pool 同时把真实属性标注值域和合成 `gt_standard` 值域当作同一套 DSL。新版 PDF 与 v9 数据已经区分：
  合成 shape 使用 `uniform|complex` 和 `none|exist`，真实属性子集仍需要透明填充、渐变和 shadow/glow。
- v9 image 虽有显示形态字段，但 `image_type` 为 `N/A`，不能直接并入现有 13 类 image-type 任务。

### 影响范围

- 影响后续 v5.7 shape/line 全量重建、真实 shape 属性弱监督、line 属性恢复和 line points 数据生成。
- 现有 v5.3 SFT 数据保持历史协议，不能仅切换 prompt 路径后继续使用。

### 修复方式

- 新增五个独立 v5.7 pool，保留旧文件：synthetic shape/line 使用 v9 精确字段；真实 shape 属性保持独立
  非几何值域；line 属性改为嵌套 `fill/border`；line points 明确直线、圆角和曲线取点规则。
- 暂不扩展 image-type pool；后续显示形态重建必须独立建任务，禁止把 `N/A` 映射为 `other`。

### 回归测试

- 使用仓库 `load_prompt_pool` 加载并渲染全部新变体，校验动态 bbox、pool id/version、main 变体和 UTF-8/YAML
  合法性；对 prompt 文本执行旧字段与必需新字段检查。

### 后续防线

- prompt 版本升级必须同时核对 PDF、实际源 JSON 值域和将要生成的 target；不能只根据文档示例改字段。
- 新 pool 进入训练配置前必须重建或显式迁移数据，并对 target 做 exact-key/value-domain 校验。

## 2026-08-05：测试集图片后缀与真实编码格式不一致

### 现象

- 在冻结 175 张测试集上准备 PNG/JPEG 压缩鲁棒性实验时，按 `.png` 后缀得到 79 个候选，但 Pillow
  解码后只有 77 个真实 PNG；`pic_891.png` 和 `pic_939.png` 的实际编码格式均为 JPEG。
- 第一版临时抽样只检查后缀，曾抽中 `pic_891.png`。该问题属于 data/eval 输入标准误判，不是模型能力、
  codec 或 metric 问题。

### 根因

- 历史测试资产的文件扩展名没有与内容编码保持一致，而临时实验错误地把路径后缀当成图片格式真源。
- 常规 layout 评测只需要图片可解码，因此此前没有暴露；只有把“原始 PNG”作为实验条件时才会破坏条件定义。

### 影响范围

- 若不修复，所谓 PNG baseline 会混入已经过 JPEG 编码的输入，压缩率、PSNR 和模型性能变化都不可比。
- 异常在正式推理前由输入格式校验发现；最终 20 张实验样本已从 77 张真实 PNG 中重新确定性抽取，正式
  100 次推理结果不受污染。

### 修复方式

- 格式敏感实验同时要求路径后缀为 `.png` 且解码后的 `Image.format == "PNG"`，并把后缀候选数、真实
  格式候选数和排除项写入 experiment manifest。
- 原始测试集不改写；两个异常文件只从本次 PNG 候选池排除。

### 回归测试

- 最终 manifest 记录后缀池 79、真实 PNG 池 77 和两个排除项；20 个 source 派生的 100 个输入逐一检查
  实际格式、原始尺寸、SHA-256 和 4M pixel cap。
- 100/100 推理成功，PNG/JPEG 条件各 20 张，prediction overlay 和 detail crop 各 100 个，HTML 本地链接
  缺失为 0，`validation.json` 状态为 `ready_to_review`。

### 后续防线

- 任何以编码格式、alpha、色彩空间或压缩方式为自变量的实验，都必须以解码元数据而不是扩展名作为格式
  真源，并在推理前 fail closed。
- 数据 split 的常规完整性检查后续应增加“扩展名与解码格式一致率”统计，但是否修复历史文件名需要单独
  迁移，不能在评测脚本中静默改写原始数据。

## 2026-08-05：Q95 聚合下降被误读为 PPT 类别普遍敏感

### 现象

- 20 张 PNG/JPEG 压缩鲁棒性临时测试中，Q95 相对原始 PNG 的 all-label micro-F1 从 `0.792115`
  降到 `0.776256`（`-0.015859`），初看像是 PPT 在轻微 JPEG 压缩下普遍退化。
- 该指标变化是真实复算结果，但“PPT 类别效应”的解释不成立；这属于 eval sampling/aggregation 解释问题，
  不是 codec 或 matcher 计算错误。

### 根因

- 真实 PNG 候选池本身就是 73/77 张 PPT，固定随机 20 张中为 19 张 PPT、1 张 medical，几乎没有非 PPT
  对照，不能从这个样本估计类别差异。
- Q95 总 TP 变化为 `-17`；`ppt_0004` 与 `ppt_0017` 两张合计贡献 `-18`，其余 18 张合计反而为
  `+1`。排除 `ppt_0004` 后 Q95−PNG F1 变为 `+0.003989`，两张都排除后为 `+0.008678`。
- `ppt_0004` 是嵌套标注粒度翻转：shape prediction `12→2`、icon `5→7`，多个 shape+inner icon 被模型
  合并理解为整体 icon；`ppt_0017` 是四个约18×30 px的低对比细白边矩形消失。Q95 仍做4:2:0色度
  降采样与DCT量化，而生成式greedy序列对早期离散token边界不连续，因此高PSNR不保证JSON对象列表稳定。

### 影响范围

- 本轮数据只支持“Q95 在该 20 张样本上观察到小幅下降，且集中于两个敏感 source”，不支持“PPT 普遍
  比其它类别敏感”或“Q95 必然下降约 1.6 F1 点”。
- paired source bootstrap 10,000 次的 Q95 ΔF1 95% 区间为 `[-0.067467,+0.018864]`，跨 0；该区间
  只覆盖 source sampling uncertainty，不覆盖重复 serving 的数值/动态 batching 波动。

### 修复方式

- 增加逐 source F1 delta 排序、TP/FP/FN贡献拆分、leave-one-out/leave-two-out敏感性和类别预测数量对照，
  不再只报告一个聚合点估计。
- 生成独立的 Q95 技术诊断报告；明确把验证事实、理论机制和未验证训练域假设分开。没有修改 prediction、
  GT 或 matcher。

### 回归测试

- 从 `sample_results.json` 独立重算 20 张 PNG/Q95 的 TP/FP/FN、precision/recall/F1，与正式汇总在
  `1e-12` 内一致；逐图ID唯一、条件各20张。
- 诊断数据物化为SQLite后由报告中保存的真实SQL查询重新读取；canonical report artifact validation和
  portable HTML结构验证通过。当前Chromium环境与打包器不兼容，因此仅完成structural-only验证，未声称
  浏览器viewport/source interaction已验证。

### 后续防线

- 需要推断“某类别更敏感”时必须做分层抽样或至少保证各类别有足够对照；按文件格式过滤后的类别分布也要
  在实验前报告。
- 小样本鲁棒性评测必须同时给逐样本分布、离群贡献和paired uncertainty；聚合点估计不能直接解释为普遍效应。
- 后续应增加同一PNG的A/A重复推理，以及JPEG Q95 4:4:4/4:2:0对照，再决定是否加入整图grounding的
  JPEG增强。

## 2026-08-05：V9 `gt_standard` 的 v5.7 reconstruction 重建与 schema 边界

### 现象

- V9 新增 `shape_type=card` 与 `splits[].split_corners`，旧 context builder 只转换外层
  `corners`，会把 split 控制点保留成原图像素坐标，形成混合坐标 target。
- 初版审计错误地把合法的 image `clip_shape=none|regular_hexagon`、oval 的空 `corners`，以及
  shape-style straight line 的 `corner_style` 判为异常。
- 旧 `line_context_points` 默认依赖 `data/archive2` 的 point/full-image manifest；该归档在当前机器已
  不存在，按旧 README 路径重建会在预检阶段失败。

### 根因

- V5.7 prompt 已切换到新 synthetic DSL，但采样、严格校验和几何坐标转换仍部分沿用 v5.3 假设。
- `corner_style` 的约束对象是有折点的 `straight` line，不是 `line_style=path`；image 显示轮廓和值域也
  不能从少量 rectangle 示例推断。
- 历史 real-point 数据源没有随 derived selection 一起保留可重建真源，构建器又没有 synthetic-only
  的显式模式。

### 影响范围

- 影响 V9 shape/line reconstruction 与 point subset，不影响现有 v5.3 数据。Grounding 只消费
  `type+bbox`，使用独立 clipping/drop 规则。
- 全量审计确认 100,000 个 train JSON 均可解析且与 PNG 尺寸一致；reconstruction 过滤 4 个非法 shape
  bbox、992 个非法/越界 line bbox，以及 2,428 条不满足 curved 每段固定四点合同的 line。V9
  `image_type=N/A` 仍不进入 13 类 image-type 训练。

### 修复方式

- 新增 `scripts/tasks/prepare_gt_standard_v5_7.py`，以 40 进程完成图像/JSON 审计，并按 shape type 与
  shape/line 细粒度 strata 做不放回采样；60,000 条以下的 shape 类全部保留。
- Context builder 将 card split geometry 纳入 crop 覆盖范围并统一量化到 contextual-crop `0..999`；line
  prompt/校验同步允许 straight shape-style line 使用 `corner_style`。
- 增加 `--line-point-synthetic-only`，允许 V9 多分支 point subset 独立构建；不把缺少原始真源的旧 derived
  selection 提升为 truth。
- 新产物写入 `data/v5_7`：grounding sync 100,000、shape 300,000、line 300,000、synthetic line-points
  15,000；旧 v5.3 目录未覆盖。

### 回归测试

- `tests/test_prepare_gt_standard_v5_7.py` 覆盖完整审计、card/split schema 和确定性 selection；context
  builder 测试覆盖 card split 坐标与 synthetic-only point 模式。
- Shape/line 各 1,000 条 canary 通过 schema、坐标、媒体引用和可视化检查。
- 全量 615,000 条 reconstruction 逐行校验无重复 ID、缺图、legacy line 字段或非法 `0..999` 坐标；
  100,000 条 grounding SFT 逐行通过 `{bbox_2d,label}` 合同。

### 后续防线

- Prompt 版本升级必须同时审计 source value domain、selection validator 和所有几何转换字段；只更新 YAML
  不代表旧 target 已兼容。
- 新增嵌套几何字段时，必须同时进入 geometry coverage 与 coordinate codec 测试。
- 可选归档源缺失时应在预检阶段失败或使用语义明确的 source-only 模式；禁止从 derived target 反写或推断
  raw truth。

## 2026-08-05：删除未采用的 `line_attribute_recovery` 任务

### 现象

- v5.3 line 预标注脚本残留一个 `line_attribute_recovery` 兜底请求，v5.7 prompt 清单又延续了该名称，容易被
  误认为正式训练子任务。
- 实际 v5.3 训练未使用该任务，line 非几何属性始终属于完整 `line_context_reconstruction` 合同。

### 根因

- 一次性预标注流程中的失败恢复分支被错误提升为独立 prompt/task 概念，没有与真实训练数据源和训练配置
  对照确认。

### 影响范围

- 未产生训练数据污染；影响仅限两份未采用的 prompt 和旧预标注脚本中的额外 API 兜底路径。

### 修复方式

- 删除 v5.3/v5.7 `line_attribute_recovery` prompt，以及预标注 CLI、请求、融合、审计计数中的对应逻辑。
- 保留从完整 reconstruction 结果提取样式属性的逻辑；几何子集仍由 `line_context_points` 负责。

### 回归测试

- `tests/test_prelabel_line_reconstruction.py` 21 项通过；ruff 与残留引用检查通过。

### 后续防线

- 只有存在独立 target contract、派生数据源和训练配置消费方时，才把辅助恢复调用登记为训练任务。
- 后续 line 任务清单只包含完整 `line_context_reconstruction` 和几何子集
  `line_context_points`，不得重新引入属性恢复任务。

## 2026-08-05：`line_context_points` 接入全部 active raw 真实路径

### 现象

- v5.7 初版 `line_context_points` 只有 15,000 条 v9 合成多分支数据，缺少当前人工 compact raw 中已经存在
  的真实路径监督。
- 真实 source points 去除 160 个相邻重复坐标后仍有 17 个点对在 Qwen `0..999` 量化时落到同一坐标。

### 根因

- 旧 builder 的真实 points 入口只支持已丢失的 archive manifest，无法直接从当前
  `size + layout[].parameters.points` 真源重建。
- 坐标转换只在源像素空间做去重，没有在离散量化边界再次检查相邻点。

### 影响范围

- Raw JSON 未改写。122,218 条非空真实 line points 此前未进入 v5.7 point subset；222,792 条空 points
  不具备几何监督，继续排除。
- 量化重复不导致 segment 坍缩，但会产生无意义的零长度相邻边。

### 修复方式

- 新增 `prepare_real_line_context_points.py`，从 active train split 不放回选择全部非空真实 line：
  112,350 条单分支、9,868 条多分支。
- Context builder 新增 `real_point` source adapter，从 compact raw 回查 bbox、原图和有序 points；真实 crop
  保持 clean，并与现有 15,000 条 `synthetic_realism_v1` 合成多分支数据合并。
- 派生阶段移除 160 个源相邻重复点和 17 个量化后相邻重复点，不删除对应实例；最终仍为 137,218 条。

### 回归测试

- Context builder focused 测试覆盖真实 selection、空 points 排除、真实/合成融合、source/quantization 两级
  去重和严格 `is_single + points` target。
- 全量逐行检查确认 137,218 个唯一 ID、137,218 张对应 crop，无缺图、多余字段、越界坐标、坍缩 segment
  或相邻重复点；validation 文件为 0 行。

### 后续防线

- Point subset 必须从 active raw truth 重建，selection 只保存 source identity，不得把 derived target 提升为
  真源。
- 任意连续坐标转离散坐标的路径任务都要在量化后再次去重并拒绝坍缩 segment。

## 2026-08-05：v5.7 暂停 `shape_context_attributes` 任务

### 现象

- 历史 `shape_context_attributes` 弱标注仍使用 v5.3 合同，而准备中的 v5.7 prompt 已切换到新版
  `uniform/card` 字段，prompt、校验器和现有 target 无法形成一致合同。

### 根因

- 真实 shape 属性弱标注是独立 API 辅助任务，其版本演进没有与新版完整 reconstruction 真值同步。

### 影响范围

- v5.7 不训练 `shape_context_attributes`；现有 v5.3 prompt、弱标注 sidecar 和 22,058 条派生数据只作为
  历史资产保留，不改写、不删除。

### 修复方式

- 删除未启用的 `shape_context_attributes.v5.7.yaml`，并从 v5.7 prompt 清单中移除该任务。
- 后续 v5.7 数据配置不得登记 `shape_context_attributes`，其余 grounding、shape/line reconstruction 和
  line points 任务不受影响。

### 回归测试

- 全仓引用检查确认不存在 v5.7 shape-attribute prompt 或训练配置消费方；v5.3 历史配置仍引用原 v5.3
  prompt。

### 后续防线

- 只有在新版真实属性合同、校验器和重建数据同时完成后，才能在后续版本重新启用该辅助任务。

## 2026-08-05：完成 Banana v5.7 训练配置

### 现象

- v5.7 六类派生数据已生成，但尚无独立 catalog 和训练 YAML；继续复用 v5.3 配置会错误加入已取消的
  background/shape attributes，并遗漏 V9 合成 grounding。

### 根因

- 数据准备、prompt 升级与训练 mix 分阶段完成，最终任务边界和超参数尚未固化为一个可加载配置。

### 影响范围

- 只新增 v5.7 本地运行资产，不修改 v5.3 配置、历史数据或训练内核。

### 修复方式

- 新增 `configs/data/banana_v5_7.yaml`，按 `6:2:4:4:3:1` 登记 real grounding、V9 synthetic
  grounding、shape/line reconstruction、line points 和 reviewed image type。
- 新增 `configs/train/banana_sft_4b_v5_7.yaml`：fresh full SFT、20,000 steps、8 卡 token-budget local
  batch `1..2`、GA=4、语言模型学习率 `8e-6`、视觉/aligner `3e-6/8e-6`、warmup `0.1`、weight decay
  `0.01`、`0.5M..2M` 像素预算。
- Shape/line 相关任务绑定 v5.7 prompt；13 类 image-type 合同未变化，显式保留 reviewed v5.3 prompt。

### 回归测试

- 严格 config loader 成功展开六个 catalog source；所有 train/val 路径存在，train 行数分别为 58,440、
  100,000、300,000、300,000、137,218、21,184，所有 val 均为空。
- 六个 source 的首条 target 可解析，六个 prompt pool 均可编译且包含 `main` variant；权重和为 20，解析
  概率为 30%/10%/20%/20%/15%/5%。

### 后续防线

- 启动训练前必须从 v5.7 YAML 做一次 rank-0 canary，核对日志中的 source weights、2M pixel budget、
  prompt/input fingerprint 和 optimizer group learning rates；不得通过修改 v5.3 配置启动本轮训练。

## 2026-08-05：Qwen3VL 30B-A3B MoE SFT 真实权重门禁

### 现象

- 本地完整 `Qwen3-VL-30B-A3B-Instruct` 是 HF `qwen3_vl_moe`，但 `qwen3vl` 只登记 dense
  `qwen3_vl`，因此在读权重前被 descriptor preflight 拒绝，不能开始 SFT。
- 既有 MoE objective 只确认 router trace 非空，不确认是否覆盖所有稀疏层；真实 48 层模型若漏 hook，
  auxiliary loss 可能静默基于不完整 trace。
- artifact 的 `text_config.use_cache=true`，而训练此前只在 gradient checkpointing 打开时关闭 cache；
  关闭 checkpointing 的训练会无意义构造 generation cache。
- 第一版两步门禁曾错误显示 fresh/resume 一致：默认 warmup 令 checkpoint-1 的学习率为零，掩盖了模型参数
  恢复缺陷。将 warmup 设为零后，fresh step 2 的 loss/grad norm 为 `4.28125/19.6372`，旧 resume 则为
  `4.765625/18.2464`，确认不是数值噪声。
- 旧 checkpoint 中 `pytorch_model_fsdp.bin` 的 904 个 adapter 张量全部是沿 dim 0 的 rank-local DTensor；
  rank 0 文件只有完整 adapter 的 50%，遗漏半片已有非零 LoRA-B 更新，却被当作完整模型状态恢复。

### 根因

- Qwen3VL 的 variant registry、sharding、PEFT 和 objective policy 只为 dense profile 建模，没有把
  `qwen3_vl_moe` 作为同一公开 `qwen3vl` 模型族中的独立 model group。
- router objective 的最初 gate 来自微型模型，只验证单项 shape，没有把 topology 完整性提升为模型合同。
- `use_cache` 被误当成 gradient-checkpointing 附属开关，而不是所有训练态的统一禁用项。
- Transformers 5.10.1 的 FSDP checkpoint kwargs 对 PEFT 使用 `adapter_only=true`；Accelerate 1.13 的
  FSDP2 保存/加载路径因此绕过 full-state 选项，产生 rank-local adapter DTensor。Shaft SFT resume 此前在
  FSDP wrap 后继续消费该文件，且 backend-native checkpoint identity/stat guard 没有绑定标准
  `adapter_config.json` 与 `adapter_model.safetensors`，所以不完整状态既未被拒绝也未被完整 adapter 覆盖。

### 影响范围

- 影响 Qwen3VL MoE SFT 的装配、router auxiliary 正确性、FSDP/PEFT 路由与训练内存；dense Qwen3VL
  forward、数据和原始模型权重均未修改。
- 不完整 native adapter 的问题影响 Shaft 的 SFT FSDP+PEFT exact resume 语义，不只影响 MoE；DDP adapter
  恢复与 full-finetune FSDP 模型状态不使用这条逻辑。DPO/GRPO 尚未按同一策略验收，不能由本次结论外推。
- 两张 80GB 卡不具备 30B 全参数 AdamW 的容量，本轮验收范围明确为真实 BF16 FSDP LoRA，不外推 full SFT。

### 修复方式

- 在 `qwen3vl` 内新增 `qwen3_vl_moe` group：padded-only sequence policy、
  `Qwen3VLMoeTextDecoderLayer/Qwen3VLMoeVisionBlock` FSDP wrap、fused expert/router
  `target_parameters` 和通用 `QwenVLMoeTrainingObjectivePolicy`。
- 新增 `model.experts_implementation`，真实 gate 显式使用 `grouped_mm` 并绑定 exact-resume contract；
  dense profile、其它模型族和 Qwen3VL MoE ZeRO-3 均 fail closed。
- 按 `num_hidden_layers/decoder_sparse_step/mlp_only_layers` 推导 router 层数，同时校验每层 expert 维度；
  所有训练微调路径统一关闭 `use_cache`，推理/在线生成仍在各自作用域显式恢复。
- 抽出公共 PEFT artifact resolver：固定选择标准 safetensors（否则 bin），绑定 adapter config/weights 的
  size 与 SHA-256，并把同一 artifact 放入 backend-native generation identity 和启动 stat guard。
- `ShaftSFTTrainer` 在 FSDP wrap 前向每个 rank 完整预载标准 adapter，严格校验 key、shape 与加载后 value；
  进入 HF resume 时只跳过不完整 native 模型文件，optimizer、scheduler、scaler、RNG 与 Trainer state 仍按
  backend-native 路径恢复。FSDP+PEFT 强制 `state_dict_type=full_state_dict`，并对
  `load_best_model_at_end=true` fail closed。
- adapter init/export 与 FSDP resume 复用同一个 model 层 exact-load 函数，不再各自维护 key/shape/load
  逻辑；FSDP preload 还禁止 `model_init` 在上游重新替换模型，并把布尔 `resume_from_checkpoint=true`
  一次解析为固定 canonical checkpoint 后再交给 Trainer。
- 两步门禁显式使用 `warmup_ratio=0`，要求 checkpoint-1 已有非零 LoRA-B 更新且 checkpoint-1/2 artifact
  必须不同，避免“零学习率第一步”再次制造假阳性。

### 回归测试

- focused config/model/objective/PEFT/sequence tests 与两进程 CPU tiny Qwen3VL MoE fresh/resume/HF reload 通过；
  新增 focused Trainer 编排测试，锁定 preload 先于上游 train、只跳过同一 checkpoint 的 native 模型加载，
  且 optimizer/scheduler 恢复入口仍会执行。
- CUDA 0/1 真实 30B-A3B gate 通过：两卡 FSDP LoRA 两步 fresh，并从 checkpoint-1 在独立 output 目录
  恢复至 step 2。fresh checkpoint-1 adapter SHA256 为
  `912f771ad16464263bf640380c1ca543526c4cf445482d6a489457190b93d706`；fresh checkpoint-2、fresh best、
  resumed checkpoint-2 与 resumed best 均为
  `6c6a5fcd733e406c0ad7afca6ea7fffa91f9814b07a7b140b7c70d145f39ba7e`。
- 两步 loss 为 `5.40625/4.28125`，grad norm 为 `21.4922/19.6372`；step 1 raw router aux 为 `8.0625`，
  覆写权重 `0.002` 后 weighted 值为 `0.016125`。resume 的 step 2 loss、grad norm 与学习率逐项一致；
  两步 optimizer update 均应用。fresh 每卡峰值约 `38.72 GiB allocated / 40.75 GiB reserved`。
- 逻辑总参数 `31,397,204,976`，trainable/optimizer/adapter 参数均为 `326,450,944`，904 个 adapter tensor
  全部覆盖；门禁从真实 config 推导并验证 48 个 language/router/expert 层、27 个 vision block，且所有
  LoRA-B 非零；checkpoint-1→2 的 904 个 A/B tensor 也全部发生变化。
- 标准 HF+PEFT 重载真实基座后完成有限 BF16 forward，并验证 router trace 为 48 层、每层 128 experts；
  router、fused gate-up/down、language attention 与 vision LoRA 均有非零更新。

### 后续防线

- MoE release gate 不能只看退出码或 loss：必须检查完整 router topology、raw/weighted coefficient、
  expert/router/普通模块更新、标准 adapter exact resume、backend-native optimizer state 和标准 PEFT reload。
- FSDP+PEFT 的逻辑模型真源是完整标准 adapter；不得恢复或比较 rank-local `pytorch_model_fsdp.bin`。
  checkpoint identity/stat guard 必须覆盖选中的 adapter config/weights，且 gate 的第一保存点必须已经发生
  非零参数更新。
- 不以短门禁宣称长程收敛或 full-parameter 容量；Qwen3VL MoE 的 DeepSpeed ZeRO-3 必须先实现并验收
  routed-expert leaf-module contract，不能照搬 dense 或 Qwen3.5/3.6 profile。
- `gpu-holder` 只负责自动让卡；门禁全程不得停止、重启、发信号或修改其配置。

## 2026-08-05：v5.7 synthetic grounding 改为 100% 加噪

### 现象

- `grounding_layout_sync` 的 100,000 条 V9 合成 detection 仍直接引用 clean source PNG，与本轮要求的全量
  合成域扰动不一致。

### 根因

- 历史 replay 规则刻意保留 clean synthetic detection，而 reconstruction 已采用
  `synthetic_realism_v1`；两条生成链没有共享像素增强真源。

### 影响范围

- Grounding bbox、实例、canonical target 和 train/val split 不变；只改变 synthetic grounding 的媒体像素
  和媒体快照。真实 grounding 与正式 eval 不受影响。

### 修复方式

- 抽取 `shaft.data.synthetic_realism` 作为 reconstruction 和 synthetic grounding 共用实现。
- V9 每个 train id 物化一张保持原尺寸的 task-local PNG；每条强制包含 Gaussian noise，并可叠加
  resample round-trip、Gaussian blur 或 JPEG compression，不再保留 clean row。
- 40 进程在 staging 重建 structured/SFT 后发布；`media_snapshot_id` 从
  `banana-v5.7-media-v1` 升为 `banana-v5.7-media-v2`。

### 回归测试

- Builder/context focused tests 通过，ruff 通过；100-row canary 的像素变化、尺寸、媒体引用和 target
  100/100 一致。
- 全量产物为 100,000 个唯一 ID 和 100,000 张 PNG；Gaussian noise 100,000 次，stack depth
  `1/2/3 = 4,305/41,241/54,454`，validation 为 0。
- 新旧 100,000 条 target 逐条完全一致；全量 PNG header/尺寸 100,000/100,000 通过，分层抽取 1,000 张
  与 source 比较均发生像素变化。

### 后续防线

- `grounding_layout_sync` 的 build summary 必须满足
  `profile.synthetic_realism_v1 == gaussian_noise == rows`；任一 clean/source-direct row 都应拒绝发布。
- 修改派生媒体后必须升级 `media_snapshot_id`，避免 checkpoint exact-resume 把不同输入像素视为同一快照。

## 2026-08-06：Qwen3VL 多变体注册破坏远程 serving alias 推理

### 现象

- Qwen3VL 注册 dense/MoE 两种 HF architecture 后，vLLM OpenAI engine 使用 `banana`、
  `arrow_mixed_4b` 等合法 served-model alias 时，在发出 HTTP 请求前被完整模型 adapter resolver 拒绝。
- 全量测试同时暴露两条旧断言：FSDP 训练仍期待 `use_cache=true`，Qwen3VL freeze 测试仍使用旧版
  `model.layers.*` 参数路径。

### 根因

- 远程推理只需要模型族拥有的 template 与 inference policy，却错误复用了训练、本地加载和分片执行所需的
  完整 adapter 解析；served alias 本身不能提供 HF `config.json`，也不应被塞入本地模型 catalog。
- 旧测试没有随统一训练态 `use_cache=false` 合同和 Transformers 5.10.1 的
  `model.language_model.layers.*` 参数层级同步更新。

### 影响范围

- 影响 Qwen3VL vLLM OpenAI 远程推理入口；本地 HF 推理、训练、导出及 dense/MoE 完整变体判定不应放宽。
- cache 与 freeze 两项是测试口径滞后，生产实现和真实 4B/30B 参数索引均支持当前实现。

### 修复方式

- 新增最小 `ShaftInferenceContract`，只承载 `template_type` 与 model-owned inference policy。
- `ModelMeta.resolve_inference_contract()` 对 catalog/descriptor 命中的模型使用对应变体；未知远程 alias
  仅在所有注册变体的有效 inference contract 完全一致时允许，否则继续 fail closed。
- vLLM OpenAI adapter 不再持有包含 sequence、sharding、PEFT 和 objective 状态的完整
  `ShaftModelAdapter`；完整 `resolve_adapter()` 保持严格，不默认 dense。
- cache/freeze 测试改为统一训练 cache 合同和真实 Qwen3VL 参数路径，不改变生产逻辑。

### 回归测试

- vLLM OpenAI 请求、顺序多图、像素预算、deadline/cancellation 等 8 条原失败用例恢复通过。
- 新增远程 alias contract 边界测试：Qwen3VL dense/MoE 共享推理合同时可解析；模板或 policy 不同的
  多变体 family 必须拒绝。
- infer、model builder、freeze、model meta focused suite 全部通过。

### 后续防线

- 远程 serving alias 不能承担本地 artifact identity；远程推理只解析实际消费的最小 contract。
- 只有所有候选变体共享同一推理合同才能省略 variant；训练、本地加载、分片与导出始终要求完整 descriptor。
- 模型族测试中的参数名必须来自当前上游真实权重索引或模型实现，不能长期保留能命中前缀但不存在的伪路径。

## 2026-08-06：v5.7 reconstruction real_v1 评测与 review renderer 合同错位

### 现象

- 为新下载的 `data/real_v1` 准备 v5.7 shape/line reconstruction 输入时，临时评测器把“不适用”的
  `body_bbox` 写成空数组；context target 转换器将“字段存在但为空”正确判为非法 bbox，canary 在推理前失败。
- 第一版 review 复用了 v5.2 renderer。v5.7 合法 `card` 输出包含 outer corners、fill 数组和 splits，旧
  renderer 不认识 `card`，会把合法预测显示成透明 render；line 的 nested fill/border 与 shape 的
  `effect=exist` 也需要显式适配。若直接交付，会把 renderer 缺口误判成模型未输出几何。

### 根因

- 临时评测器用空值占位表达“GT 几何不可比较”，混淆了“字段缺失”和“字段存在但非法”两种语义。
- review renderer 没有按 prompt/validator 的 v5.7 输出合同版本化，而是假设旧 shape/line schema 可以直接
  复用；renderer 与当前 codec/validator 的允许类型、嵌套字段和隐式几何规则没有共同门禁。

### 影响范围

- 只影响本轮临时 input preparation 与第一版 review 解释，不影响 `data/real_v1`、训练数据、checkpoint、
  模型原始输出或正式 Shaft 主链。输入 canary 失败时尚未启动推理。
- Shape 共 427 条预测为 `card`；旧页面会让其中可解析的合法 card 缺少 render，因此人工 review 会系统性
  低估模型输出完整性。定量 contract/attribute/geometry 指标使用独立 validator 和保存的 prediction，不受
  旧 renderer 影响。

### 修复方式

- 对不可比较 geometry 删除不适用字段，不再写空 `body_bbox/body_corners/corners/tail`；只在 real_v1 有
  明确点/角标注时进入 geometry support。
- review renderer 增加 v5.7 card outer geometry、分区 fill、split 与 border；line nested fill/border 显式
  转成 renderer 输入。`effect=exist` 因没有 shadow/glow 子型只显示 warning，不编造视觉效果。
- Polygon/line 缺失预测几何时保持透明，不使用 proposal 或 GT fallback；仅 oval 按 v5.7 DSL 使用 proposal
  bbox 作为隐式外形。页面第一张图明确标注 proposal bbox 不是 GT，并只展示 prediction fields。

### 回归测试

- 全量 input canary 通过：4,545 个唯一实例，shape/line 为 2,494/2,051；显式 geometry support 为
  1,975/1,666，所有输入图、crop 尺寸和 proposal bbox 边界通过。
- Card renderer canary 验证合法 outer/split 生成非空 RGBA；缺 corners polygon 验证保持全透明；line/shape
  普通 canary 均生成非空 render。
- 60 进程重建 review：source/overlay/render 各 4,545，全部 render 为 RGBA，78 个 HTML 本地缺失链接 0，
  review record 不含 GT；Chrome 人工检查确认 proposal、overlay、棋盘格 render 和 prediction fields 对齐。
- 全量推理 4,545/4,545 完成、request error 0；结束后只关闭本轮自启 vLLM，未操作 `gpu-holder`。

### 后续防线

- Reconstruction review renderer 必须与 prompt/validator schema 同版本；新增 shape type、nested field 或隐式
  几何规则时，先用当前合同的合法/非法 canary 验证 renderer，再允许生成全量页面。
- “无可比 GT”“模型漏字段”“字段非法”“renderer 不支持”必须四分，不得都降级成透明图或一个 invalid 率。
- 评测放行不能只看 JSON decode 或 attribute micro；至少同时报告 strict contract、exact comparable
  attributes、type+geometry、structure count 与 geometry support。

## 2026-08-06：Baidu02 flash-attn wheel 与系统 glibc 不兼容

### 现象

- Baidu02 启动 Qwen3VL SFT 时，8 个 rank 都在 `from_pretrained()` 初始化
  `flash_attention_2` 阶段失败；外层模型 loader 统一报“verify model path and transformers version”。
- 最底层异常是 `flash_attn_2_cuda.so` 要求 `GLIBC_2.32`，而 Baidu02 为 Ubuntu glibc 2.31。
- flash-attn 修复并全量 uv 重装后，真实训练可完成模型、数据、optimizer 和 batch plan 初始化，但 8 个 rank
  在第一个 forward 同时报 `libcudnn_graph.so.9: undefined symbol: cudnnGetLibConfig` 并 SIGABRT。

### 根因

- `.venv` 中预装的 flash-attn 2.8.3 是其它构建环境产出的非 manylinux wheel；其 ELF 动态符号依赖
  `GLIBC_2.32`，超过目标主机提供的 glibc 版本。模型目录和 Transformers 本身均无异常。
- 全量 uv 重装时还暴露第二层复现问题：`extra-build-dependencies` 只给 flash-attn 声明了无版本约束的
  `torch`，且没有覆盖 causal-conv1d。隔离构建环境因此解析到 Torch 2.13/CUDA 13.0，而项目运行时锁定
  Torch 2.10/CUDA 12.8；本机 CUDA toolkit 12.3 随后触发明确的版本不匹配错误。
- Baidu02 系统安装 cuDNN 9.0.0，uv/PyTorch wheel 自带 cuDNN 9.10.2；`/root/.bashrc` 又把
  `/usr/lib/x86_64-linux-gnu` 显式加入 `LD_LIBRARY_PATH`，使系统 cuDNN 抢在 wheel RPATH 前被加载。
  先前 flash-attn、NCCL 和 CPU 模型加载 smoke 都没有执行 cuDNN operator，因此没有覆盖这条动态库路径。

### 影响范围

- 仅影响 Baidu02 上选择 `flash_attention_2` 的模型加载与训练/推理启动；数据、模型权重、训练配置和
  主机上的 `gpu-holder` 均未修改。失败发生在权重初始化阶段，没有产生 checkpoint。

### 修复方式

- 保持 flash-attn 版本 2.8.3，在 Baidu02 使用本机 CUDA/GCC 和 glibc 从源码强制重编译，替换不兼容的
  `.so`；不把训练 attention backend 降级为 SDPA。
- 完整移走旧 `.venv`，使用 Python 3.11 和当前 `uv.lock` 执行 `uv sync --all-extras --locked`；当前环境
  包含 train、GPU kernels、distributed、RLHF、serve 与 dev 全部 extras。
- `pyproject.toml` 为 flash-attn 和 causal-conv1d 都增加 runtime-matched Torch build dependency，并补齐
  ninja/packaging 等构建依赖；通过 build variables 强制源码构建，避免跨 glibc binary 回流。Baidu02 的
  `pyproject.toml` 与 `uv.lock` 已同步到当前主机真源。
- 不卸载或升级系统 cuDNN；仅从 Baidu02 `/root/.bashrc` 的 `LD_LIBRARY_PATH` 中移除显式
  `/usr/lib/x86_64-linux-gnu`。该目录本来就在系统默认搜索路径，移除显式优先级后，PyTorch 可按 wheel
  RPATH 使用 uv 环境内的 cuDNN 9.10.2，同时保留 CUDA、driver/NVML 等原有路径。
- 清理失败启动仅留下的 `outputs/qwen3vl-sft/4b/banana-v5.7_trial/shaft_progress.json` 和空目录，
  确保 trial 可从干净输出目录重新启动。

### 回归测试

- `uv lock --check` 通过，`uv sync --all-extras --locked --dry-run` 为 `Would make no changes`；
  `uv pip check` 检查 208 个包且无冲突。关键版本为 Torch 2.10.0、Transformers 5.10.1、TRL 1.9.2、
  vLLM 0.19.1、flash-attn 2.8.3、causal-conv1d 1.6.2.post1。
- 新扩展 ELF 的最高 glibc 依赖由 `GLIBC_2.32` 降为 `GLIBC_2.14`；
  `import flash_attn_2_cuda/causal_conv1d_cuda` 成功。
- 在 A800 上执行 bf16 flash-attn 与 causal-conv1d CUDA kernel，输出 shape 正确且全部为有限值；8-rank
  NCCL all-reduce 和每卡 flash-attn kernel 全部通过。
- 使用净化后的库路径执行 8-rank cuDNN Conv2d + NCCL smoke，8/8 rank 均解析 cuDNN `91002`、输出有限；
  从全新交互 bash 读取修改后的 `.bashrc` 后，单卡 bf16 Conv2d 同样通过且实际加载 wheel 内 cuDNN。
- `Qwen3VLForConditionalGeneration` 从本地 4B 权重以 bf16 + `flash_attention_2` 完整加载成功；
  trial 配置仍解析为 14,000 steps、default/ViT/aligner LR `2e-5/3e-6/2e-5`。验证结束后训练进程为 0。
- 验收后删除 12 GB 旧环境备份；失败训练的 8 个 core dump 已确认不存在。trial 输出目录保持不存在，
  `gpu-holder` 进程数和运行状态未被修改。
- 首 batch cuDNN 失败产生的 8 个新 core dump（约 32 GB）、无 checkpoint 的 trial metadata/progress 和
  对应离线 W&B 失败 run 已精确清理；trial 输出目录再次恢复为不存在。
- `.bashrc` 修复落盘后，实际训练终端运行在旧 tmux server 内的 zsh：tmux 的 global environment 仍缓存
  含 `/usr/lib/x86_64-linux-gnu` 的旧值，而 `.zshrc` 也不会读取 `.bashrc`。19:54 和 20:00 的两次重试因此
  继续复现相同 SIGABRT。最终同时在 `.zshrc` 固定 wheel cuDNN 优先路径、净化 tmux global environment；
  新 tmux pane 与新 zsh 均实际解析 cuDNN `91002`，bf16 Conv2d 正常。两次 0-step 重试的 metadata、离线
  W&B run、输出目录和共 16 个 core dump 均已精确清理。

### 后续防线

- 跨不同 glibc 基线复制 `.venv` 或 binary wheel 前，必须先检查扩展 ELF 的 `GLIBC_*` 最大依赖，并在
  目标机做 import + 最小 CUDA kernel smoke。
- 依赖 PyTorch ABI 的 uv build dependency 必须使用 `match-runtime = true`，不能写无约束的裸 `torch`；
  多机部署前必须同时核对 `pyproject.toml` 与 `uv.lock` 指纹，不能只对齐代码 commit。
- GPU 环境验收不能只测 `torch.cuda.is_available()`、flash-attn 或 NCCL；必须至少执行一个真实 cuDNN
  operator，并记录 `torch.backends.cudnn.version()` 与实际映射的 `.so` 路径。系统库目录不得显式置于
  PyTorch wheel 的 CUDA runtime libraries 之前。
- 修改 shell 启动文件中的动态库路径后，必须核对实际使用的 shell 与 tmux server environment；仅修改
  `.bashrc` 不会覆盖已有 tmux 或 zsh。Baidu02 需要同步维护 `.zshrc` 与 `tmux set-environment -g`，重启前
  从新 tmux pane 打印 `LD_LIBRARY_PATH` 并运行最小 cuDNN operator。
- 训练 startup 的模型 loader 外层错误不能替代最底层动态链接异常；排障时必须保留并优先查看 chained
  exception，避免把 ABI 问题误判为模型路径或 Transformers 版本问题。

## 2026-08-06：RLHF 依赖仍锁在 TRL 0.x，阻断 1.x 能力与兼容验证

### 现象

- `pyproject.toml` 将 RLHF extra 限制为 `trl>=0.19,<1.0`，标准 lock 为 0.29.1，无法使用 TRL 1.x 的新
  trainer 能力，也无法开始后续 distillation/OPD 接入。
- 直接换成 TRL 1.9.2 后，DPO/PPO/GRPO config 与 trainer API 基本兼容，但真实 GRPO CPU smoke 在构造时
  暴露 `SmokeTokenizer` 缺少标准 `convert_tokens_to_ids()` 接口。
- `docs/config_reference.md` 仍记录 TRL 0.29.1 与 vLLM 0.19.1 不兼容，和新版依赖事实不一致。这属于框架依赖
  与测试替身合同滞后，不是模型能力、eval、codec、metric 或数据问题。

### 根因

- TRL trainer/config import 在 `training` 和 `algorithms` 两处重复维护，依赖允许范围只存在于
  `pyproject.toml`，运行时没有统一的版本门禁和可诊断错误。
- 仓库 smoke tokenizer 只覆盖旧 TRL 实际调用的最小方法，没有实现真实 Hugging Face tokenizer 已具备的
  token-to-id 公共接口。
- vLLM 兼容区间是 TRL package metadata 的动态事实；旧文档把一次 0.29.1 解析结果写成了长期结论。

### 影响范围

- 影响 DPO、PPO、GRPO 依赖安装和后续 RL feature 开发；SFT、独立推理和已有 checkpoint 格式不受影响。
- GRPO 构造失败只发生在 `smoke_vlm` 测试模型；真实 Qwen tokenizer 已实现该公共接口，但若不修复测试替身，
  CI 无法验证 TRL 1.x 下的 GRPO 训练、保存与精确续训。

### 修复方式

- RLHF extra 升级到 `trl>=1.9.2,<2.0.0`、`datasets>=4.7.0`，lock 固定为 TRL 1.9.2；保持
  Transformers 5.10.1 与 vLLM 0.19.1 不变。
- 新增单一 `training.trl_compat` 真源，统一检查受支持版本、导入 DPO/PPO/GRPO config 与 trainer，并在依赖
  缺失、版本越界或 experimental PPO 路径变化时给出包含 installed/required 的显式错误。
- 让 `SmokeTokenizer` 实现 `unk_token` 与 `convert_tokens_to_ids()` 的真实公共合同，并更新其 fingerprint；
  没有在生产 GRPO 路径添加 smoke/model-family 特判。
- 文档改为记录当前 TRL 1.9.2 从 metadata 声明的 vLLM 区间 `>=0.17.0,<=0.25.1`；版本 gate 通过不改变
  vLLM rollout RNG 尚不可精确 checkpoint/resume 的独立限制。

### 回归测试

- TRL 1.9.2 + Transformers 5.10.1 + datasets 4.8.4 + vLLM 0.19.1 环境下，RLHF focused 86 项通过，覆盖
  config 物化、trainer 构造、pipeline、分布式参数合同和 checkpoint 指纹。
- `tests/test_pipeline_rlhf_smoke.py` 8 项通过：DPO 真实 tiny 训练/保存/精确续训、PPO 最短训练，以及 GRPO
  真实 tiny 训练/保存/精确续训均调用 TRL 1.9.2 本体。
- 仓库完整 smoke suite 24 项通过；实际读取 TRL package metadata 的 GRPO/vLLM 0.19.1 compatibility gate
  通过。
- 相关文件 ruff 与 compileall 通过；全程 `CUDA_VISIBLE_DEVICES=''`，未占用 GPU。

### 后续防线

- 升级 TRL 时必须先核对当前 PyPI release metadata 和 v0→v1 migration guide，再运行真实 config/trainer/step/
  resume smoke；只验证 import 不构成兼容验收。
- TRL import 路径、受支持版本与安装提示只能由 `training.trl_compat` 维护；算法层不得再次直接 import TRL。
- PPO 仍位于 `trl.experimental.ppo`，即使满足 1.x 版本范围也不能视为稳定 API；其路径或构造签名变化必须
  fail closed 并通过 focused smoke 后再升级 lock。

## 2026-08-08：Qwen3.6-27B 双机 16 卡 canary 暴露不可移植的数据指纹

### 现象

- Baidu01/Baidu02 源码、模型、配置和 JSONL SHA256 对齐后，第一次 16-rank 启动在
  `model-plan-local` gate 被拒绝；两端分别使用 Python 3.11.15/3.11.3，live Python/stdlib 实现指纹不同。
- 复用 Baidu02 现有 `.venv` 并对齐到 3.11.15 后，模型/sequence/batch gate 均通过，但
  `pre-model-data` gate 报两节点 `train_execution/train_stream` 指纹不同。两端训练 JSONL 大小、mtime、
  内容 SHA256 和 prompt pool SHA256 实际完全一致。
- 数据 gate 修复后，Baidu01 又暴露 `flash-linear-attention` 安装不完整：只有 `fla.layers/models`，缺失
  `fla-core` 提供的 `fla.modules`。一次未加 `--no-deps` 的叶包重装还会解析到 Torch 2.13/CUDA 13，不能在
  当前驱动和锁定运行时使用。

### 根因

- Shaft 的模型语义身份有意绑定 live Python/stdlib 实现；不同 patch-level Python 不能伪装成同一多机执行
  环境。
- Arrow record store v4 的 source fingerprint 同时绑定 `size/mtime_ns/ctime_ns`。`ctime` 是文件系统本地元数据，
  rsync 可以保留内容和 mtime，但不可能跨主机保留 ctime，因此等价本地 replica 必然产生不同 cache/data
  contract。这是框架数据身份误判，不是训练数据内容不同。
- Baidu01 的 `flash-linear-attention` metadata 存在，但共享 `fla` namespace 中的 `fla-core` 文件和 dist-info
  缺失。无约束重装会让解析器跟随 `fla-core -> torch` 选择最新 PyPI CUDA 13 栈；CUDA 13 包卸载时还会删除
  与 CUDA 12 wheel 共用 namespace 下的动态库文件，需要按锁文件恢复并原位重装 CUDA 12 runtime wheels。

### 影响范围

- 三类问题都在 canary 内发现；没有写 checkpoint、没有启动长期训练，也没有修改训练数据、模型权重或
  `gpu-holder`。
- record cache format 从 v4 升到 v5，旧 Arrow cache 会正常失效并重建；normalized record schema 和 target
  语义未变化。
- 只排除不可移植的 ctime；cache key 仍使用 canonical path、dataset/record/validation identity、size 和
  mtime。内容相同但 canonical path 或 mtime 不同的 replica 仍会 fail closed。

### 修复方式

- 以 Baidu01 为源码真源：Baidu02 原 dirty worktree 先保存为
  `pre-baidu01-source-sync-20260808` stash，再 fast-forward/rsync 到 Baidu01 当前源码；未弹出该恢复 stash。
- 不创建新环境；在 Baidu02 原 `.venv` 内对齐 Python 3.11.15 并按 `uv.lock` 同步。两端模型 plan fingerprint
  最终一致。
- Arrow record store 升级到 v5，source fingerprint 删除 `ctime_ns`，保留 size+mtime 普通变更失效语义；
  增加 ctime-only replica 回归测试。
- Baidu01 在原 `.venv` 中按锁文件恢复 Torch 2.10.0+cu128，并以 `--no-deps` 补齐 `fla-core`/CUDA 12 wheel
  文件；验证 torchvision、FLA、Qwen3.6 class、真实 BF16 CUDA 运算和 `uv pip check`。
- 双机 NCCL 强制 TCP：Baidu01 使用 `eth0`，Baidu02 使用 `eth4`，两端均设置 `NCCL_IB_DISABLE=1`；canary
  使用 DeepSpeed ZeRO-3、16 rank、full BF16、1M pixel cap 和一个真实 optimizer step。

### 回归测试

- `tests/test_data_sources.py` 16 项和相关 ruff 检查通过；新测试证明仅 ctime 不同的两个 replica source
  fingerprint 相同。两端修复文件、canary 配置、JSONL/prompt/model manifest SHA256 对齐。
- 16-rank canary 完整通过 data/model/finetune/optimizer/trainer gate，并完成 1/1 optimizer step：
  `train_loss=0.563232421875`、runtime `349.8346s`、useful tokens `35,941`、约 `102.77 token/s`；峰值单卡
  allocated/reserved 为 `28.85/39.15 GiB`，rank time skew `0.000802`，无 OOM、掉 rank 或 core dump。
- 训练中两端链路采样约双向各 `6.45 Gbit/s`；这说明 ZeRO-3 collective 正常，也确认跨管理网通信是主要
  性能瓶颈。canary 结束后两端 launcher/rank 均正常退出。

### 后续防线

- 多机本地 replica 启动前必须同时核对 Python patch version、源码/配置、模型 manifest、JSONL/prompt
  SHA256 和 canonical path；数据用 `rsync -a` 保留 mtime。不能把文件计数或 Git HEAD 当作完整一致性证明。
- `ctime` 不得进入跨节点逻辑数据身份。若未来要求“不保留 mtime 也可识别等价 replica”，应设计节点级
  content manifest/leader hash 缓存，不能让每个 rank 重复 hash 大 JSONL，也不能退回 stat-only 假一致。
- 在 uv 环境内修复单个叶包必须优先使用锁文件；确需原位重装时使用 `--no-deps`。禁止对带 Torch 依赖的
  包执行无约束 `--reinstall`，恢复后必须跑 CUDA、torchvision、模型类 import 和真实 operator smoke。
- 本 canary 未配置 `media_snapshot_id`，因此日志正确标记 exact-resume unsafe；正式可续训配置必须声明
  immutable media snapshot。双机 27B full SFT 虽可运行，但单步约 350 秒，不能把 16 卡视为接近 2 倍吞吐。

## 2026-08-08：Baidu02 Qwen3.6-27B LoRA DDP 在 2M batch 上显存峰值 OOM

### 现象

- Baidu02 使用 `banana_sft_27b_qwen36_v5_7_lora.yaml` 启动 8×A800 DDP LoRA；配置和运行时合同均确认
  为 per-device microbatch 1、GA=8、gradient checkpointing、2M pixel cap、10k token cap。
- 第 1 个 optimizer step 完成后，rank 3 在第 2 step 的 backward 中申请 5.88 GiB 失败：单卡总容量
  79.33 GiB、进程已占 73.50 GiB、系统只剩 5.81 GiB，PyTorch 另有 2.49 GiB reserved but unallocated。
- rank 3 未继续进入相同 collective，其他 rank 在 600 秒后报 NCCL ALLREDUCE timeout；torchrun 最终把
  rank 7 的 watchdog SIGABRT 显示为 launcher root failure。输出目录只有 startup/progress metadata，
  没有 checkpoint；异常退出使 `shaft_progress.json` 暂时残留 `status=running`。
- 将 token hard cap 从 10k 降到 8.5k 并启用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 后，重试从
  step 2 延后到 step 22 才失败，但 rank 7 仍在 backward 申请 6.67 GiB 时 OOM：进程已占 72.65 GiB，
  PyTorch allocated 71.27 GiB，reserved-but-unallocated 仅 471 MiB。

### 根因

- 这次的首个真实错误是 rank 3 CUDA OOM，NCCL timeout 是掉 rank 后的二次症状，不是网络或 collective
  顺序先出错。
- DDP 会在每张卡完整复制约 55.6 GB 的 Qwen3.6-27B BF16 base；本次 all-linear LoRA 另有
  124,730,880 个 trainable params。虽然 microbatch 已为 1 且开启 gradient checkpointing，2M/10k 上限下
  的视觉与语言激活仍把 80 GB 卡推到边界。
- 第一次失败中 2.49 GiB allocator reserve 说明碎片化参与了触发；但 8.5k + expandable-segments 重试时
  未分配 reserve 已降到 471 MiB 仍然 OOM，证明根因不是单纯碎片化，而是 DDP 完整复制 27B base 后没有
  足够的长程激活峰值余量。
  GA=8 主要改变 optimizer step 的累计次数，不会让 8 个 microstep 的激活同时常驻，因此不是直接根因。

### 影响范围

- 仅影响 Baidu02 的 27B LoRA DDP 长训；模型、数据、prompt、LoRA target 和学习率均按目标配置正确加载。
- Baidu01 full ZeRO-3 训练线不使用该 DDP 内存拓扑，不能从本次 OOM 推断其同样失败。
- 诊断只读取 tmux、metadata、GPU 和系统日志；没有重启训练、清理失败目录或操作 `gpu-holder`。

### 修复方式

- 初次诊断按“检查失败原因”范围未修改训练配置，也未自动重启。随后按用户决定，将 Baidu02 正式配置
  `banana_sft_27b_qwen36_v5_7_lora.yaml` 从 DDP 切换为 FSDP v2 full-shard：Qwen3.6 decoder/vision
  transformer auto-wrap、`use_orig_params=true`、reshard-after-forward、full state dict、limit-all-gathers；
  保留 model-side gradient checkpointing，并关闭当前栈不稳定的 FSDP activation-checkpoint wrapper。
- 由于 Shaft 当前 planned batching 只允许 DDP，FSDP 配置同时切换为
  `grouping=none/cardinality=fixed`。数据源、prompt、2M pixel、8.5k token、LoRA、LR、GA8、14k steps 和
  2k/limit4 保存策略均保持不变；这是先恢复可训练性的配置级方案，尚未解决极端样本 rank skew。
- 最小风险验证可以先在启动环境加入 `PYTORCH_ALLOC_CONF=expandable_segments:True` 做短 canary；这只
  缓解 allocator 碎片，不能作为完整 14k 稳定性的保证。
- 稳健方案是把 27B LoRA 改为已有验证骨架支持的 FSDP full-shard，从每卡移除完整 base replica；由于
  Shaft 当前不允许 `bounded_cost + FSDP`，需要同时切换为 fixed cardinality/grouping none，并重新做真实
  2M optimizer-step canary。若必须保留 DDP，则需降低 pixel/token hard cap、缩小 LoRA target/rank 或使用
  QLoRA，均属于训练语义变更，不能静默应用。

### 回归测试

- 已从 `tmux AI:0.0` 核对 run id、sequence contract、batch contract、finetune summary、optimizer summary
  和完整 rank traceback；确认实际 trainable params 为 124,730,880，microbatch/GA 为 1/8。
- 8.5k + expandable-segments 重试真实运行到 step 22；这可以证明调整改善了短程存活，但同时反证当前
  2M DDP 拓扑不足以稳定完成 14k 长训。
- FSDP 配置已在 Baidu01/Baidu02 分别完成严格解析；实际 auto-wrap 为
  `Qwen3_5DecoderLayer + Qwen3_5VisionBlock`，`reshard_after_forward=true`、`use_orig_params=true`，
  duration/GA/save 为 14000/8/2000，尚未启动真实模型 canary 或长训。
- 用户授权后，Baidu02 正式配置进一步从 `BS1/GA8` 改为 `BS2/GA4`，全局 sample batch 仍为 64；旧 run
  在无 checkpoint 时以 SIGINT 正常退出，新 run 在同一 tmux pane 前台启动并显式使用
  `WANDB_MODE=offline`。实际 batch contract 为 local/global/optimizer pack `2/16/64`、GA4；前 5 步无 OOM，
  观察显存最高约 47.4 GiB/卡。首步 114s、前两步均值 94.6s，随后第 3–5 步约 39–40s；相较旧配置前 7 步
  约 37.7s/step，尚未证明吞吐改善。当前 `grouping=none + padded` 的批内 padding 是继续判断 BS2 收益时
  必须同时量化的混杂因素，不能把更多显存占用直接等同于更高 useful-token throughput。
- 失败后 8 卡均只剩 `gpu-holder` 约 414 MiB 显存占用，训练 rank 已全部退出；没有残留 torchrun/train.py
  进程，也没有 checkpoint。

### 后续防线

- 27B LoRA 在 80 GB 上不能只用“base weights 能放下”作为容量判断；必须用目标 pixel/token cap 完成至少
  数十个不同 batch 的 forward/backward canary，并记录 peak allocated/reserved。
- 分布式失败必须按时间和 rank 追溯首个 Python/CUDA traceback；launcher 的最后 SIGABRT 或 NCCL watchdog
  不能覆盖更早的 OOM。
- bounded-cost 的变长 batch 建议在显存接近上限时默认评估 expandable segments，并保留显式的显存安全边界，
  不能以一次勉强通过代替长训容量验收。

## 2026-08-08：Baidu01 Qwen3.6-27B full ZeRO-3 被未平衡的极端长样本拖入 NCCL timeout

### 现象

- Baidu01 使用 full BF16 + DeepSpeed ZeRO-3、microbatch 1、GA4、2M pixel cap、10k token cap 训练，已正常
  完成 85/8000 step，平均约 21.5 秒/step；`shaft_progress.json` 最后一次写入为 21:22:14。
- 21:22:45 起，rank 0–6 在后续 ZeRO collective 等待，10 分钟后触发 600 秒 watchdog：default PG 卡在
  `_ALLGATHER_BASE` seq 350203，另一个 PG 卡在 `_REDUCE_SCATTER_BASE` seq 204265。rank 0–6 的 default
  PG 最后 enqueue 到 350204，而 rank 7 的 last enqueued/completed 均停在 350202，说明 rank 7 没有进入
  其他 rank 已等待的后续 collective。
- 全部日志没有 Python/CUDA OOM；内核日志没有 Xid、ECC 或 host OOM。输出目录只有 metadata/progress，尚未
  到 save step 4000，因此没有 checkpoint，异常退出后 progress 暂时残留 `status=running`。

### 根因

- DeepSpeed 路径受当前框架合同限制，使用 `grouping=none/cardinality=fixed`，没有 bounded-cost 的全局
  batch balancing；每个 rank 按确定性 sample plan 直接获得一条样本。
- 复现 step 85 后的确定性 sample schedule 后，optimizer step 86 的最后一个 GA slot 中，rank 7 获得
  `grounding_layout` row 26597：精确估算为 8,250 LLM tokens、7,696 vision patches；同一 microbatch
  最轻 rank 只有 819 tokens，token 长度相差约 10.07 倍。此前几个 microbatch 的 rank 间最大 token 比也
  达到约 3.64–5.88 倍。
- 这解释了 collective 轨迹：其他 rank 已完成较轻计算并进入后续 ZeRO allgather/reduce-scatter，rank 7
  仍停留在较重的执行位置。当前证据可以确定直接触发条件是未做 cost balancing 的极端 rank 负载长尾；
  日志没有 flight-recorder kernel stack，因此不能进一步严格区分“合法但超过 600 秒的长计算”与“该极端
  shape 触发 CUDA kernel 不返回”。NCCL timeout 是 rank 失步后的结果，不是首发网络故障。

### 影响范围

- 影响当前 Baidu01 27B full ZeRO-3 的 `grouping=none` 长训稳定性；已通过的均衡 16-rank 单步 canary 不足以
  覆盖 256k draw stream 中的长度长尾。
- 模型、训练数据和 prompt 均按配置正确加载；这不是 loss、标注语义或 checkpoint 损坏问题。
- 本轮只做日志、确定性 sample plan 与精确 cost 重放，没有清理输出、重启训练或操作 `gpu-holder`。

### 修复方式

- 本轮按诊断范围没有改配置或重启。不能用单纯调大 NCCL timeout 作为修复；若极端 shape 触发 kernel hang，
  延长 timeout 只会更晚失败。
- 稳健方向是让 DeepSpeed/FSDP 路径也拥有全局 cost-balanced fixed-cardinality sampler，或在不改变训练样本
  语义的前提下离线按精确 token/vision cost 做 rank 配平。临时规避只能降低 token/pixel hard cap，属于
  配置语义变化，需要用户确认。
- 下一次短 canary 应开启 `TORCH_NCCL_TRACE_BUFFER_SIZE` 与 timeout dump，并定向重放该 step 86 batch；
  若 rank 7 能在更长时间内完成，可量化安全 timeout；若同一 kernel 长期无进展，则按 kernel/shape 问题处理。

### 回归测试

- 用正式配置、seed 42、8-rank、GA4 重建 256,000 条 finite sample plan，plan fingerprint 为
  `0b022c7dbd0970870bdadfc263f20d9b8da3ec0da73af9252a1acc211c094313`。
- 使用 Qwen3.6 正式 processor/template 与 0.5M–2M pixel policy 精确计算 microstep 336–343 的 cost，确认
  step 86 最后一个 slot 的 rank 7 为 8,250 tokens/7,696 patches，最轻 rank 为 819 tokens/1,980 patches。
- 两台机器失败后均无残留 torchrun/train.py；GPU 只剩既有 `gpu-holder`。

### 后续防线

- 大模型分片训练 canary 必须覆盖训练流的 cost 分位点和极端样本，而不是只验证随机首个 optimizer step。
- `grouping=none` 用于 ZeRO3/FSDP 时，应在启动阶段审计每个 global microbatch 的 rank cost ratio，并对超阈值
  fail closed 或离线配平；只设置全局 max_length/max_pixels 不能保证 rank 间负载接近。
- 分布式 timeout 必须比较各 rank 的 last enqueued/completed sequence。若单一 rank 落后且无首发 NCCL/Xid，
  应先还原该 rank 的 sample cost，不能把 watchdog 末尾信息直接归因于网络。

## 2026-08-08：Baidu02 FSDP2 LoRA 增大 microbatch 未改善稳定吞吐

### 现象

- Baidu02 的 Qwen3.6-27B BF16 LoRA 从 `BS1/GA8` 改成 `BS2/GA4`，全局 sample batch 均为 64。新 run
  前 15 step 累计约 48.4s/step；扣除首步 114s 和第 2 步后的稳定区间约 41.3s/step，仍慢于旧 run 前 7 step
  的约 37.7s/step。增大 microbatch 没有产生预期的 FSDP 通信摊薄收益。
- 新 run 无 OOM，观察显存最高约 47.4 GiB/卡；因此当前结果不是显存容量不足，也不能用“显存未占满”解释。
- Transformers 启动日志明确警告：FSDP 下模型侧 `gradient_checkpointing` 会在 backward 引入冗余 AllGather，
  应优先使用 FSDP activation checkpointing。

### 根因

- 正式配置是 `model.finetune.mode=lora`。虽然 YAML 中保留 `qlora_load_in_4bit=true` 等字段，Qwen loader 只在
  `mode=qlora` 时构造 `BitsAndBytesConfig`；当前 27B base 实际仍以 BF16 加载，必须依赖 full-shard 才有稳定
  显存余量。
- 模型有 64 个 decoder layer 和 27 个 vision block，FSDP2 按这些 block wrap，并使用
  `reshard_after_forward=true`。每个微步均需反复 materialize 分片参数；模型侧 gradient checkpointing 又在
  backward 重算 forward，放大了参数 AllGather 开销。
- Shaft 当前 model policy 明确禁止 Qwen3.5/3.6 的通用 FSDP activation-checkpoint wrapper，正式配置只能用
  模型侧 checkpointing；这是当前最直接的 FSDP 性能缺口，不能只靠 YAML 打开
  `activation_checkpointing=true`。
- 当前 Transformers 5.10.1 + Accelerate 1.13 使用 FSDP2。该路径不支持 backward prefetch，且 Transformers
  仅在 FSDP1 分支消费 `forward_prefetch/backward_prefetch/limit_all_gathers`；这些 YAML 字段不能作为当前
  FSDP2 run 的有效性能旋钮。
- `grouping=none + padded` 会把每个 rank 的两条样本 pad 到本地最长样本，并保留 rank 间 cost skew；BS2
  节省的微步/通信次数会被额外 padding 与长尾等待抵消。因此本次慢不是单一 FSDP 实现问题，而是
  FSDP checkpoint 通信与未做 cost grouping 的叠加。

### 影响范围

- 影响当前 8×A800、BF16 base、FSDP2 full-shard LoRA、2M/8.5k padded 训练的吞吐；loss 与全局 batch 语义
  正常，不是数据或 objective 错误。
- 不能外推为 true QLoRA + DDP 的吞吐结论；后者会改变冻结 base 的数值精度和分布式拓扑，需要独立实验。
- 本轮诊断仅检查配置、运行时合同、安装栈源码和实时指标；没有停止或修改当前训练，也未操作 `gpu-holder`。

### 修复方式

- 若必须保持 BF16 LoRA：框架需要为 Qwen3.6 实现并验证 FSDP-aware activation checkpointing，在启用时关闭
  模型侧 gradient checkpointing；hybrid linear-attention 层不能未经定向 CUDA canary 就解除现有禁用策略。
- 同时为 FSDP/DeepSpeed 增加 fixed-cardinality cost-balanced sampler 与 exact-resume 状态，使 BS2 的两条
  样本按相近 token/vision cost 组 batch，并平衡 8 个 rank。只继续增加 BS 会进一步放大 padding。
- 若允许改变数值范式：独立测试真正的 `mode=qlora + DDP`，以 4-bit frozen base 换掉 full-shard 参数通信；
  不能把当前 `mode=lora` 下无效的 qlora 字段误认为已经进行了 4-bit 训练。
- 短期配置 canary 可测试关闭模型侧 gradient checkpointing，但必须先用 BS1 覆盖 2M/8.5k 长尾并记录峰值；
  这会显著增加 activation memory，不能直接用于正式长训。

### 回归测试

- 新 run 的实际 batch contract 为 local/global/optimizer pack `2/16/64`、GA4，FSDP effective contract 为
  version 2、`reshard_after_forward=true`、activation checkpointing false；LoRA summary 为 mode `lora`。
- 15 step 观察无 rank 退出或 OOM；显存约 35–47 GiB/卡，稳定步时约 41.3s。该窗口足以否定“BS2 已明显
  加速”的判断，但不足以证明 14k 长程稳定性或最终平均吞吐。
- 用户随后授权将 Baidu02 正式训练切为 true QLoRA + DDP。配置实际解析为 `mode=qlora`、NF4 double quant、
  BF16 compute、DDP、BS2/GA4，并恢复 `bounded_cost + token_budget`（8500 tokens、16384 vision patches）；
  独立输出目录为 `outputs/qwen36vl-sft/27b/banana-v5.7-qlora`。运行时合同和 finetune summary 均确认
  BitsAndBytes 0.49.2 与 qlora mode 生效，不再是只保留无效 qlora 字段的 BF16 LoRA。
- QLoRA DDP 前 8 step 正常完成、无 OOM，显存最高约 72.2 GiB/卡。前两步因 NF4 kernel/optimizer 冷启动
  均值约 119s/step，但第 3–8 步平均约 31.9s/step，比 FSDP BF16 LoRA 稳定区间约 41.3s/step 快约 23%。
  因而不能用前两个冷启动 step 判断长期吞吐；当前证据表明删除 FSDP AllGather 的收益已在稳定区间体现，
  但约 7 GiB 的显存安全余量仍需用 2M/8.5k 长尾继续验证。当前 run 按用户要求继续执行。

### 后续防线

- runtime contract 只应展示后端实际消费的 FSDP2 选项；对 FSDP2 不支持或上游忽略的 prefetch/
  limit-all-gathers 字段应 fail closed 或标为 inactive，不能制造参数已生效的错觉。
- 大模型吞吐实验必须同时记录 useful tokens/s、padding ratio、各 rank token/vision cost、collective time 和
  peak memory；GPU util 与静态显存占用都不能单独作为训练效率结论。

## 2026-08-09：planned batching 接入 FSDP full-shard 与 DeepSpeed ZeRO-3

### 现象

- 既有 `bounded_cost` planner 能在 DDP 下平衡多模态长尾，但配置层拒绝 FSDP/DeepSpeed。27B full BF16
  必须依赖 ZeRO-3 时只能退回 `grouping=none`，8 卡同一 microstep 曾同时出现 8,250 LLM tokens 与
  7,696 vision patches 的 rank cost 长尾，产生 collective 等待风险。
- 直接删除 normalize 限制并不安全：Accelerate 可能对已经按 rank 规划的 BatchSampler 再分片/补齐；
  DataLoader prefetch 还会让 live cursor 领先真正完成的 optimizer boundary。旧 backend-native checkpoint
  也没有把 planning state 与某次 FSDP/ZeRO generation 原子绑定。

### 根因

- canonical global microbatch 已存在，但 runtime 只有“global flattened sampler -> Accelerate 唯一分片”的
  DDP 消费方式，没有 sharded backend 的 rank-local ownership contract。
- planning callback 与 checkpoint storage protocol 分层正确，但 backend-native 路径此前缺少 typed commit
  marker，无法证明 native shard 集合与 committed sampler state 属于同一 generation。
- 该问题是训练 runtime/checkpoint integration 缺口，不是 grouping 算法、数据、prompt、processor 或
  eval/codec/metric 的误判。

### 影响范围

- 新能力只覆盖 SFT、steps duration、`bounded_cost + fixed + packing=none + padded`、固定每-rank
  cardinality、GA，以及 FSDP `full_shard + full_state_dict` / DeepSpeed ZeRO-3。
- DDP 既有执行与 `shaft_checkpoint_commit.json` 格式不变。token-budget、length、greedy/varlen、RLHF、
  world-size elastic resume、TP/CP/SP 仍 fail closed；不能以本次 tiny/canary 证据外推真实 27B 长训稳定性。

### 修复方式

- 复用 `ShaftBatchPlanner`、`ShaftPlannedBatchSampler` 与 `ShaftBatchPlanningState`。canonical plan 仍包含全部
  rank；DDP 保持全局扁平输出，FSDP/DeepSpeed 由 sampler 确定性只 yield 当前 rank，并让 Trainer 绕过
  Accelerate DataLoader sharding，消除二次切分、补齐、重复和遗漏。
- sharded callback 在 optimizer boundary 预览 state，收敛所有 rank 的 step/microstep/status/fingerprint，
  仅在完整 GA frame 和 optimizer update 成功后 commit。skipped update fail closed，不推进 committed cursor。
- planned backend-native checkpoint 使用唯一 prepared -> committed generation。commit marker 绑定 backend、
  step、world、Trainer/scheduler/RNG 小状态内容身份、完整 native shard 路径/非零尺寸集合和 planning
  binding；run-root 只选择同时满足 backend 与 planning validator 的最新 generation。大 shard 不额外重读
  hash，字节级内容继续交给 backend-native loader 校验。

### 回归测试

- focused 配置/sampler/DataLoader/checkpoint 测试覆盖：首版 accept matrix、全部未支持组合拒绝、draw 恰好一次、
  rank 无交叠/无丢失、等 microstep、无 `BatchSamplerShard`、GA/world/config/generation/state drift fail closed、
  skipped update 不提交，以及 heavy-text/heavy-vision 同 global microbatch 的 rank-skew 改善。
- CUDA 0、1：2-rank FSDP full-shard 和 DeepSpeed ZeRO-3 均完成 fresh、checkpoint、interrupted resume；
  sample stream、model、optimizer、scheduler、RNG 与 planning state 等价。
- CUDA 0–7：真实 8-rank ZeRO-3 一步 canary 完成 8 个 global draws，覆盖 8,250-token 和
  7,696-vision-patch 极端样本；checkpoint 含 8 个 model shard、8 个 optimizer shard，direct/run-root
  resolver 与 planning generation 验证通过。训练进程正常退出，未操作 `gpu-holder`。

### 后续防线

- 新 backend 只能消费 canonical planner 的 rank view，禁止在 pipeline、trainer 或 backend adapter 建第二份
  grouping/state；任何会 split/dispatch/even-pad planned batches 的组合必须在首个 forward 前拒绝。
- checkpoint 只能保存 optimizer-boundary committed state。任何 rank 异常、OOM、skipped update、pending/
  torn marker、native artifact 缺失或 generation 不一致必须让全体 rank fail closed。
- 扩展 token-budget、length、packing/varlen 或 RLHF 前必须分别补 topology-specific sampler、loss 和
  backend-native exact-resume 证据，不能复用本次 fixed padded 结论直接解除限制。

## 2026-08-09：27B full ZeRO-3 的独立资源上限未覆盖联合显存峰值

### 现象

- Baidu01 正式 Qwen3.6-27B full BF16 + ZeRO-3 在启用
  `bounded_cost/fixed/none/padded` 后正常运行至完成 step 94，随后在 step 95 的最后一个 GA microstep 报
  `Triton Error [CUDA]: out of memory`。失败前每卡常驻约 79.8–80.8k MiB，80 GiB 卡仅余约 1–2 GiB。
- 旧 `grouping=none` 运行没有先报告 OOM，而是在 step 86 由其他 rank 报 NCCL timeout。

### 根因

- 两次失败对应同一个逻辑 draw `2751`、同一 grounding 样本
  `json__prod_030188__degraded_00`：其精确成本同时为 8,250 LLM tokens 和 7,696 pre-merge vision patches。
  旧 fixed stream 中它位于 step 86/GA3/rank7；新 planner 重排到 step 95/GA3/rank0。旧运行的 NCCL timeout
  只说明其它 rank 在等待这个长尾 rank，并不能证明该样本已完成或能够稳定装入显存。
- `max_tokens_per_microbatch=10000` 与 `resource_budgets.vision_patches=8192` 是两个独立 hard cap。planner
  保证各维分别不超限并平衡 rank cost，但没有声明或校验二者同时接近上限时的联合 activation/workspace
  显存。该样本两个维度都合法，归一化和为 1.7645，却超过当前 27B ZeRO-3 的真实可执行 envelope。
- 先前 8-rank canary 验证的是同一 global microbatch 内一个 heavy-text draw（8,250 tokens、低 vision）和
  一个 heavy-vision draw（7,696 patches、短文本），不是单个 draw 同时命中两个极值，因此没有覆盖本次
  Triton 联合峰值。这是验收缺口，不是 grouping 丢样本、重复分片或 pixel budget 未生效。

### 影响范围

- sharded planned batching 的 rank 分片、配平和 exact-resume 结论仍成立；缺口在大模型运行时 admission
  control。降低 `buffer_size` 只会推迟该 draw，增大 NCCL timeout 也不会解决单卡 OOM。
- 10k token 和 2M pixel 各自是合法输入上限，不能继续直接等同于“可同时达到”的 27B full 训练显存合同。
  4B、QLoRA 或显存状态余量更大的拓扑不能直接套用同一联合阈值。

### 修复方式

- 本轮只完成根因定位，未擅自修改训练语义或重启。下一步应先对真实联合长尾样本做隔离 canary，建立
  token/vision 联合显存 envelope；再选择动态降低该样本的图像预算/文本上限，或通过 optimizer/parameter
  offload 等方式释放持久显存。
- 若在框架中增加联合 budget，必须作为 batch contract、planner spec、metadata 和 exact-resume fingerprint
  的单一配置真源；不得只在训练脚本中对个别 sample id 打补丁或静默丢样本。

### 回归测试

- 通过确定性重建前 380 个 planned microsteps，复现 step 94 已完成 frame 与 step 95 失败 frame：step 94
  最重样本约 5,744 tokens + 7,788 patches（联合归一化 1.5251）可执行；step 95/GA3 同时出现
  8,250+7,696、8,166+7,744 和 7,938+7,680 三个联合长尾 draw。
- 后续 GPU gate 必须增加“同一个 sample 同时 heavy-text + heavy-vision”的真实 processor/collator 输入，
  不能再用分离的两个 draw 代替联合峰值。

### 后续防线

- 多维 cost balancing 与显存 admission 是两个不同契约：前者减少 rank skew，后者保证单 rank 可执行。
  任一新 topology 在发布前必须覆盖单维极值、联合极值、GA frame 和 optimizer workspace 四类 canary。
- 大模型正式长训前至少跑到训练流中首个已知联合长尾 draw，不能只观察前几个轻量 step 后宣布显存可行。

### 2M 图像预算下的序列上限审计

- 对 v5.7 前 10,000 个确定性 scheduled draws 使用真实 Qwen3.6 processor 做 exact-cost
  审计。总序列上限为 6,000/6,144/6,500/7,000 时，分别有 46/39/23/19 条需要截断，
  占 0.46%/0.39%/0.23%/0.19%，全部来自 `grounding_layout`。
- 当前不改框架的保守值是 6,000：它把接近 2M 的联合长尾压到与已成功样本接近的
  envelope，同时只影响不到 0.5% 的 scheduled draws。`data.max_length` 负责真实截断，
  `data.batching.max_tokens_per_microbatch` 负责 planner hard guard，两者必须同步；仅降低后者会在
  长样本上直接报 oversize。

## 2026-08-10：OPD 不能继续接入单体 RLHF pipeline

### 现象

- 旧训练入口按算法名分支选择 SFT/RLHF，`ShaftRLHFPipeline` 又直接持有 DPO/PPO/GRPO 的 dataset、collator、
  trainer 参数、采样和 checkpoint 差异。若继续加入 OPD，只能新增更多 `if/elif opd`，并把 teacher/student、
  rollout 与 distribution loss 塞进已有 RL 主链。
- OPD 的 teacher 身份、生成 RNG 和 objective 会影响 exact resume，但公共 resume contract 原先通过硬编码算法
  名称与算法字段扩展，训练公共层开始反向依赖具体算法。

### 根因

- 顶层只有“算法”概念，没有 SFT、RL、OPD 三个并列训练域；公共编排层承担了本应由域和算法 runtime
  持有的业务决策。
- Arrow record 类型、resume policy 和 RL trainer 参数都使用封闭映射或算法专用字段，新增算法需要修改多个
  中央模块，扩展边界不是注册式合同。

### 影响范围

- 直接影响新增 OPD、后续 RL 算法扩展、输入真源和 exact resume 可审计性；若不重构，单次 smoke 能跑通也
  无法证明 teacher 不会进入 optimizer、不同 rank 的 token normalization 正确或恢复轨迹等价。
- 既有 SFT 算法行为未改；DPO/PPO/GRPO 的执行行为保留，但 canonical 顶层名称收敛为 `rl`，旧
  `rlhf` 类名、函数和 pipeline key 仅作为兼容别名保留。

### 修复方式

- 新增 training-domain registry，由 algorithm profile 声明唯一 domain，中央入口只做 registry dispatch。
- 把 DPO/PPO/GRPO 差异迁入 `src/shaft/rl` runtime registry；RL pipeline 只做公共阶段编排。新增独立
  `src/shaft/opd` 与 `ShaftOPDPipeline`，拥有 prompt-only data、student/teacher role、rollout、loss、trainer
  和 resume policy，不导入 TRL experimental OPD trainer。
- record type 与 training resume policy 改为公共注册机制；算法实现自行注册扩展。OPD checkpoint contract
  绑定 teacher artifact、tokenizer/processor/template 兼容指纹、rollout/objective、sample assignment 与 RNG。
- canonical RL 实现与 API 位于 `pipeline/rl.py` 的 `ShaftRLPipeline/run_rl`；`pipeline/rlhf.py` 只保留
  `ShaftRLHFPipeline/run_rlhf` 薄导出，不承载训练逻辑。运行时 point 同步收敛为 `pipeline.rl.run`，OPD
  独立使用 `pipeline.opd.run`，二者均拥有各自的 start/done observability interceptor。

### 回归测试

- CPU loss oracle 覆盖 causal shift、completion-only mask、pad/eos 相同、forward KL、reverse KL 和 JSD。
- tiny CPU 两步 fresh 与 checkpoint resume 验证 student 更新、teacher 冻结，model、optimizer、scheduler、RNG
  与 sample stream 恢复等价；标准 HF 输出目录生成成功。
- 2-process CPU DDP 验证各 rank 的 student 更新、teacher 不变和 committed checkpoint；RL focused 回归在
  canonical 命名收敛后通过。
- 发布权重 CUDA gate 已固化为一个 manual integration：真实 Qwen3VL-2B student、Qwen3VL-4B teacher，依次
  覆盖单卡有序双图 BF16 LoRA 非零更新/PEFT validate 与两卡单图 DDP GA=2 sampled-rollout exact
  resume/export。
- 真实 Qwen3VL CUDA 单卡/两卡尚未执行：当前 CUDA 0–7 被用户 27B ZeRO-3 长训占用。本轮未干预该任务，
  也未操作 `gpu-holder`，因此不能声明 OPD production-ready。

### 后续防线

- 新训练域只能注册 domain runner；公共 CLI、pipeline 与 training 层不得读取具体 OPD/RL 参数或新增算法
  `if/elif`。新 RL 算法只能实现 runtime contract，不能修改 RL pipeline 拼装逻辑。
- CPU tiny、CPU DDP、真实模型 CUDA 与真实多图是不同证据层级，文档必须分别标注。GPU 空闲后需补真实
  Qwen3VL 单卡 generate/score/backward/export reload 与两卡 DDP exact-resume gate。
- external teacher、vLLM buffer、top-k/chunk、sharded backend、packing/varlen 在实现状态持久化和专项测试前
  必须 fail closed，禁止静默降级。

## 2026-08-10：真实 Qwen3VL 类 OPD gate 暴露三处模型族合同缺口

### 现象

- teacher 与 student 使用内容完全相同、但保存在不同目录的 Qwen3VL processor/tokenizer 时，OPD preflight
  错误报告 processor contract 不一致。
- 放开错误拒绝后，真实 Qwen3VL score forward 报 `mm_token_type_ids` 长度仍是 prompt 长度，而
  `input_ids/attention_mask` 已包含 completion，mask shape 不匹配。
- generate、student/teacher score、backward 与参数更新成功后，最终 HF export 将训练态
  `text_config.use_cache=false` 写入 `best/config.json`，与源 artifact 的部署态 `true` 不一致。

### 根因

- OPD 直接使用通用 component state fingerprint 比较 processor；该 fingerprint 包含 `name_or_path` 等加载位置，
  混淆了 artifact locator 与输入行为兼容性。
- trainer 只扩展了 `input_ids/attention_mask`，没有模型 adapter 扩展点描述 processor 产生的其它序列对齐字段。
  tiny smoke model 不产生 `mm_token_type_ids`，因此此前测试无法暴露问题。
- 训练加载会保留部署 cache 默认值并临时禁用 `use_cache`，但只有 SFT trainer 的 `save_model()` 使用
  `export_model_cache()`；OPD 等其它 checkpoint-capable trainer 没有共享该导出合同。

### 影响范围

- 第一项会拒绝正常的独立 teacher/student artifact；第二项使真实 Qwen3VL OPD 无法训练；第三项会生成可
  reload 但部署配置漂移的 HF artifact。
- Qwen 的问题字段是 `mm_token_type_ids`，但根因适用于任何在 prompt processor 输出中携带额外序列字段的
  模型族，因此不能在 OPD trainer 内追加 Qwen 专用判断。

### 修复方式

- 新增 locator-neutral 的 input component semantic signature，processor/template 按行为状态比较；teacher
  权重和 student 权重仍分别由 model-plan artifact identity 与 checkpoint contract 绑定。
- 在 `ProcessorPolicy` 增加显式 `rollout_sequence_fill_values` 和统一 scoring-input builder。通用层验证
  batch/sequence shape 并 fail closed；Qwen policy 声明 completion 的 `mm_token_type_ids=0`，OPD trainer 只
  调用 model adapter，不感知模型族字段名。
- 将 `export_model_cache()` 下沉到 `ShaftCheckpointCommitMixin.save_model()`，使 OPD、DPO、GRPO 与 SFT 的
  periodic/final model save 共享部署 cache 恢复合同；SFT 的分布式同步 override 继续保留。

### 回归测试

- processor policy 单测证明 Qwen `mm_token_type_ids` 从 prompt 长度扩展到完整 rollout 长度，同时 image grid
  与 pixel tensors 保持原始语义。
- integration gate 使用真实 Transformers `Qwen3VLForConditionalGeneration` 模型类、两个不同随机初始化的
  student/teacher、发布版 Qwen3VL processor/tokenizer 和两张有序图片，完成 sampled generate、student/
  teacher score、backward、student 参数更新与 teacher artifact 不变。
- 同一 gate 完成 checkpoint-1 -> checkpoint-2 exact resume，model、optimizer、scheduler、RNG、Trainer state
  和 final HF export 等价；`best/` 可由标准 AutoModel/AutoProcessor reload，部署态 `use_cache` 与源 artifact
  一致。

### 后续防线

- 新模型族接入 OPD 时必须通过 processor policy 声明所有 prompt-length sequence fields 的 completion 扩展
  语义；未知 sequence field 必须在首个 forward 前拒绝，禁止 trainer 内按模型名或字段名打补丁。
- processor/tokenizer 的“行为兼容”与 model artifact 的“来源身份”必须使用不同合同。前者排除 locator，后者
  继续绑定 revision/content identity，二者不得互相替代。
- tiny 通用模型只能验证算法数学与 Trainer 生命周期；发布前至少增加真实上游模型类 + 真实 processor 的
  CPU gate，以及发布权重 CUDA 单卡/多卡 gate。当前 CPU gate 不能替代尚未执行的发布权重 GPU 验收。

## 2026-08-10：OPD optimizer-window 归一化、prompt 截断与 Qwen logits 内存合同

### 现象

- OPD 已按每个 microbatch 的 DP 全局 completion token 数归一化，但 `gradient_accumulation_steps>1` 时，HF
  最终累积的是多个 microbatch mean 的平均，不等于整个 optimizer window 的 token mean。
- `data.max_length` 文档要求给 rollout 预留 completion 空间并结构化截断 prefix，OPD collator 实际只在 prompt
  超限时报错，执行语义与合同不一致。
- Qwen3VL student/teacher score 对完整 prompt+completion 物化 full-vocab logits；长多模态 prompt 下，绝大
  部分 prompt logits 不进入 completion loss，却会显著放大显存。
- GA=2 fresh 训练完成后，同一进程立即从 checkpoint 恢复曾被错误拒绝为 optimizer implementation drift。

### 根因

- completion 长度只有 student rollout 后才知道，不能直接使用 HF 在 dataloader 预取阶段计算的
  `num_items_in_batch`；此前只处理了 rank 维度，没有处理 GA 时间维度。
- OPD collator 绕过了 SFT/DPO 已有的 template prefix-layout/truncation 机制，形成了第二套不完整输入路径。
- model adapter 只描述 rollout sequence fields，没有描述模型 forward 能否只投影一个经过验证的 logit tail。
- optimizer/trainer selected-callable identity 递归绑定了 PyTorch 运行期依赖；foreach optimizer 路径执行后，
  上游内部 cache 变化会改变 fingerprint，混淆“代码行为变化”和“同一实现的运行期状态变化”。

### 影响范围

- completion 长度在 microbatch 间不同时会改变真实优化目标；短 completion microbatch 被过度加权，且
  `train_loss` 也不是 optimizer-window 全局 token mean。
- 长 prompt 无法按配置严格上限训练；直接 token slice 又会破坏 image expansion、special token 和 media
  顺序。
- 发布权重 Qwen 的 OPD score 可能在真实训练开始前因无用 full-vocab prompt logits 产生动态 OOM。

### 修复方式

- `ShaftOPDTrainer` 每个 microbatch 一次性 all-reduce detached numerator/denominator；同一 GA window 内，
  新 denominator 到达后先重标已有梯度，再对当前 numerator backward，使 clipping/optimizer step 最终看到
  全 DP、全 GA 的一个 completion-token mean。非末 microbatch 报告零，窗口末报告统一全局 loss。
- template 新增通用 `ShaftTemplatePromptPlan/Row`；OPD collator 复用同一 exact processor-token layout 做
  结构化 prompt 截断和 processor input assembly，不在 OPD 内复制 token/media 规则。
- model policy 新增 `ShaftRolloutScoringPlan` 与可选 tail-logits input 声明。Qwen VL 声明
  `logits_to_keep=completion_width+1`，同时严格校验输出 logit span 和 completion mask；OPD trainer 不读取
  Qwen 字段名。
- resume contract 将 selected class 绑定为稳定的 module + qualname reference，并另行绑定 owning-module
  source policy；optimizer/scheduler builder 绑定自身 code/default/closure，但不递归绑定可变的外部运行期
  dependency cache。Torch/Transformers/Accelerate 版本仍由独立 runtime package contract 约束。

### 回归测试

- GA=2 oracle 使用 1-token 与 3-token completion，证明累积梯度逐参数等于一次按 4 个 completion tokens
  归一化的参考梯度，并验证窗口报告 loss。
- OPD collator 回归证明长 prompt 截到预留后的严格预算，同时保留 processor/media 输入合同。
- 真实 Transformers Qwen3VL 模型类、多图 processor、sampled rollout、backward、exact resume 与 HF reload
  gate 继续通过；两进程 CPU DDP 与 GA=2 resume/commit 路径通过。
- GA=2 sampled rollout 的 checkpoint-1 -> checkpoint-2 uninterrupted/resume 再次达到 model、optimizer、
  scheduler 与 RNG 精确等价，且同进程恢复不再受 optimizer runtime cache 影响。
- OPD pipeline smoke 同时覆盖 full 与 LoRA student：两者均完成参数更新、GA=2 exact resume；final export
  分别生成标准 `model.safetensors` 与 `adapter_model.safetensors/adapter_config.json`，PEFT 权重非空且有更新。

### 后续防线

- 任何 token-normalized objective 都必须分别证明 DP rank 维度与 GA microbatch 维度；单独的跨 rank
  all-reduce 不能作为 optimizer-window 正确性的证据。
- rollout 输入不得在算法 trainer 中手工切 prompt 或硬编码模型字段。结构化截断归 template，sequence/logit
  forward 差异归 model policy，无法对齐时在 forward 前 fail closed。
- 发布权重 CUDA 单卡与两卡 DDP 仍是 OPD production gate；CPU 真模型类证据不能替代显存与 NCCL 运行证据。

## 2026-08-10：OPD trainer 直接绑定本地 rollout/teacher 会重新制造 backend 分支

### 现象

- 三个 training domain 已经分离，但 `ShaftOPDTrainer` 仍直接调用 student `model.generate()`，并直接持有、
  移动和调用本地 teacher module。
- 一旦增加 vLLM rollout、外部 teacher 或 logits service，最直接的改法会在 trainer/pipeline 中新增
  `if/elif backend`，使 OPD 自己再次退化为单体分支实现。

### 根因

- 首版先完成了算法数学和 HF 生命周期，却没有把“生成 completion”和“提供 teacher score”建模为两个
  独立的开放扩展轴。
- checkpoint contract 绑定了 rollout 配置和 teacher artifact，但没有显式声明所选执行实现是否能够
  exact resume。

### 影响范围

- 当前 `hf_local` 行为本身正确；风险位于后续扩展和恢复能力声明，不属于模型能力、eval、codec、metric
  或 data 误判。
- 若不先收口，新增后端会同时污染 OPD trainer、pipeline、resume 和测试，且可能错误允许无法恢复外部
  RNG/buffer 的配置保存 checkpoint。

### 修复方式

- 新增 OPD 域内 `OPDExecutionRegistry`，分别注册 `OPDRolloutBackend` 与 `OPDTeacherProvider`，在大模型
  加载前解析为不可变 `OPDExecutionPlan`。
- execution plan 统一检查 exact-resume capability；trainer 只消费 `OPDExecutionRuntime`，不再直接包含
  本地 HF rollout 或 teacher 生命周期逻辑。
- 当前 `HFLocalOPDRolloutBackend` 继续由 Torch/CUDA RNG 持有采样状态；
  `LocalHFOPDTeacherProvider` 负责冻结、device placement、eval 和 no-grad score。
- resume implementation identity 显式绑定所选 backend/provider 的名称、实现 callable 与 module policy。
- feature review 发现 pipeline 的 teacher role materialization 仍是本地 HF 专用；因此 V1 normalize 明确只
  接受 `hf_local`，不把内部 scorer registry 误报为 external teacher 已可选。

### 回归测试

- contract 单测证明替代 backend/provider 可在局部 registry 中注册和解析，且未声明 exact resume 的实现
  会在 checkpoint preflight 被拒绝；未知名称列出已注册实现并 fail closed。
- 配置负例证明 `vllm` rollout 或 `external` teacher 在加载任何模型前被 V1 capability gate 拒绝。
- OPD component 与 pipeline smoke 全部通过，继续覆盖 full/LoRA、teacher 不变、student 更新、sampled
  rollout 和 GA=2 exact resume。
- `ruff check`、`compileall` 与 `git diff --check` 通过。
- 两进程 CPU DDP focused 回归通过；真实 Qwen3VL 类 + 发布版 processor 的 CPU 多图、sampled rollout、
  backward、exact resume 与 HF reload gate 通过；默认全仓 `pytest -q` 100% 通过。

### 后续防线

- 注册表必须拥有 selection、capability 和 implementation identity，而不只是工厂名字；中心调用方只能
  resolve/build/invoke。
- 任何带外部 RNG、异步队列或 rollout buffer 的实现，在 state/cursor/request seed 接入 checkpoint 前
  必须保持 `exact_resume_supported=false`。
- provider/backend 不能接管模型族 tensor 字段；多模态输入扩展继续唯一归属 `ProcessorPolicy`。

## 2026-08-10：移除 RLHF pipeline/CLI 兼容双轨

### 现象

- DPO/PPO/GRPO 的实现已经迁入 `src/shaft/rl`，唯一业务实现也已是 `ShaftRLPipeline/run_rl`，但仓库仍
  注册 `rlhf` CLI、`shaft_rlhf` pipeline key，并导出 `ShaftRLHFPipeline/run_rlhf` 与一个兼容模块。
- 这些名字不包含业务逻辑，却形成第二套公开入口、文档和测试义务，与 SFT/RL/OPD 三域并列的完成定义
  不一致。

### 根因

- 迁移 RL 单体实现时为了短期兼容保留了薄 shim；feature 完成后的全局 review 未立即执行“已有正确真源后
  删除临时桥接”的最后一步。

### 影响范围

- 训练数学、数据、checkpoint 和已有 RL runtime 行为不受影响；问题属于入口与命名双轨，不是模型能力或
  eval/codec/metric/data 误判。
- 若继续保留，新代码仍可能引用旧名字，使后续又出现两个 pipeline key、两个 CLI 文档和兼容分支。

### 修复方式

- 删除 `src/shaft/pipeline/rlhf.py`，移除 `ShaftRLHFPipeline/run_rlhf` 导出与 `shaft_rlhf` registry key。
- `RLCommand` 只注册 `rl`；训练 CLI 现在严格对应 `sft / rl / opd` 三个 domain。
- `rlhf.*` 配置节点与 `rlhf` 可选依赖 extra 保留：前者仍是 DPO/PPO/GRPO 的结构化配置命名，后者表示
  TRL/RLHF 依赖集合，不是第二训练入口。

### 回归测试

- registry contract 明确断言 pipeline keys 只有 `shaft_sft/shaft_rl/shaft_opd`，command keys 只有
  `sft/rl/opd`。
- CLI、pipeline registry 与通用 CLI focused 测试通过；`ruff`、`compileall`、`git diff --check` 和默认全仓
  `pytest -q` 100% 通过，证明没有旧公开导入者残留。

### 后续防线

- 域迁移完成后不得永久保留无必要的类名、函数、registry key 或 CLI shim；HF/TRL artifact 兼容不等于
  Shaft 必须保留内部旧入口。
- `rlhf.*` 配置语义与 `rl` training domain 是不同层级，不能因为清理入口而机械重命名稳定配置 schema。

## 2026-08-11：通用 collator 泄漏 Qwen sequence 字段语义

### 现象

- OPD collator 直接读取、padding 并回填 `mm_token_type_ids`；同样的字段属性还存在于通用 template row、
  SFT/DPO collator 与 varlen batch builder。
- `ProcessorPolicy` 虽然声明了 rollout 补零值，但 target 拼接、prefix 截断、DPO pair 扩展和 varlen 拼接仍由
  多个调用方分别解释同一字段。
- 当前 Qwen3VL 可运行，但接入任何使用不同字段名、不同 token 轴或非零 padding 的模型都会再次要求修改
  OPD/SFT/DPO 通用代码。

### 根因

- `ShaftProcessedBatch` 只保存 processor 原始 tensor，没有携带本 batch 实际解析出的 sequence-field layout；
  template row 因而退化为 Qwen 专属字段容器。
- token layout 只表达 canonical/processed boundary，没有显式表达不可被 prefix truncation 删除的 media span，
  template 只能借 `mm_token_type_ids` 猜测保护范围。

### 影响范围

- 影响 SFT、DPO、OPD 和 varlen 的 processor sequence 输入装配与结构化 prompt 截断；不改变 loss、optimizer、
  sampling、eval、codec、metric 或原始数据。
- 这是框架扩展边界问题，不是模型能力或数据质量误判。已有 Qwen 训练结果不因本次重构失效；输入 tensor
  语义保持一致。

### 修复方式

- 在模型层新增 `ShaftProcessorSequenceField`，由 `ProcessorPolicy.processor_sequence_fields` 唯一声明字段名、
  batched token axis、padding 值和 continuation 值；允许模型专用子类覆盖非恒定扩展/拼接行为。
- `ShaftProcessedBatch` 保存实际出现的 resolved field contract，并统一实现 prompt row 提取、prefix index
  投影、continuation 扩展、padded/varlen collation。原始输出、resolved contract 与 collator 输出必须完全
  一致，缺项或多项都在模型 forward 前失败。
- template row 只返回 `processed_prefix_indices`，SFT/DPO/OPD collator 统一调用 processed-batch contract；删除
  通用 template、collator、OPD 与 varlen builder 对 `mm_token_type_ids` 的所有引用。
- `ShaftProcessorTokenLayout` 增加 `protected_processed_spans`；Qwen policy 负责从 image placeholder/token-type
  run 生成保护区，template 只按通用 span 阻止媒体 token 被截断。

### 回归测试

- 单测使用未在通用代码出现过的 `alternate_positions`，覆盖 token axis=2、非零 continuation、负数 left
  padding、prefix selection，以及缺少 continuation rule/processed-batch contract 的 fail-fast 负例。
- OPD collator 主链用动态 processor 输出 `alternate_sequence`，证明无需修改 collator 字段名单即可正确
  left-pad；单 token media protected-span 测试证明截断不再依赖 Qwen token type。
- OPD focused、SFT/DPO collator、varlen、template 与 processor-policy 测试通过；真实 Transformers Qwen3VL
  CPU 多图 rollout/backward/HF reload、2-process CPU DDP、full/LoRA sampled-rollout exact resume 均通过。
- 默认全仓 `pytest -q` 100% 通过；`ruff check`、`compileall` 与 `git diff --check` 在最终收口后再次执行。

### 后续防线

- 新模型的 processor sequence 输出只能在模型 policy 注册，不得在 template/collator/trainer 增加字段属性或
  字符串分支。无法用常数 continuation 描述时必须实现模型专用 field 行为，不能默认补零。
- 新字段必须同时覆盖 padded、DPO pair、OPD rollout；声明 varlen 支持时还要覆盖非 1-D token axis 拼接。
- media protection 只由精确 token layout 提供；template 不得读取任何模型 token-type 或 image-token 字段。

## 2026-08-11：OPD CUDA release gate 暴露设备隔离与 telemetry 能力漂移

### 现象

- 发布权重门禁外层 pytest 需要同时看见 CUDA 0、1，但单卡子进程继承两张可见卡，被框架的单进程
  DataParallel 安全检查正确拒绝，门禁无法进入训练。
- 修复设备隔离后，真实单卡、DDP fresh 和 DDP resume 都成功；最终通用 checkpoint 比较器却要求恢复
  `shaft_training_efficiency_rank*.json`，而 OPD checkpoint 没有这些文件。
- `train.efficiency.enabled` 默认是 `true`，OPD pipeline 只清理旧 summary，却既不构造 monitor 也不拒绝配置，
  形成静默忽略的错误能力声明。

### 根因

- 门禁 helper 没有区分“外层测试可见卡集合”和“单卡/双卡子进程实际拓扑”。
- checkpoint 等价 helper 把 SFT release gate 的 telemetry 合同无条件套给 OPD。
- SFT efficiency protocol 依赖 collator 的 supervised-token/loss-mass/vision work；OPD 的实际 optimizer frame
  还包含 rollout、student score 和 teacher score，当前没有可复用的正确 measurement protocol。

### 影响范围

- 第一项只影响 manual gate 编排，不影响生产 topology guard；不得通过放宽 guard 修复。
- 第二、三项影响 OPD 配置和验收的真实性，不影响已经完成的权重、optimizer、scheduler、RNG 或 Trainer
  exact resume。问题属于 observability/config contract，不是模型、data、eval、codec 或 metric 能力误判。

### 修复方式

- `_run_qwen_opd_cuda_gate()` 从外层 `CUDA_VISIBLE_DEVICES` 解析稳定设备列表：world-size 1 子进程只暴露第一
  张卡，world-size 2 torchrun 暴露前两张；资源守护和生产安全检查均保持不变。
- OPD V1 normalize 明确拒绝 `train.efficiency.enabled=true`，测试/门禁配置显式关闭。后续必须先定义 OPD
  专用 rollout/student-score/teacher-score protocol，不能补一份名为 efficiency 的错误 SFT 指标。
- checkpoint 等价 helper 增加显式 `efficiency_expected` 合同。OPD 仍严格比较 adapter、optimizer、scheduler、
  scaler presence、每 rank RNG 和 Trainer state，并反向断言双方均不存在 efficiency snapshot；其它已启用
  SFT 门禁继续恢复并比较完整 snapshot set。

### 回归测试

- OPD config 负例证明默认/显式开启 telemetry 会在模型加载前失败；全部 OPD fixture 显式关闭该能力。
- 真实 Qwen3VL-2B student + 4B teacher CUDA release gate 于 2026-08-11 在 CUDA 0、1 通过：单卡有序双图
  BF16 LoRA sampled rollout、非零 adapter 更新与 PEFT reload；两卡 DDP GA=2 fresh 和 checkpoint-1→step-2
  resume；adapter、optimizer、scheduler、每 rank RNG、Trainer state 与最终 export 完全一致。
- 全仓 CPU 回归、ruff、compileall 与 diff-check 在文档收口后再次执行。

### 后续防线

- manual gate 必须显式控制每个子进程的可见设备集合；外层 device-count preflight 不能代替子进程拓扑。
- 共享配置默认值不代表每个 training domain 自动支持该能力；未消费的 `train.*` 能力必须 fail closed。
- 跨算法 checkpoint helper 只能比较明确声明的共同状态；算法专属 extension 必须由显式 capability 参数控制，
  不能因为文件缺失而跳过，也不能无条件要求不存在的状态。

## 2026-08-11：Qwen3.6 full SFT 静默丢弃 MTP speculative head

### 现象

- 原始 `models/Qwen3.6-27B` 有 15 个 `mtp.*` tensor；Baidu01 full SFT checkpoint-4000/8000 均没有这些
  tensor，但保存的 `text_config.mtp_num_hidden_layers` 仍为 1。
- 原始 artifact 为 27,781,427,952 个 BF16 元素、55.56GB；SFT checkpoint 为 27,356,728,560 个 BF16
  元素、54.71GB，差值恰好是 424,699,392 个 MTP 元素（约 0.85GB）。

### 根因

- Shaft 的 Qwen3.5/3.6 loader 复用 Transformers `AutoModelForImageTextToText`；当前 Transformers
  `Qwen3_5PreTrainedModel` 明确把 `^mtp.*` 配为 unexpected-on-load ignore，标准模型类不实例化 MTP。
- Trainer 只能保存运行时 model state，因此被 loader 忽略的 MTP 权重不会进入 full checkpoint；artifact
  validator 当前只区分 full/adapter，没有交叉验证 `mtp_num_hidden_layers` 与 `mtp.*` 权重集合。

### 影响范围

- 不影响标准 autoregressive 推理、SFT next-token loss、detection/reconstruction 质量或输出正确性；MTP 是
  speculative draft head，缺失只会使 vLLM/SGLang 无法对该 checkpoint 启用对应的 MTP speculative decode。
- 若仅把原始 MTP 权重复制到已 full-SFT 的 target model，speculative verification 仍保证 target 输出正确，
  但 draft/target 已失配，acceptance rate 和性能可能显著下降；不能把这种 artifact 标为 trained MTP。
- 这是 model/artifact/deployment capability 问题，不是 data、template、codec、metric 或现有评测误判。

### 修复方式（规划）

- 第一阶段先加入 MTP artifact capability：根据 config 与权重 index 严格区分 `absent / inherited / trained`；
  请求 MTP 部署时权重不完整必须 fail closed。无 MTP 的导出不得继续保留一个误导性的 enabled config。
- 第二阶段在 Qwen36 专用 loader/model adapter 中实现可训练 MTP module，严格加载官方 15 个键；通过现有
  `TrainingObjectivePolicy` 注册 `mtp_loss`，从 input/labels 推导 next-N 对齐并保持 prompt、padding、media
  token mask，不把模型专属逻辑写入通用 collator/trainer。
- full SFT 首版只支持 padded layout 和 full/MTP-only calibration；LoRA/QLoRA、varlen、OPD/RLHF 在有独立
  正确性门禁前 fail closed。checkpoint/export 必须保存 MTP tensor、状态和 provenance；部署由 vLLM/SGLang
  负责 speculative execution，Shaft 只提供 typed server config、capability preflight 与可复现实验记录。

### 回归测试（待实现）

- artifact 测试覆盖 config=1 但缺键、残缺键、shape/dtype 错误、完整 inherited/trained round-trip，以及
  `init_from_checkpoint`/resume/export 后 MTP tensor 与 provenance 一致。
- objective 测试覆盖 next-N shift、assistant-only labels、padding/media mask、全 masked batch、主 loss + MTP
  loss 系数组合，以及 MTP-only 时只允许 draft head 获得梯度。
- 真实 Qwen3.6 gate 覆盖单卡 forward/backward、ZeRO-3 save/resume/HF reload，并用 vLLM 对比 standard 与
  MTP 的 greedy 输出一致性、acceptance rate、TPOT 和并发吞吐；不能只以“服务能启动”作为验收。

### 后续防线

- 模型 config 声明的任何 auxiliary/draft capability 必须与 checkpoint tensor 集合一致；HF loader 的
  ignored/unexpected keys 不能静默穿过模型装配和 artifact validation。
- speculative decoding 是部署性能能力，不得与模型质量混为一谈；任何 inherited/frozen draft head 都必须
  与 joint-trained/calibrated MTP 分开标识和 benchmark。

## 2026-08-11：layout recognition 结果同步误用了 result_vlm 路径

### 现象

- 结果同步 subtask 默认下载和上传 `layout_recognition/result_vlm`，但评测仓库的正式真源实际是
  `layout_recognition/result`。
- 三个 v5.7 4M detection run 首次被写入错误远端目录；本地快照还混有该错误路径的历史残留，容易让路径
  看起来像既有合同。

### 根因

- 同步脚本和 README 把早期临时命名 `result_vlm` 固化为默认值，准备脚本、v5.3 enrichment 脚本又引用了
  同一错误路径。
- 首次核对只检查了错误目录中的预测 JSON schema，没有同时检查仓库正式 `result` 目录的 run 根合同，因而
  漏掉正式 run 需要 `method.json`、顶层 `methods.json` 和仓库评测器生成的 `score.json`。

### 影响范围

- 错误远端目录只包含本次误传的三个 run，没有覆盖正式历史结果；已整体删除。
- 三份 prediction 内容本身有效，但首次 upload commits 不属于正式结果。内部浮点 bbox summary 与正式整数
  JSON 的仓库评分存在轻微差异，正式展示必须以导出后 `score.json` 为准。
- 这是结果发布路径和评测产物合同问题，不是模型能力、data、codec 或 detection metric 实现错误。

### 修复方式

- 删除远端 `layout_recognition/result_vlm` 及本地同名目录；同步默认路径、README、converter 命名和 v5.3
  enrichment 目标统一改为 `layout_recognition/result`。
- 从 Hub 完整同步正式 `result` 后重建三个 4M run；每个 run 包含 `method.json`、175 个 prediction JSON 和
  同版 `layout_recognition/eval.py` 生成的 `score.json`，随后更新顶层 `methods.json`。

### 回归测试

- Hub 根目录反查确认 `layout_recognition/result_vlm` 不存在、`layout_recognition/result` 存在。
- 三个 run 均验证 175 个 prediction 文件与 `data/real_v1/gt` 文件名全集一致，parse-ok 为 100%，
  `method.json.name`、`score.json.method` 与目录名一致。
- 远端每个 run 抽查 `method.json`、`score.json` 和首/中/末 prediction 共 5 个文件，SHA-256 与本地一致；
  `methods.json` 已包含三个新方法。

### 后续防线

- 发布前必须从任务根目录开始核对正式路径和 run 根合同，不能仅凭局部 JSON schema 推断目标目录。
- 工作格式指标不能直接作为发布评分；坐标量化、label 映射等导出完成后必须运行仓库自身 evaluator。
- 同步脚本只允许一个默认真源路径；临时或历史路径必须显式覆写，不能再进入 README 推荐命令。

## 2026-08-11：prediction 同步越权生成并上传 method/score

### 现象

- 三个 v5.7 4M detection prediction run 在同步到 `layout_recognition/result` 时，同时由本地人工生成并上传了
  每个 run 的 `method.json` 和 `score.json`。
- 上一条“result_vlm 路径”事故记录错误地把这两个文件视为同步方必须补齐的 run 根合同；实际 method 注册
  和正式 score 由结果仓库的自动程序统一计算，prediction 提交方不应预先生成。

### 根因

- 混淆了“本地导出后自检”与“正式结果仓库自动评测”两个职责：本地 evaluator 的输出被错误提升成发布
  真源，并随 prediction 一起上传。
- 上传验收只验证文件完整性和 SHA-256，没有先确认自动评测系统对 method/score 的所有权边界。

### 影响范围

- 影响三个 run 的 6 个非 prediction 文件；每个 run 的 175 份 prediction JSON 内容未受影响。
- 手工 score 可用于本地诊断，但不是自动程序产生的正式评分，不能在训练报告或交接文档中以“正式 score”
  引用。这是发布流程/评测真源错误，不是模型能力、codec 或 detection metric 本身错误。

### 修复方式

- 从本地同步目录和 Hugging Face 远端三个 run 中删除全部 `method.json` / `score.json`，只保留
  `real_v1/pred/*.json`。远端清理 commit 为
  `7f64de72b06f19e8ff1b84d60600712926c06e08`。
- 修订 Qwen3.6-27B v5.7 训练报告和 `notes/agents/vlm_training_knowledge.md`：撤销手工 score 的正式口径，
  checkpoint 选择只引用明确标注为内部 sweep 的指标，并声明正式 method/score 等待自动程序生成。

### 回归测试

- Hub 远端反查三个 run：prediction 均为 175 份，`method.json` / `score.json` 均为 0 份。
- 本地同步目录得到同样结果；未删除、改写或重新量化任何 prediction JSON。
- 重新构建训练报告，artifact validation/package 通过，并确认 artifact/HTML 不再包含手工正式 score 来源或
  `0.814782` / `0.792395` 等已撤销发布数值。

### 后续防线

- 向 `layout_recognition/result/<run>` 同步时，默认只允许 prediction payload；除非自动评测系统合同明确授权，
  禁止代理生成、修改或上传每个 run 的 `method.json` / `score.json`。
- 本地 evaluator 输出必须保留在临时评测目录并标注 `internal`，不得复制进正式 result run；训练报告必须区分
  internal sweep 与 automatic official score。
- 发布验收除了 schema 和文件数，还必须检查产物所有权：哪些文件由提交方提供、哪些由自动程序派生。

## 2026-08-11：OPD 生产门禁暴露多模态 rollout、tail logits 与异步计时合同错误

### 现象

- 真实 Qwen3VL + TRL vLLM server 首次生成时，vLLM 返回 145-token prompt，本地 processor scoring prompt
  只有 82 token；64 个本地 `<image_pad>` 被服务端再次展开成 127 个。
- 修复 rollout 后，Qwen `logits_to_keep` 只返回 3 个 tail logits，但 teacher request 强制要求
  `causal_position_mask` 等于完整 84-token 输入的 shifted width，实际 completion mask 只有 2 个位置。
- FSDP 门禁中，本地 HF rollout 用 `torch.inference_mode()` 生成的 tensor 随后进入可训练 score graph，触发
  inference tensor 与 autograd/FSDP 不兼容。
- review 发现 OPD phase 只用 `perf_counter()`；CUDA kernel 异步提交时，这些 wall 字段不能代表设备执行时间。

### 根因

- vLLM 的预分词多模态接口要求“一张媒体对应一个未展开占位符 + 原始媒体”，而 collator 只保存了 processor
  已展开的 scoring IDs，把 generation 表示与 scoring 表示错误合并为一个状态源。
- teacher request 把“完整 input sequence”误当成“模型实际返回的 logit span”。Qwen 模型 policy 已声明
  tail-logits contract，但 request validator 没有消费该抽象。
- `inference_mode` 产生的 tensor 具有比 `no_grad` 更强的不可训练属性；rollout 结果虽然不求导，仍会被拼接成
  student forward 输入。
- CPU wall timer 只能度量主机可见延迟；没有 CUDA events 就无法分辨 kernel dispatch 与设备执行。

### 影响范围

- 第一项会使 vLLM rollout 实际条件序列与 student/teacher scoring 序列不一致，属于训练语义错误，不是模型
  能力、data、eval、codec 或 metric 误判。
- 第二项只在声明 tail logits 的模型族出现；完整 logits 的 smoke model 无法暴露，因而 CPU smoke 通过不代表
  Qwen score request 正确。
- 第三项影响 FSDP/部分 autograd 路径；普通单卡/DDP 可能延迟暴露。
- 第四项不改变训练结果，但会低估 GPU phase 时间并形成错误效率结论；wall RPC 延迟和 device time 必须分开。

### 修复方式

- `OPDRolloutRequest` 现在同时保存 tokenizer-only 的 `generation_prompt_token_ids` 与 processor-expanded 的
  `prompt_token_ids`。vLLM 只接收前者，返回的展开序列必须逐 token 等于后者；原始多图顺序保持不变。
- `ShaftRolloutScoringPlan` 继续作为 logit span 真源；teacher mask 只要求位于 shifted input 范围内，provider
  再用真实返回 logits shape 做强校验。objective-specific distribution 校验收敛到 `OPDObjectiveRegistry`。
- local HF rollout 改用 `torch.no_grad()`，保留无梯度语义但返回普通 tensor。
- OPD telemetry 保留 wall phase，并新增 deferred CUDA event phase/optimizer-frame 字段；events 只在
  checkpoint/finalize 批量同步。DDP throughput 按每个 step 的最慢 rank critical path 聚合。

### 回归测试

- 直接协议探针证明：未展开 19-token prompt 经 vLLM 展开后与本地 82-token processor prompt完全相等；已展开
  prompt 直接发送会错误变成 145 token。
- `tests/test_opd.py` 覆盖双 prompt 发送/漂移拒绝、tail-logit request、非 inference tensor、deferred CUDA
  event、最慢 rank critical-path 聚合、full/chunk/top-k-tail oracle、远端 teacher
  body/identity/idempotency，以及标准库 live HTTP server 上的 urllib+safetensors 往返。
- 真实 Qwen3VL-2B LoRA student + 4B local teacher + 独立 TRL vLLM server 在 CUDA 0/1 完成一步
  weight-sync、单图 rollout、student/teacher score、backward 和 telemetry。
- 两卡 tiny FSDP 与 DeepSpeed ZeRO-3 均完成 fresh step-2、checkpoint-1 resume、最终权重一致与每 rank
  telemetry `[1,2]`；未操作 `gpu-holder`。
- 最终 focused OPD/config、两进程 CPU DDP 与仓库默认 `pytest -q` 全部通过。

### 后续防线

- 多模态外部生成必须显式区分“backend preprocessing 输入”和“本地 scoring 真源”；任何 backend 返回的
  canonical prompt 都要与 scoring 真源比对，禁止 decode/re-tokenize 修补漂移。
- request/mask 必须对齐模型 policy 声明的 output span，不能从 input length 推断所有模型都返回完整 logits。
- 会重新进入可训练 forward 的无梯度 tensor 使用 `no_grad`；`inference_mode` 只用于结果永不进入训练图的边界。
- GPU 效率结论必须同时标明 wall 与 device timing；没有 device event 或真实长期 A/B 时，不得根据 dispatch
  时间宣称 kernel 优化收益。

## 2026-08-11：四卡门禁暴露 external teacher 路由、vLLM 生命周期与 topology 验收错误

### 现象

- 真实 `POST /v1/score` 返回 HTTP 422，并声称缺少 query 参数 `request`；service core 和假 transport 测试此前
  均通过。
- 三 rank DDP + 独立 vLLM server 首次同步 LoRA student 时，非主 rank 报
  `DistributedDataParallel has no attribute merge_adapter`；主 rank 已进入 vLLM communicator，最终与其它
  rank 一起等待到 NCCL timeout。
- 四卡 Qwen3VL-4B greedy-varlen fresh/resume 均成功，但 checkpoint telemetry 恢复验证器固定启动两 rank，
  用错误 topology 验证四 rank snapshot 后产生假失败。

### 根因

- `Request` 为 `create_opd_teacher_app()` 内的可选导入，模块又启用了 postponed annotations；FastAPI 注册路由
  时无法解析局部类型，因而把 `request` 误判为 query 参数。
- Shaft 把 TRL `VLLMGeneration` 延迟到首个 `compute_loss` 才创建，此时 Trainer 已把 student 包成 DDP。
  TRL 上游则在 Trainer 初始化阶段以未包装 PEFT model 创建 generation；延迟生命周期破坏了该合同。
- integration helper 把历史两卡门禁的 `--nproc_per_node=2` 固化为常量，没有从 checkpoint efficiency
  transaction 的 `world_size` 真源恢复拓扑。

### 影响范围

- external HTTP teacher 的真实 FastAPI 服务无法用于训练；这是 service adapter 错误，不是 teacher loss、
  codec、data 或模型能力问题。
- DDP + PEFT + vLLM rollout 无法完成首次权重同步，且 rank 间副作用阶段不一致会把直接异常放大成长超时。
  单 rank vLLM gate 无法暴露该问题。
- packing 训练结果与 checkpoint 本身有效；第三项只影响多于两 rank 的验收工具，但会阻止四卡证据成立。

### 修复方式

- FastAPI endpoint 在拿到真实 `Request` 类后显式绑定函数 annotation，再注册 route；仍保持 FastAPI 为可选
  serve 依赖，不提升成模块级硬依赖。
- `OPDRolloutBackend` 增加统一 `prepare(model, accelerator, processing_class)` 生命周期。vLLM backend 在
  `ShaftOPDTrainer.__init__`、分布式 wrapping 前创建 TRL generation 并永久绑定 canonical student；运行期
  request 只携带 wrapped model 做 score，不再作为 generation 权重源；每次同步前还要求该 wrapper 解包后
  与 prepared student 对象同一，防止静默同步错误模型。
- Qwen integration runner 和 packing gate 支持显式 world size；telemetry restore helper 从 committed
  transaction 读取 world size，并以同 topology 启动 CPU Gloo validator。
- integration probe 只复制 student 的全部 trainable 参数来证明真实更新；冻结 teacher 的所有参数用
  `(object identity, PyTorch mutation version)` 前后对比，完整覆盖替换和原地写入，同时避免每 rank 复制并
  逐元素比较 2B/4B 冻结权重。

### 回归测试

- FastAPI `TestClient` 真实 body injection、vLLM pre-wrap model-view 单测与未准备 fail-fast 合同通过。
- CUDA0–3 四 rank DDP + CPU HTTP teacher 完成 `topk_tail(top_k=3)`、`token_chunk_size=1`、GA、每 rank
  telemetry、student 更新/teacher 不加载，以及 fresh/checkpoint-1→step-2 resume；rollout、teacher request、
  RNG 离散轨迹一致，浮点状态在四 rank reduction 的 `2e-10` 严格绝对误差内一致。
- CUDA0–3 分别完成四 rank FSDP 与四 rank DeepSpeed ZeRO-3 tiny fresh/resume/export，最终权重完全一致，
  telemetry 每 rank steps 为 `[1,2]`。
- CUDA0–2 三 rank DDP Qwen3VL-2B LoRA student + 每 rank 4B local teacher，CUDA3 独立 TRL vLLM server，
  完成 weight-sync、单图 rollout、score/backward、非零 update 和 wall/device telemetry。
- CUDA0–3 Qwen3VL-4B greedy-varlen packing 完成 fresh/resume、四 rank telemetry restore、全 checkpoint 状态
  等价、PEFT export/reload 与 adapter 再训练。所有进程均由门禁自身回收，未操作 `gpu-holder`。

### 后续防线

- 依赖 distributed wrapper 状态的外部组件必须有显式 prepare/teardown 生命周期；禁止在首个 batch 中隐式构造
  并猜测 wrapper 类型。新增 backend 应先证明 canonical model view 的所有权。
- HTTP service 必须至少有一条真实框架路由/body 注入测试；service-core 单测不能替代 ASGI 路由解析。
- 任何 checkpoint validator 必须从 artifact topology 真源启动匹配的 world size，不允许在通用 helper 中固化
  rank 数；world-size elastic resume 仍是独立、未支持的能力。

## 2026-08-11：文档真源漂移与专题文档膨胀收口

### 现象

- README、配置参考和扩展指南仍引用已经删除或未被 Git 跟踪的训练配置、测试文件和 TODO。
- OPD、开发流程、CI handover 与多个 TODO 分散维护相同的能力矩阵；已完成实现仍留在 TODO，本地 Banana
  实验配方和机器磁盘状态进入了框架参考文档。
- Qwen3.6、planned batching 与 external teacher/vLLM 的部分“当前状态”互相矛盾。

### 根因

- feature 完成后不断新增专题文档，却没有指定每类事实的唯一所有者，也没有在实现演进时退休设计稿。
- 框架规范、实验记录、历史交接和待办使用了相同的“当前状态”口吻，导致日期化证据被误当成长期接口。

### 影响范围

- 不影响训练、推理、数据或 checkpoint；影响工程师选择命令、理解能力边界和规划后续开发。
- 未跟踪配置与不存在测试路径会让 clean clone 用户直接得到无效操作指引。

### 修复方式

- `docs/README.md` 收敛为唯一导航入口；当前行为由 architecture/module/config/testing 等正式文档负责，
  `development_log.md` 只保留历史事实。
- 将 OPD execution contract 并入 `architecture.md`，将开发/收口流程并入 `extension_guide.md`；删除重复的
  OPD 专题、开发流程和过期 CI handover。
- 将旧通用 TODO 与 PPO TODO 合并为一份日期化当前 TODO；完成项回到正式文档，实验过程留在开发日志。
- 删除框架参考中的本地 Banana 配方、机器容量快照和过期 provider 限制，示例只引用 Git 跟踪配置或明确的
  `/path/to/...` 占位符。

### 回归测试

- `git diff --check` 通过。
- 对 README、AGENTS 与全部当前文档执行本地一致性检查：Markdown 相对链接均存在，正式配置引用均被 Git
  跟踪，`docs/README.md` 无孤儿文档，已删除文档无活动引用。

### 后续防线

- 不再为单个 feature 默认新建架构文档；先并入现有总架构、模块参考或配置参考，只有需要独立长期维护且
  无法清晰归属时才新增文件。
- 当前 TODO 永远只保留一份；完成项立即删除并按需要写入正式文档或开发日志。
- 框架参考不得登记未跟踪的业务配置、机器路径、磁盘余量或一次性实验结论。

## 2026-08-12：real_v1 detection 金标更新与全量结果重算

### 现象

- Hugging Face `EditFigure/evaluation_and_results` 中 `layout_recognition/data/real_v1/gt` 更新；旧本地
  GT 共 7,257 个实例，新版为 7,086 个实例，175 个 JSON 均发生变化。
- 旧报告和临时排行榜中的 detection 指标基于旧 GT，不能与新版指标直接混用。

### 根因

- 评测金标本身发生版本更新，不是模型输出、codec 或 metric 实现变化；实例总量与类别分布均有调整。

### 影响范围

- 影响所有以 `data/real_v1/gt` 为真值的历史 detection precision、recall、F1、mIoU 和模型排序。
- 不影响已经落盘的模型预测，也不代表模型能力在本轮发生变化。

### 修复方式

- 从数据集 revision `7f64de72b06f19e8ff1b84d60600712926c06e08` 拉取 175 份最新 GT，校验文件名集合与
  JSON 有效性后，使用精确镜像方式覆盖本地 `data/real_v1/gt`。
- 使用同 revision 的 `layout_recognition/eval.py`，对结果仓库现有 17 组完整预测和本地 4B v5.7
  checkpoint-12000 4M 预测统一按 class-aware、greedy one-to-one、IoU >= 0.5 重算 detection 指标；结果写入
  `temp/layout_recognition_detection_leaderboard_gt_20260812/`，未写入预测仓库的 `method.json` 或
  `score.json`。

### 回归测试

- 本地 GT 与下载 staging 的 175 个文件逐字节一致，文件名无缺失/多余，全部可解析。
- 18 组预测均覆盖 175 张图，prediction parse success 为 100%，无 missing/extra prediction；每组
  overall GT 均为 7,086。
- 独立复核 17 个 rank 连续、per-run JSON 数量一致，并验证汇总 precision/recall/F1 代数关系。

### 后续防线

- real_v1 指标产物必须记录 GT dataset revision、GT 实例数和 evaluator SHA256；缺少任一项时不得与其他
  版本排行榜直接比较。
- 旧版 7,257-GT 报告仅作为历史记录；后续部署选型与跨模型排名统一使用 7,086-GT 新版结果，或显式重算。

## 2026-08-13：v5.7 训练集按 ID 隔离未阻止测试图别名泄漏

### 现象

- `data/raw/splits/vlm.test.json` 的 175 个测试 ID 与 v5.7 五个训练源的 `source_sample_id` /
  `source_json` / `source_image` 直接交集为 0，但图片内容审计仍找到 12 张测试图的改名副本。
- 11 对图片 SHA256 完全一致；`prod_034263.png` 与测试图 `00360.jpg` 是同尺寸重编码副本，
  pHash 距离为 0，平均像素绝对差约 1.03。
- 这 12 个真实训练源共派生 72 条 v5.7 SFT：`grounding_layout=29`、
  `line_context_points=16`、`image_context_reconstruction=27`。合成 shape/line reconstruction 不受影响。

### 根因

- grounding split 和真实 reconstruction builder 只按 test manifest 的 `id` / 文件 stem 排除，没有将测试图的
  精确哈希、感知哈希或已知别名映射纳入 split 合同。
- 新一批人工数据将同一位图以 `prod_*` 重命名，因此绕过了基于 `pic_*` / 数字 ID 的排除。

### 影响范围

- v5.7 训练数据并非与 175 图测试集完全隔离；所有使用该数据训练的 checkpoint 在 real_v1 评估上都存在
  小规模泄漏风险，相关指标不能当作严格无泄漏结果。
- 使用 2026-08-12 刷新的 7,086-instance GT 隔离这 12 图后，4B ckpt14000 的 163 图 detection F1 为
  `0.805634`，原 175 图 F1 `0.809527` 相对高 `0.003893`；另外三个 v5.7 checkpoint 的观测偏移介于
  `+0.000174` 与 `+0.007047` F1。12 图主要抬高 recall；4B ckpt14000 的 matched-box mIoU 在移除后反而
  从 `0.921571` 升至 `0.922373`。
- 这是 data split 语义问题，不是模型能力、eval codec 或 metric 实现问题。

### 修复方式

- 本轮只进行只读审计，未删除 raw 或派生数据。修复时应将 12 个泄漏别名按源图组从真实 train split 排除，
  重建受影响的 grounding、line points 和 image context 数据，不在最终 SFT JSONL 上就地打补丁。

### 回归测试

- 审计 v5.7 实际 catalog 的 816,842 条 SFT，所有行均有可追溯来源；直接 ID/路径比对命中 0，内容比对命中
  12 个真实源组和 72 条派生样本。
- 对 20,060 张真实 train 源图与 175 张 test 图执行 pHash 32x32 DCT 全量比对；Hamming `<=6`
  仅命中上述 12 对，无解码错误。唯一距离 8 的候选经人工核验为不同图，是稀疏白底版式的假阳性。
- V9 合成源 100,500 张图中，与 test 图共享字节大小的 12 个候选均无 SHA256 匹配。
- 按同一 evaluator 分别微平均 175 图、干净 163 图、泄漏 12 图，并对四个 v5.7 结果交叉复算；4B
  ckpt14000 的 12 图 F1 为 `0.853748`，11 张泄漏 PPT F1 为 `0.847953`，剩余 64 张干净 PPT 为
  `0.793594`。该切片差异包含样本构成效应，不能代替清洗后重新训练的因果 A/B。

### 后续防线

- test split 必须是“内容组”而非“文件名集”：构建真实 train split 时先按 SHA256 排除完全副本，再以 pHash
  Hamming `<=6` 产生近重复候选并对边界样本人工复核。
- grounding/reconstruction builder 应共享一个版本化的 excluded source-group manifest，而不是各自仅读 ID；
  发布派生数据前必须断言源图组与 eval 集交集为 0。

## 2026-08-13：禁止将 GT-conditioned reconstruction 回填为正式两阶段预测

### 现象

- 向 `layout_recognition/result` 的 detection 结果补充 shape/line 子属性时，27B checkpoint-4000/8000 的
  reconstruction 产物可按 `image_id + detection_index + label` 与各自 4M detection 一一对应。
- 4B checkpoint-14000 现有 reconstruction 却来自 GT 派生 crop 与任务噪声：shape/line 请求数为
  `2494/2051`，其正式 4M detection 只有 `2099/2002`，没有 detection index 合同。
- 若仅按 bbox IoU 猜配后写回，JSON 仍可被 evaluator 正常解析，因而容易把 GT-conditioned 上限结果误当成
  正式业务两阶段属性结果。

### 根因

- 旧的 reconstruction 离线评测协议与新的端到端协议共用了近似的结果 JSON 外观，但没有把 proposal 真源
  作为结果安装门禁。
- 属性 evaluator 只消费 bbox 匹配后的 parameters，不知道 reconstruction crop 是来自 GT 还是 detector，
  因此无法在评分阶段自动识别这种数据泄漏。

### 影响范围

- 影响任何需要把独立 reconstruction 产物合并进 detection `pred/*.json` 的正式提交。
- 若错误合并，detection 指标不会变化，但 shape/line 属性与几何分数可能被隐性抬高；这是 eval/data 协议误判，
  不是模型真实端到端能力提升。

### 修复方式

- 新增临时可复现合并器 `temp/merge_detection_reconstruction_attributes.py`：只接受带精确
  `detection_index` 的记录，并逐项核对 image size、label 与 reconstruction `proposal_bbox_full` 是否等于
  目标 detection bbox；禁止 GT-crop 或 IoU 猜配。
- crop 0..999 几何统一转换为原图整数像素；line 的 prompt `fill/border` 输出转换为当前 real_v1 GT/evaluator
  使用的 `fill_color/has_border/border_style/border_color`。合同检查仍记录语义错误；用户确认后，对 JSON 和
  evaluator 可解析的输出保留原始参数，交给自动评测按错误计分，只对无法建立结构化参数的输出 fail closed。
- 本轮只安装并上传通过门禁的 27B checkpoint-4000/8000；4B checkpoint-14000 保持 detection-only，等待用
  自身 4M detection proposal 重跑 reconstruction。

### 回归测试

- 两个 27B run 均保持 175 个文件，元素 type/bbox/顺序与原 detection 逐项一致；所有全局几何均在图像范围内。
- checkpoint-4000/8000 分别消费 4,105/4,099 条 reconstruction，未缺失、未多用。
- 其中 checkpoint-4000 有 24/4,105（0.58%）条、checkpoint-8000 有 52/4,099（1.27%）条语义
  合同异常，全部保留后 evaluator prediction parse rate 仍为 100%。
- 上传后分别下载远端含语义异常项的 `00127.json` 并与本地比较 SHA256，一致；run 中没有人工生成的
  `method.json/score.json`。

### 后续防线

- 正式 attribute submission 必须记录 `proposal_source`、pixel budget、checkpoint identity 和
  `gt_read=false`；缺少精确 detection index 时禁止安装。
- GT-conditioned reconstruction 只能标为上限/诊断协议并保留在独立评测目录，不得通过 bbox matching 转换成
  detection-driven 结果。
- 长期应把 proposal provenance、坐标转换和 gt-standard 安装合同收口为正式 eval/export 模块与单测，避免
  继续依赖临时脚本。

## 2026-08-14：长尾推理超时与非法枚举使批量评测误报不完整

### 现象

- Qwen3.6-27B QLoRA 在 real_v1 的 1M/2M/4M detection sweep 中，极密集样本 `00345`、`pic_739`
  在服务保持健康、generation token 持续增长时仍被 900 秒 HTTP timeout 中断，导致 174/175 的中间摘要。
- detector-derived reconstruction 的 checkpoint-8000 记录 `00190__det_0017_line` 已成功返回 JSON，但模型把
  一个枚举字段输出成 list；`_validate_line()` 直接执行集合成员判断，抛出
  `TypeError("unhashable type: 'list'")`，使整条预测被当成请求错误并得到 3995/3996。

### 根因

- 请求超时是固定常量，没有与 27B、8k max output 和密集样本长尾解耦；批量运行也没有在失败后只恢复缺失
  样本的完整合同。
- GT-standard/重建合同校验器假设枚举一定是可哈希标量。校验器本应接收任意模型 JSON 并返回字段错误，却在
  非法类型上自身崩溃。这是 codec/eval robustness 问题，不是该条预测未生成。

### 影响范围

- 固定 900 秒会把仍在正常生成的 detection/reconstruction 长尾误判为服务失败，造成重复加载模型、重复计算
  和不完整汇总；不能通过缩短 8k output 或伪造空预测来规避。
- 非法枚举触发异常时，该实例无法进入 review，也会把“contract-invalid 模型输出”错误升级为“请求失败”，
  污染格式健康统计。

### 修复方式

- 临时 27B sweep runner 支持 2,400 秒请求上限、已有 raw/pred/pixel/viz 的逐样本恢复，以及预算子集并行；
  完成判定仍要求 175/175、errors 为空、内部与官方指标逐项一致。
- `scripts/tasks/prepare_gt_standard_v5_7.py::_validate_line()` 对 `line_type`、`line_style`、`dash_style`、
  `begin_arrow`、`end_arrow`、`corner_style` 先做字符串类型检查，再做集合成员判断。非法 list 现在生成明确的
  contract issue，预测原样保留给 review。
- checkpoint-8000 只补跑缺失实例后达到 3996/3996、errors 为空；该非法输出计入 contract-invalid，不再丢样本。

### 回归测试

- `uv run pytest -q tests/test_prepare_gt_standard_v5_7.py`：2 tests passed；新增用例同时把六个枚举字段设为
  list，断言校验器返回对应 issues 而不抛异常。
- `uv run ruff check scripts/tasks/prepare_gt_standard_v5_7.py tests/test_prepare_gt_standard_v5_7.py` 通过。
- 四个 QLoRA checkpoint × 1M/2M/4M 共 12 组均有 175 raw/pred/pixel/viz；7,086 GT 口径下内部/官方
  P/R/F1/mIoU 全部一致。两个 detector-derived reconstruction 分别为 3996/3996 与 3942/3942、errors 为空。

### 后续防线

- VLM batch eval 的 HTTP timeout 必须是显式协议字段，并大于服务端允许的最坏 max-output 时长；长尾失败应按
  sample id 恢复，不能删除已完成缓存。
- 所有模型输出校验器都必须是 total function：任意合法 JSON 类型只能返回 contract issues，不能因不可哈希、
  类型比较或字段缺失抛异常。新增枚举字段时应覆盖 list/dict/null/bool 等反例测试。
- 汇总层只能消费完整摘要：样本数、errors、pixel budget、finish reason、官方/内部指标与 proposal provenance
  均应纳入门禁。

## 2026-08-14：PEFT 角色分组使结构组学习率静默失效

### 现象

- LoRA/DoRA/QLoRA 运行时，视觉参数如
  `base_model.model.model.visual.blocks.*.lora_A.default.weight` 被 optimizer 优先归入
  `lora_params`，而不是 `vision_tower`。
- 因此配置 `train.param_group_lrs.vision_tower` 可以通过 schema，却不会命中视觉 adapter 参数；启动日志
  只显示 PEFT 角色组时也很难直接发现结构 LR 未生效。

### 根因

- 旧 optimizer 同时维护“模型结构组”和“PEFT 参数角色组”，并让后者拥有更高优先级；分组逻辑还依赖
  `resolved finetune plan` 判断 mode。这把模型装配/保存角色错误地提升成了学习率语义。
- 运行时参数名只做了宽松 wrapper 删除，没有形成覆盖 PEFT saved key、DoRA、`modules_to_save` 与 FSDP
  delayed wrap 的可审计规范化合同。
- 这是训练 optimizer 语义错误，不是模型能力、数据或 eval 指标误判。

### 影响范围

- 影响所有曾在 adapter 模式下期望用 `language_model / vision_tower / aligner / generator` 设置差分 LR 的
  训练。使用单一全局 LR 的 adapter run 数值语义不受组选择影响。
- `lora_params/modules_to_save` LR key、resolved optimizer fingerprint 和 optimizer state group layout 均属
  breaking migration；旧 checkpoint 不允许 exact resume。
- 旧 full/PEFT 权重仍可 inference，也可显式 `init_from_checkpoint`，但必须创建新的 optimizer/scheduler，
  不能把 init 偷换成 resume。

### 修复方式

- `ModelModuleGroups` 统一发布四个结构组名称和最长边界前缀解析；config、freeze、optimizer 不再各自维护
  字符串集合。
- optimizer 只按 `(module_group, decay)` 分组。finetune mode 只通过 `requires_grad` 影响参数集合，
  optimizer 调用链删除 `finetune_plan`。
- canonicalizer 只处理确认过的 `_fsdp_wrapped_module`、PEFT `base_model.model`、LoRA/DoRA adapter namespace
  与 `modules_to_save` wrapper，保留真实层级和 PEFT tensor role，并保证 deterministic/idempotent。
- 正式模型的 trainable 参数必须全部归属；无结构 metadata 时只有全局 LR 可使用；显式配置未命中也会在
  optimizer 创建前报错。JSON、启动日志和 resume fingerprint 全部由同一个 resolved plan 派生。
- training resume contract 升级到 v3，resolved optimizer plan 升级到 v2，旧 contract 在加载 optimizer state
  前拒绝。

### 回归测试

- 配置测试拒绝 `lora_params/modules_to_save` LR key，并覆盖四个结构组的大小写归一化。
- optimizer 测试覆盖 full/LoRA/DoRA/QLoRA、saved adapter key、fused target parameter、
  `modules_to_save`、最长 aligner 前缀、coverage/unconsumed 错误、decay 与 raw/canonical summary。
- FSDP wrapper 测试证明 wrap 前后 canonical plan fingerprint 一致；cosine warmup 全阶段保持视觉/语言 LR
  比例 0.3。Qwen3.6-27B metadata 与真实权重 key 做只读审计，未解析 adapter trainable key 必须为 0。
- 本机 1,513 项全量快速回归通过；最终新增的 summary JSON 用例随 optimizer 40 项 focused 回归通过，
  当前 1,514 项均有通过证据。distributed suite 53 项为 50 passed、3 个可选环境门禁 skip。真实
  Qwen3.6-27B meta 模型注入 r16 LoRA 后，1,212 个 trainable tensor 全部归属；language/vision/aligner
  参数量为 116,727,808 / 7,699,968 / 303,104，LR 为 1e-5 / 3e-6 / 1e-5。
- CUDA 0/1 的 1-step FSDP canary 未进入模型加载：gpu-holder 占用后只剩约 0.01/0.62 GiB，NCCL preflight
  无法建立；保持最小 CUDA context 等待 180 秒后仍未自动释放，按硬超时退出。未操作 holder、未执行训练
  step，因此 GPU runtime/checkpoint 保存验收仍标记为资源阻塞，不能宣称通过。

### 后续防线

- 新模型族必须在 `ModelModuleGroups` 声明完整 trainable 路径；不能通过 substring 猜组或添加 default
  fallback。
- 新增 PEFT/FSDP wrapper 形式前必须先取得真实 runtime/saved key，并补 canonicalization 反例与幂等测试。
- 任何显式 LR group 都必须在 canary summary 中非空；长训练启动前审计三项：group、LR、参数量。

## 2026-08-14：框架能力口径审计与唯一总 TODO 收口

### 现象

- README 将 SFT、DPO、PPO、GRPO 并列在“当前能力”和训练命令中，但真实成熟度并不相同：SFT 是生产主线，
  DPO/GRPO 尚缺真实 Qwen release gate，PPO 只有 debug smoke。
- DPO/GRPO 的 FSDP+PEFT exact resume 只在 SFT 完成恢复验收；本轮用现有 DPO/GRPO 配置做只读预检时，
  `LoRA + FSDP + full_state_dict + periodic checkpoint` 仍都被配置层接受。
- required CI 只覆盖 CPU framework 与 tiny/fake smoke，不能代表 distributed、GPU、真实模型或长程收敛；
  推理 API 也只有单样本同步合同，没有原生 batch/streaming/async 服务层。
- 当前 TODO 虽已合并为一份，但文件名仍绑定日期，README、配置注释和参考文档需要共同维护具体日期路径。

### 根因

- 文档混用了“代码可装配”“CPU contract 通过”“真实发布权重 gate 通过”和“生产验收完成”四种证据层级。
- 通用 FSDP+PEFT 校验表达的是 SFT 已验证约束，没有为 DPO/GRPO 的未验证恢复组合建立算法级 fail-closed。
- TODO 使用日期化文件名，把稳定导航入口和一次审计日期耦合；正式文档中也散落少量行动性待办措辞。

### 影响范围

- 使用者可能把 RL 命令可见、配置可加载或 required CI 绿灯误解为生产支持，并选择未验收的 checkpoint/
  resume 组合。
- PPO 默认示例保持安全关闭 debug 开关，但若仍作为普通训练命令展示，会形成“官方命令存在却不是可用生产
  入口”的错误预期。
- 这是能力声明、测试证据与文档真源问题；不是模型能力退化，也不是 eval、codec、metric 或 data 标准误判。
- 本轮不修改 `src/shaft`、配置语义、训练/推理行为或测试实现。

### 修复方式

- 将日期化 TODO 收敛为唯一稳定入口 `docs/TODO.md`；删除旧文件，不保留 redirect 或算法级平行 TODO。
- 总 TODO 统一记录 RL fail-closed、release gate、OPD 容量、batching/并行、推理/eval 与生态扩展事项；正式
  文档只描述当前能力和明确拒绝边界。
- README 将 SFT 标为生产主线、DPO/GRPO 标为实验能力、PPO 标为 debug-only，并从普通训练命令块移除
  PPO；OPD 保持专项能力口径。
- 架构、配置、模块和测试文档明确：FSDP+PEFT exact resume 当前只对 SFT 验收，DPO/GRPO 即使通过通用
  配置预检也不得使用；required CPU 绿灯不能外推为真实 GPU/模型生产门禁。
- 推理文档补充单样本、同步、图片必需和无原生 batch/streaming/async 服务层的当前合同。

### 回归测试

- 文档审计前当前工作树已通过 `1523` 项 framework 与 `30` 项 smoke；本轮只改文档和 PPO 配置首行注释，
  未重复执行训练测试。
- 补跑配置加载 focused 回归：`uv run --locked pytest -q tests/test_config_loader.py`，结果 `29 passed`。
- 对仓库 Markdown 相对链接执行存在性检查，并检查旧 TODO 路径、平行 TODO 文件、活动文档 TODO 引用和
  `git diff --check`。
- 只读配置探针保留为问题证据：DPO、GRPO 的 FSDP+PEFT periodic-checkpoint 组合当前均显示 `ACCEPTED`，
  因此总 TODO 的 P0 不能仅靠文档关闭。

### 后续防线

- 仓库当前待办永远只维护 `docs/TODO.md`；完成项立即删除，历史过程写开发日志，当前支持行为写正式参考。
- 对外使用“支持”一词时必须同时注明算法、模型、finetune mode、batching、distributed topology、checkpoint
  和证据级别；注册表存在、CLI 可见或配置可加载都不能单独作为支持声明。
- 未完成的 fail-closed 属于 P0；文档可以提前标明“不支持”，但不能把文档声明当成运行时防线。

## 2026-08-14：框架参考与当前数据任务文档解耦

### 现象

- `docs/data_v5_7.md` 被列入根 README 的重点文档和 `docs/` 当前真源，并被模块、配置、脚本文档引用。
- `docs/module_reference.md` 与 `docs/scripts.md` 还直接记录 Banana v5.7 的数据组成、标注规则、构建命令和
  完整性基线，使具体数据生产任务看起来像 Shaft 公共能力合同。

### 根因

- 文档按“当前工作重要性”组织，而不是按“框架公共接口”与“具体项目任务”组织。
- 离线 task 脚本属于仓库可复现工具，但其业务数据版本和运行基线不因此成为框架模块语义。

### 影响范围

- 框架使用者会误把 Banana v5.7 当成理解或使用 Shaft 的必读前置，也可能误认为框架 data 层绑定该任务的
  label、prompt、selection 和数据规模。
- 这是文档归属和模块边界漂移，不是模型能力问题，也不是 eval、codec、metric 或 data 处理结果误判；
  `src/shaft` 和运行行为不受影响。

### 修复方式

- 将任务说明从 `docs/data_v5_7.md` 迁到 `scripts/tasks/banana_v5_7.md`，明确其是当前 Banana v5.7 数据生产
  任务说明，不是框架能力文档。
- 从根 README、`docs/README.md`、模块参考和配置参考中移除 Banana v5.7 的重点入口与业务合同；配置参考的
  prompt sampling 示例改为通用 dataset。
- `docs/scripts.md` 只保留 task 脚本与框架的通用边界，具体构建命令和基线由 task 文档维护。

### 回归测试

- 检查 `README.md` 与 `docs/`，确认不再把 Banana v5.7 列为框架重点文档或模块能力。
- 检查仓库 Markdown 本地链接、旧路径引用与 `git diff --check`；本轮不需要训练或推理回归。

### 后续防线

- `docs/` 只维护 Shaft 架构、公共配置、模块接口、通用设计、测试和运行参考。
- 具体客户、数据版本、单次训练 bundle、业务 prompt 和完整性数字应与 `scripts/tasks/`、recipe 或外部任务
  目录共置；即使当前优先级很高，也不得进入框架重点文档列表。

## 2026-08-14：27B 周期 checkpoint 只保留模型态

### 现象

- 27B full fine-tune 每到 `checkpoint-4000/8000` 都同时保存模型权重、optimizer、scheduler、RNG 与
  DeepSpeed/FSDP native state。可恢复训练态远大于单份部署权重，两个阶段快照会快速耗尽训练盘。
- 实际需求只是在固定 step 留下可部署、可供后续 `init_from_checkpoint` 的权重，不需要从这些目录 exact
  resume；仅关闭 root `save_final_state` 不会改变 periodic checkpoint 的内容。

### 根因

- Shaft 配置没有暴露 HF `TrainingArguments.save_only_model`，训练层默认把所有 periodic save 都解释成
  resumable generation。
- Shaft 的 commit、input identity、plugin neutrality 与 resume resolver 也按“未来一定 exact resume”校验；
  如果只把 HF 参数透传为 true，DDP manifest 会继续错误要求 optimizer/RNG，分片后端还可能留下旧训练态或
  发布不完整 shard。
- 这是 checkpoint 存储与恢复语义问题，不是模型能力、data、eval、codec 或 metric 误判。

### 影响范围

- 新字段只影响 SFT 的 periodic `checkpoint-*`；默认 `save_only_model=false` 保持既有 exact-resume 行为。
- model-only checkpoint 可由标准 HF/PEFT loader 部署或作为 `init_from_checkpoint`，但不能恢复 optimizer、
  scheduler、scaler、RNG、数据 cursor 或 Trainer schedule。
- DPO/PPO/GRPO/OPD 未经各自保存门禁，不会因为共享字段存在而静默扩大支持面。

### 修复方式

- `train.save_only_model` 成为唯一用户配置真源，并直接映射 HF `TrainingArguments`；training config 内部只派生
  `disabled / model_only / resumable` 三态，不增加第二个可配置 checkpoint mode。
- model-only 保存复用 Trainer 的标准 HF/PEFT 权重路径和 Shaft 的 all-rank callback convergence，提交前拒绝
  optimizer、scheduler、scaler、每-rank RNG、DeepSpeed/FSDP native state 与 resumable marker 残留，再原子
  发布 `shaft_model_only_checkpoint.json`。marker 绑定 step、full/adapter 类型和必需模型文件尺寸，但不对
  51GiB 权重做第二遍全量 hash。
- direct checkpoint resume 明确拒绝 model-only marker；run-root resolver 跳过这类 generation。
  `save_only_model + resume_from_checkpoint` 在配置阶段直接失败，并提示改用 `init_from_checkpoint`。
- FSDP 只接受 `full_state_dict`；DeepSpeed ZeRO-3 只接受
  `stage3_gather_16bit_weights_on_model_save=true`；sharded backend 同时禁止
  `load_best_model_at_end=true`，确保每个发布目录本身就是完整 HF/PEFT artifact。
- 两份 Banana v5.7 27B full ZeRO-3 配置改为每 2000 step 仅保存模型态、最多四份，并关闭 root final model/
  state 重复保存，8000-step 训练保留 `checkpoint-2000/4000/6000/8000`。

### 回归测试

- 单进程真实 Trainer 执行一步保存，验证标准 HF 权重可由 `from_pretrained` 重载，且目录没有 optimizer、
  scheduler、scaler、RNG 或 native backend state；resume resolver 必须拒绝同一目录。
- 两进程 CPU DDP/torchrun 真实执行两步并发布两个快照，两个 rank 均成功收敛；`checkpoint-1/2` 都通过
  model-only validator，run-root resume 不会把它们识别为恢复点。
- 配置测试覆盖 bool normalize、SFT-only、禁止 resume/no-save、FSDP full-state、ZeRO-3 gather 与
  sharded best-model 互斥；checkpoint 单测覆盖 backend-native publication wrapper、残留清理、marker 损坏和
  input/plugin exact-resume 门禁解耦。
- 27B YAML 继续由 config loader 回归读取；没有目标 GPU 资源时，不能把 CPU DDP 与配置门禁表述成真实
  27B ZeRO-3 训练验收，正式长训练仍应先做目标机器保存 canary。
- 本轮完整回归结果为 framework `1538 passed`、smoke `30 passed`、distributed `51 passed / 3 skipped`；
  ruff、`git diff --check` 与 Python compileall 同步通过。

### 后续防线

- 新算法若要开放 model-only periodic save，必须显式声明能力并分别通过单卡、多 rank、full/adapter、
  deploy/init reload 与禁止 resume 的门禁；不能只复用 SFT profile。
- 新分片 backend 必须证明保存时能聚合完整标准模型 artifact，并保证目标目录不存在 native optimizer state，
  否则配置层保持 fail closed。
- model-only marker 是发布完成证明，不是自定义权重格式；模型目录始终保持 HF/PEFT 标准布局。

## 2026-08-17：27B checkpoint 沿用 Transformers 50GB 默认分片

### 现象

- Qwen3.6-27B 原始模型目录有十多片权重，但训练生成的 `checkpoint-2000/4000/6000` 各只有两片：第一片接近
  50GB，第二片约 4.9GB。
- checkpoint 可以由 HF 正常加载，但超大单文件不利于训练盘搬运、对象存储上传和部署侧并行读取。

### 根因

- Transformers 5.10 的 `PreTrainedModel.save_pretrained` 默认 `max_shard_size="50GB"`；`Trainer._save()` 调用
  该接口时没有传入分片上限，`TrainingArguments` 也没有同名字段。
- 训练后的 state dict 比基础 artifact 少 15 个 `mtp.*` tensor，这是当前 Qwen3.6 full SFT 明确拒绝训练 MTP
  speculative head 的既有策略；其余 key 与 index/实际 safetensors header 一致。两片不是权重丢失或保存
  不完整，而是保存端采用了更大的默认 shard 上限。

### 影响范围

- 问题影响所有通过 Trainer 保存的 full HF 权重，包括 periodic checkpoint 与 final `best`；模型权重数值和
  可加载性不受 shard 数量本身影响。
- PEFT adapter 继续由 PEFT 保存，训练态是否保留仍只由 `save_only_model` 决定；本问题不是模型能力、data、
  eval、codec 或 metric 误判。

### 修复方式

- 新增唯一配置真源 `train.max_shard_size`，默认 `4GB`；接受正整数 byte 或 HF 的
  `KB / MB / GB / TB` 字符串，并在配置阶段规范化、拒绝非法值。
- 共享 `ShaftModelSaveMixin` 在 HF 的标准 `save_model -> save_pretrained` 调用期间注入上限，不复制
  Transformers `_save()`；SFT、DPO、PPO、GRPO 与 OPD 的 Trainer 保存共用该入口。
- 两份 27B full ZeRO-3 recipe 显式写入 `max_shard_size: 4GB`。单个 tensor 超过上限时仍遵循 HF 规则独占
  一个较大 shard，不承诺保持基础模型的原始分片数量。

### 回归测试

- 配置回归覆盖默认值、大小写/空白规范化、byte 整数和非法/非正输入。
- 真实 tiny `PreTrainedModel` 执行一步 model-only periodic save，在 `1KB` 上限下生成 index 与多个
  safetensors shard，并由 `from_pretrained` 完整重载。
- SFT 与 DPO 的 DeepSpeed pipeline 装配测试验证规范化值进入共享 Trainer 保存层。
- 两进程 CPU DDP/torchrun 连续发布两份 model-only checkpoint；两份都按 `1KB` 测试上限生成多 shard、
  通过 commit validator，且 run-root resolver 不会把它们当作 resume 点。
- `pytest -q tests --suite framework`、`--suite smoke` 与 `--suite distributed` 全部通过；distributed 的 3 个
  skip 是既有可选环境门禁。focused 回归、ruff、compileall 与 `git diff --check` 同步通过。

### 后续防线

- 新 Trainer family 必须复用 `ShaftModelSaveMixin`，不得另写一套 HF `_save()` 或默默回退 50GB 默认值。
- checkpoint 验收同时检查 HF layout/index 可加载性与训练态策略；不能用 shard 数量单独判断权重是否完整。
- 27B 长训练前先做目标后端保存 canary，核对 shard 上限、总 tensor bytes、index/header key 和部署加载。

## 2026-08-17：示例 checkpoint 数据身份与历史 weak-label builder 仍依赖本地状态

### 现象

- `sft_4b.yaml`、`dpo_4b.yaml`、`grpo_4b.yaml` 都开启 periodic checkpoint，却没有声明
  `data.media_snapshot_id`；配置文件本身可加载，但训练主链建立 exact-resume input contract 时会因媒体身份
  不完整而拒绝。
- tracked `build_drawio_shape_from_weak_labels.py` 默认读取 ignored `subTasks/drawio_shape_weak/...` 路径；clean
  checkout 不包含该目录，不同开发机还可能静默消费不同本地 job。
- 一个未跟踪测试直接导入 `subTasks/prediction_results_sync`，并被临时登记进正式 task suite；这会让 clean
  checkout 的 suite 声明指向不存在的测试，或让本地测试结果依赖 ignored 实现。

### 根因

- README 与配置参考已经把 `media_snapshot_id` 写进示例，但三份可运行 YAML 没有同步，配置加载测试也只验证
  schema，没有锁住 checkpoint-enabled recipe 的完整数据身份。
- 历史 builder 从一次性 subtask 迁入 `scripts/tasks/` 后保留了原机器的默认输入路径，没有把本地 job 提升为
  显式 CLI 合同。
- 临时同步任务的测试边界没有遵守“正式测试不得依赖 `subTasks/`”的仓库规则。

### 影响范围

- 三份公开 4B 示例在真正启动 checkpoint-enabled 训练时可能晚于配置加载才失败；已有显式媒体快照的正式
  Banana recipes 不受影响。
- weak-label builder 的隐式路径影响离线派生数据可复现性，不修改 raw truth；未跟踪 prediction sync 测试不属于
  Shaft 正式能力，也不影响 eval、codec 或 metric 结论。

### 修复方式

- 三份 4B 示例统一声明 `media_snapshot_id: example-media-v1`，与 README/配置参考口径一致；配置回归同时要求
  recipe 开启 checkpoint 且媒体快照非空。
- `build_drawio_shape_from_weak_labels.py` 将 `--weak-job-dir` 改为必填参数，帮助文本明确输入文件；缺输入时在
  创建或清理输出目录之前失败。
- 删除依赖 ignored subtask 的未跟踪 prediction sync 测试及其 suite 登记；为 tracked weak-label builder 增加
  只依赖仓库脚本的 task 测试和最小文档入口。

### 回归测试

- 配置 focused 回归加载三份公开 YAML，并验证 checkpoint 与 immutable media snapshot 合同。
- task focused 回归验证缺少 `--weak-job-dir` 时 argparse 失败，指定空 job 且带 `--clean` 时旧输出保持不变。
- 提交前运行 task 与 framework suite，并执行 ruff、compileall 和 `git diff --check`；训练保存主链的 smoke 与
  distributed suite 已由紧邻的 checkpoint 分片提交完整覆盖，本轮未修改该运行时代码。

### 后续防线

- 新增或修改公开 checkpoint-enabled recipe 时，测试必须同时锁住完整训练数据身份，不能只证明 YAML 可解析。
- tracked task 脚本不得把 ignored/local 目录写成默认输入；本地 job 必须通过 CLI 显式注入，并在 destructive
  clean 前完成输入预检。
- 正式测试只能依赖 tracked 代码和 fixture；一次性 subtask 的测试与实现保持在 subtask 内，不登记仓库 suite。

## 2026-08-17：OPD 把 artifact 相等误当成 teacher/student 输入兼容

### 现象

- OPD local teacher 要求 teacher/student 的 model alias、tokenizer artifact fingerprint、processor 和原始
  template semantic fingerprint 全部相等；Qwen3.5/3.6/3.8 即使 token ID 与实际 scoring tensor ABI 等价，
  也会因产品 alias 或 chat template 文本不同被拒绝。
- HTTP teacher 另用一套 input fingerprint，并把 teacher artifact fingerprint 混入其中；local、HTTP 与
  resume 字段名字相似，但语义并不一致。

### 根因

- 旧的 `model_artifact_input_identity()` 复用了训练 artifact/semantic identity，把“从哪里加载、用哪个模板
  生成 prompt”与“teacher 是否能消费 student 已生成的同一批 tensor”合并成一个严格相等判断。
- vocab 只通过模型 config 的声明值间接检查，没有统一绑定实际 logits output head；teacher `forward` 的可接受
  字段也没有进入预检合同。

### 影响范围

- 影响 local 与 HTTP OPD teacher 的启动门禁和 exact-resume identity；会误拒绝输入 ABI 等价的跨 alias
  teacher/student，也可能把 provider 特有 identity 当作输入兼容状态。
- token ID、special token、processor schema 或 logits vocab 的真实不兼容仍必须拒绝；本问题不涉及模型能力，
  也不是 eval、codec、metric 或 data 误判。

### 修复方式

- 新增 OPD 域内唯一真源 `ShaftOPDInputABI`：完整 hash `token→ID` 映射并单独记录 special token ID，从实际
  output embedding/head 解析 logits vocabulary，绑定 processor ABI config、token role 和
  `ProcessorPolicy` scoring schema，并检查 model `forward` 的显式字段与 `**kwargs` 合同。
- 兼容判断只比较 student 实际会生成的 scoring tensor ABI；model alias 与原始 chat template 不进入 ABI。
  teacher 必须接收所有 required/optional student 字段，且不能要求 student 无法保证提供的额外字段。
- local artifact plan 与 HTTP provider 共用同一 builder/validator；HTTP identity 升级为 v2，直接发布序列化
  input ABI。兼容 fingerprint 不再包含 provider 或 teacher artifact；不可变 teacher artifact 继续由独立
  model-plan/artifact fingerprint 绑定。
- resume contract 字段收敛为 `teacher_student_input_abi_fingerprint`。旧 checkpoint 不具备新 ABI 证明，exact
  resume 时明确 fail closed，不伪造迁移。

### 回归测试

- 单测证明 qwen35vl/qwen38vl alias 与不同 template 在等价输入下放行；完整 token→ID、special token ID、
  实际 logits vocab、processor config/schema 和 teacher forward 字段任一漂移都分别明确拒绝。
- HTTP identity 序列化、live loopback 与 local/HTTP fingerprint 统一语义均有覆盖。
- OPD pipeline smoke 覆盖 full/LoRA 保存、sampled rollout exact resume、telemetry、外部 teacher 与
  model-only checkpoint 主链。
- framework、smoke、distributed suites 全部通过；distributed 仅保留既有环境型 skip。ruff、Python
  compileall 与 `git diff --check` 同步通过。

### 后续防线

- OPD compatibility 只能扩展 `src/shaft/opd/input_abi.py` 的实际输入合同；不得重新复用完整 model、tokenizer、
  template artifact equality，也不得在 provider 中平行维护第二套 fingerprint。
- 新 processor/model family 必须能发布完整 tokenizer vocabulary、可验证的输出头维度、processor input schema
  与可检查的 `forward` signature；任何一项无法证明时保持 fail closed。
- HTTP identity schema 若再次变化必须升级 protocol，并让旧 client/server 明确拒绝，禁止把缺字段解释为兼容。

## 2026-08-20：prompt 轮换缺少 task formulation 与原子监督真源

### 现象

- 训练侧原有 `prompt_sampling` 只能在一个 pool 内轮换 user prompt；同一 reconstruction 样本无法正式表达
  “只输出 A”“只输出 B”“同时输出 AB”三种 prompt/target 对。
- 若只轮换问题而继续使用固定 `target_text`，prompt 与监督范围可能不一致；业务脚本即使已经分别物化三份
  target，框架也缺少把这些离线 sources 绑定到同一 pool、逐行对齐并可复现选择的合同。
- “view” 一词容易与 multi-view representation learning 混淆，无法准确描述“问什么、监督什么”的变化。

### 根因

- 旧配置把能力建模为 `data.transforms.prompt_sampling`，运行时只拥有 prompt variant，没有 task
  formulation、离线 target source 绑定或 prompt/target 原子选择合同。
- 旧 sample context 只有 global `draw_id`；直接用它驱动 curriculum 会让一个 dataset 的阶段随着其它
  dataset mixing 权重变化，无法保持 dataset-local 的渐进训练语义。
- Prompt pool、transform、审计字段和训练配置分别维护部分状态，缺少一个同时绑定 pool、全部 formulation
  source snapshots、schedule、seed、prompt program 与选择算法的 execution fingerprint 真源。

### 影响范围

- 影响需要从同一 canonical row 派生不同监督范围的 SFT 任务，以及现有 prompt wording rotation 的配置、
  planning、分布式与 exact-resume 语义。
- 普通已经 materialized 的 HF/LLaMA-Factory 风格 SFT 数据不需要在线投影，应保持直接消费。
- 本问题属于 data/prompt 投影边界偏差，不是模型能力问题，也不是 eval、codec、metric 或 data label 标准误判。

### 修复方式

- 新增唯一运行时真源 `ShaftPromptSource`：task formulation 选择一份已离线物化的 target source，formulation
  内再用独立随机域选择 prompt variant；顶层 `prompts` pool 正式编译为单一 `default` formulation，不保留
  旧 `PromptSamplingTransform` 双轨。
- `data.prompt_sources.<dataset>` 统一配置 pool、逐 formulation 标准 SFT sources、train/all split、seed 与
  step/linear schedule；未配置的 dataset 直接使用 materialized row，不需要 enabled/fallback 开关。
- `ShaftSampleContext` 增加精确 dataset-local `source_draw_id`，concat/weighted 与 shuffle/unshuffle 都从逻辑
  sample stream 直接计算；curriculum 不读取 epoch、optimizer step 或跨进程可变状态。
- Arrow preflight 验证所有 formulation sources 的单 target 行、identity 对齐和 prompt variants；planning 与
  runtime 共用同一 resolver。审计收敛到 `extra.prompt_source`，execution fingerprint 绑定 source snapshots、
  source-draw 算法、pool/prompt schema、schedule 和 seed。
- 删除空的全局 `DataTransformsConfig`、旧 `PromptSamplingConfig`、旧 builder 与散落的 `runtime_prompt_*`
  状态；dataset 通用 transform 仍只由 `DatasetSourceConfig.offline_transforms/online_transforms` 声明。
- worker 环境补齐共享 CUDA 12.8 toolkit，使 PyTorch 2.10 CUDA 12.8 与 DeepSpeed 0.19 能在测试和真实 GPU
  训练中一致加载；对两个包含多次冷启动的有界测试按实测上调 timeout，仍保留死锁检测上限。

### 回归测试

- focused config/prompting/data/planning 回归覆盖 A/B/AB 离线 target 选择、static/step/linear curriculum、独立随机域、
  materialized 退化、严格 schema/未知字段拒绝、planning/runtime 一致、fingerprint 与四种 schedule 的精确
  `source_draw_id`；收口后全部通过。
- framework suite 在调整冷启动测试的有界 timeout 后整套通过；RLHF 分布式 8 场景边界用例也按正式测试
  入口通过。smoke suite 全部通过；正式 distributed suite 其余用例全部通过，并保留 3 个既有能力门禁
  skip。
- `worker-0` 的 GPU 0/1 在确认空闲后执行真实两卡 DDP：验收夹具通过单正权重 schedule 刻意产生
  `A,A,B,B,AB,AB` 以便逐项核验，并非生产抽样顺序；训练、逐步 eval 与 `checkpoint-1/2/3` 均成功，产物
  记录 `world_size=2`、3 optimizer steps、6 logical segments、770 useful tokens 与非零 CUDA peak memory。

### 后续防线

- “问什么、监督什么”的变化必须新增 task formulation；只改措辞才新增 prompt variant。不得重新引入
  `view` 或 `prompt_sampling` 平行运行时。
- curriculum 只能消费 versioned `source_draw_id` 等逻辑 sample identity；不得使用 wall-clock、worker-local
  RNG、epoch callback 或 optimizer step 反推数据阶段。
- 新 PromptSource schema/config 字段必须进入严格解析、pool/selection fingerprint、Arrow preflight、主链 smoke
  和正式文档；未知字段保持 fail closed。
- fully materialized 数据始终是一行一个 `target_text`；formulation 模式只增加对齐 source store，不增加在线
  target 解释逻辑。

## 2026-08-21：PromptSource 完成后数据文档仍停留在阶段性设计状态

### 现象

- PromptSource 已完成配置、运行时、planning/resume 和 GPU 主链验收，但正式入口仍引用
  `prompt_source_design.md`；该文件同时包含目标、迁移步骤、删除计划、当前合同和日期化验收结果。
- 根 README 仍标注“重构中”，`docs/` 没有一份统一说明 source truth、structured、SFT JSONL、
  materialized 数据与 PromptSource canonical row 的当前数据合同。
- 准备 Banana v5.8 时，需要把 v5.7 的单 materialized target 扩展为按 formulation 分开的、逐行对齐的标准
  SFT sources；若只换 prompt 不同步选择对应离线 target，监督范围会错位。

### 根因

- feature 开发期间以专项设计文档推动实现，完成后没有及时把稳定行为迁入公共 data reference，也没有删除
  已完成的迁移计划。
- 框架公共数据合同和具体 Banana 数据版本此前已经解耦，但只有 v5.7 task 文档；新的 formulation 数据格式
  缺少 task-local 发布规范。
- 文档索引按新增文件补入口，尚未再次按“当前真源 / 历史过程 / 数据版本任务”检查重复职责。

### 影响范围

- 影响框架使用者准备 SFT/PromptSource 数据、理解 A/B/AB formulation，以及后续 v5.8 builder、catalog 和
  train recipe 的发布顺序。
- 若错误预展开或维护多份 target，会破坏 canonical identity、curriculum 比例、可重建性和 exact-resume
  审计；若在数据未物化时提前登记 catalog，会把 schema-valid 配置误报为 production-ready bundle。
- 这是 data/prompt 文档与发布边界问题，不是模型能力下降，也不是 eval、codec、metric 或现有数据标签误判。

### 修复方式

- 新增 `docs/data.md`，统一 source/selection/structured/SFT 主链、三类 materialized/PromptSource row、
  formulation pool、curriculum、fingerprint、审计和发布检查；配置逐字段说明仍由 `config_reference.md` 维护。
- 删除阶段性 `docs/prompt_source_design.md`，不保留 redirect；迁移过程与验收证据只留在开发日志。
- 新增 `scripts/tasks/banana_v5_8.md`：规定每个 formulation 都生成一份 v5.7 形态、单 `target_text` 的标准
  SFT JSONL，各文件逐行对齐，PromptSource 在线只做选择；数据未物化前不创建 production catalog 或 recipe。
- 更新 README、文档索引、架构/模块/扩展/配置参考和 v5.7 task 文档；README 移除“重构中”，v5.7 明确
  使用 PromptSource 的 `default + materialized target` 简写。

### 回归测试

- config/data/PromptSource focused 集合在训练 smoke 前的 186 个用例全部通过；首次 smoke 因当前 shell 未
  继承已安装的 CUDA toolkit，12 个用例一致失败于 DeepSpeed `CUDA_HOME does not exist`，不是代码回归。
- 显式设置共享 CUDA 12.8 后重跑 `tests/test_smoke_train_modes.py`，16 个训练 smoke 全部通过。
- `ruff check src/shaft tests`、`python -m compileall -q src/shaft tests` 与 `git diff --check` 通过；检查根
  README、`docs/` 和 task 文档共 19 个 Markdown 文件，所有相对链接均存在。
- 搜索活动代码、配置和当前参考文档，不再存在 `data.transforms.prompt_sampling`、旧
  `PromptSamplingConfig/Transform` 或已删除 `prompt_source_design.md` 的引用。

### 后续防线

- `docs/data.md` 只维护框架公共数据合同；具体数据版本、业务 schema、行数和构建命令继续与
  `scripts/tasks/` 共置。阶段性设计完成后必须合并到当前真源并删除，不长期保留双轨说明。
- A/B/AB 的业务字段必须先冻结 exact output schema，再由同一 structured/source truth 离线重算各自
  `target_text`；`prompt_args` 只保存 prompt renderer 参数，不得只切 prompt 不切 target。
- builder 或 pool 文件存在不代表数据已发布。只有 media/JSONL 物化、split/schema/codec 校验和 smoke
  完成后，才创建 catalog、更新 `media_snapshot_id` 并登记训练 recipe。
- 依赖 DeepSpeed 的本地测试必须显式使用已安装的 CUDA toolkit 环境，避免把 shell 环境变量缺失误判为
  框架回归。

## 2026-08-21：PromptSource 验收序列被误解为固定子集轮换

### 现象

- 文档和交付说明反复使用 A/B/AB，并引用双卡验收中的 `A,A,B,B,AB,AB`，容易被理解为框架只支持三种
  formulation，或按固定顺序轮换。
- 实际 reconstruction 的合法属性子集可能远多于 A/B/AB，且组合依赖由业务决定，不适合由框架自动生成
  幂集或推断依赖。

### 根因

- GPU 验收为便于逐项核对，使用了每阶段只有一个正权重 formulation 的测试 schedule；记录结果时没有明确
  区分“夹具强制序列”和“生产 weighted categorical sampling”。
- v5.8 文档使用最小 A/B/AB 示例描述整体数据合同，却没有单独写出任意 formulation id、人工合法集合与复杂
  target 依赖的配置方式。

### 影响范围

- 影响 PromptSource 配置设计、v5.8 SFT 参数准备和对随机性/exact-resume 的理解；若按固定轮换准备数据，会
  错误地离线复制 row 或把业务组合规则塞入训练运行时。
- 这是文档与验收表述偏差，不是当前选择算法缺陷，也不是模型能力、eval、codec、metric 或 source label 误判。

### 修复方式

- 明确每个 logical draw 都在当时正权重 formulations 中执行 weighted categorical sampling；hash 只保证同一
  draw 可重放，不保证短前缀比例或 round-robin 顺序。
- 公共 data reference 和 v5.8 task 文档改为“人工枚举任意合法组合，在线随机选择”。框架不自动生成幂集、
  不解析 formulation 依赖，也不增加 row-level 动态 allowlist。
- 无论组合简单或复杂，derived builder 都为每个已声明 formulation 生成一份可从 structured/source truth
  重算的标准 SFT target source；训练运行时不组装 JSON。
- 一个 pool 的 rows 若不是同一 eligibility class，则拆成不同 named datasets/pools，再由通用 mixing 组合。

### 回归测试

- 新增任意 formulation 回归：人工配置 `geometry/style/geometry_style/geometry_text_links` 四种组合与
  `1:2:3:4` 权重，4,000 draws 的占比分别落在 10%/20%/30%/40% 容差内；每种 target 原子匹配，重复读取
  同一 draw 的序列完全一致。
- `tests/test_transforms.py` 12 项通过；显式 CUDA 12.8 环境下
  `test_prompt_source_formulation_sft_smoke` 主链通过。
- changed-file `ruff`、compileall、Markdown 相对链接检查与 `git diff --check` 通过。

### 后续防线

- 文档出现具体抽样序列时必须同时标明它是随机观测、固定测试夹具还是生产策略，不能从一次可复现序列外推
  为 round-robin 合同。
- A/B/AB 只能作为最小示例；正式说明必须同时覆盖任意 formulation 数量和命名。
- 组合集合和依赖由版本化 pool/derived builder 人工维护；运行时只负责严格校验、按权重选择和审计。

## 2026-08-21：渐进式 formulation 错误依赖 prompt_args 在线生成 target

### 现象

- 初版 formulation 实现把业务 atomic attributes 放入 `prompt_args`，再由 pool 的 `target_template` 在线拼出
  A/B/AB target；这使 prompt 参数 schema 同时承担监督数据 schema。
- 后续改为离线 target 时，装配逻辑一度进入 `ShaftDataCenter`，让通用 DataCenter 开始解析 formulation id、
  source mapping 和逐行对齐规则，破坏 PromptSource 的独立边界。
- source loader 仍允许“有 `prompt_args` 就可以没有 `target_text`”，继续保留了旧在线 target 路径的暗门。

### 根因

- 混淆了 prompt templating 与 task formulation：前者只改变输入措辞，后者决定监督目标。
- 把“同一 canonical sample 的多个离线训练视图”误建模为一行多参数、在线 target program，而不是多个对齐
  的标准单 target record stores。
- PromptSource 的 source preparation API 没有在设计初期收口，导致 DataCenter 临时接管了内部层级。

### 影响范围

- 影响需要任意人工属性子集、随机 formulation sampling 和渐进式权重的 SFT dataset。
- 在线模板会把业务字段/依赖耦合进框架，无法复用 v5.7 的标准 SFT 数据合同；DataCenter 耦合会让其它算法
  和 source types 被迫理解 PromptSource 私有语义。
- 本问题属于 data/PromptSource 架构与监督语义偏差，不是模型能力问题，也不是 eval、codec 或 metric 误判。

### 修复方式

- 删除 `target_template`、`target: materialized` 和 target program；formulation pool 只声明 id、权重与 prompts。
- 每个 formulation 使用一份逐行对齐、格式与 v5.7 相同的标准 SFT JSONL，每行必须有一个非空
  `target_text`；source JSONL 禁止嵌入多 target mapping。
- `prompt_args` 恢复为纯 prompt renderer 参数。source loader 即使看到非空 `prompt_args` 也仍要求 materialized
  target。
- `formulation_sources` 移入 `PromptSourceConfig`。PromptSource 自己完成 source/pool exact match、Arrow
  validation、逐行 identity 对齐、内部组合 store、随机选择与审计；DataCenter 只调用
  `ShaftPromptSource.prepare_records`，普通 `DatasetSourceConfig` 和 `SFTRecord` 不含 formulation 字段；
  `SFTDataset` 只调用通用的 opaque runtime-field hook，不识别 formulation 或多 target 语义。
- 组合 store fingerprint 优先绑定子 store snapshot；自定义 offline transform 若返回无 fingerprint 的内联
  sequence，则逐行 hash 全部 identity 与 materialized target，禁止退化成只绑定行数。
- v5.8 文档改为“每个人工 formulation 一份标准 SFT source”；复杂依赖全部由离线 builder 配置和物化。

### 回归测试

- source tests 覆盖 `prompt_args` 不能替代 target，以及 source JSONL 禁止 `formulation_targets`。
- config tests 覆盖 PromptSource 自管 formulation source path、与 dataset 顶层 train source 互斥。
- data tests 覆盖 pool/source id exact match、逐行 identity mismatch 拒绝、planning/runtime 选择同一离线 target。
- sampling tests 覆盖任意四 formulation 的 `1:2:3:4` 随机分布、step/linear curriculum、独立 prompt variant
  随机域、旧 target 配置键拒绝和普通单 target prompt rotation。
- fingerprint 回归覆盖无显式 fingerprint 的内联 formulation stores，任一 target 内容变化都会改变组合
  store fingerprint。
- 最短 SFT 训练 smoke 使用强制 AB 权重，验证 collator/trainer 实际消费预写 AB target。

### 后续防线

- `prompt_args` 永远只描述 prompt renderer 输入；任何 target 字段、组合依赖或 target 生成函数进入其中都应
  视为边界回归。
- 标准 source record 永远是一行一个 materialized target。多 formulation 只能通过 PromptSource 管理的对齐
  record stores 表达，不增加业务专用 JSONL schema。
- DataCenter、sampler、collator 和 trainer 不解析 formulation id 或 target 结构；PromptSource 新能力必须先
  通过独立 API 暴露，再由 DataCenter 做单一调用。
- 文档和测试示例出现 A/B/AB 时必须说明它只是任意人工集合的最小示例，且生产选择是随机而非固定轮换。

## 2026-08-21：把组合 formulation 误实现为训练时间 curriculum

### 现象

- 业务需要的是人工声明任意属性组合，例如 `A`、`B`、`A+B`；这些 formulation 在整个训练期间只需按固定
  概率随机采样，彼此不要求包含关系。
- 框架却额外实现了按 dataset-local draw 推进的 step/linear schedule，并在 config、sample context、mixing、
  fingerprint 和审计中维护阶段状态。
- v5.8 preparation recipe 一度配置 `source_draw=0/50000/100000`，错误地把属性组合关系变成时间 curriculum。

### 根因

- 需求讨论中没有严格区分“target schema 的包含关系”和“采样概率随训练进度变化”两种完全不同的语义。
- 为证明多 formulation 可控，验收夹具使用了阶段性单正权重序列；该测试手段反向固化成了公共配置能力。
- `source_draw_id`、插值器和 schedule fingerprint 在能力取消后没有及时按单一真源原则清理。

### 影响范围

- 影响 PromptSource 配置面、静态随机选择、sample/batch plan schema、exact-resume fingerprint 和相关文档。
- 多余阶段逻辑增加配置与恢复复杂度，也会让数据准备者误以为必须决定 curriculum 边界。
- 本问题属于 data/PromptSource 需求建模偏差，不是模型能力问题，也不是 eval、codec、metric 或 data label
  标准误判。

### 修复方式

- 删除 `PromptSourceScheduleConfig`、`PromptSourceSchedulePointConfig`、`ShaftPromptSourceSchedule` 及全部
  step/linear 插值和 schedule fingerprint；旧 `data.prompt_sources.*.schedule` 作为未知字段严格拒绝。
- formulation 概率唯一由 pool 内静态 `sampling_weight` 决定；每个 logical draw 先随机选择 formulation，再
  在其内部按静态权重随机选择 prompt variant。
- 删除只为 curriculum 引入的公开 `source_draw_id`：`ShaftSampleContext`、batch plan 序列化和
  `extra.prompt_source` 只保留 global `draw_id`，mixing 内部 occurrence 仅用于 row permutation。
- 更新 PromptSource selection/sample-context 版本与 execution fingerprint，使旧 source-draw checkpoint
  fail closed；当前文档统一把 formulation 定义为人工声明的请求属性集合。

### 回归测试

- config 回归确认静态 PromptSource 正常加载，旧 schedule 配置被严格 schema 拒绝。
- sampling 回归覆盖组合 formulation 的 `1:1:4` 分布、任意四 formulation 的 `1:2:3:4` 分布、确定性重放、
  prompt variant 独立随机域和离线 target 原子匹配。
- data/planning/batching 回归覆盖 formulation source 对齐、planning/runtime 一致、无 `source_draw_id` 的 sample
  context 序列化以及四种 mixing 路径。

### 后续防线

- formulation 是人工声明的请求属性集合；框架不假设集合之间存在包含关系，也不推断依赖或在线组装 target。
- PromptSource 概率只允许在 pool 的 `sampling_weight` 中维护；不得重新增加 callback、draw milestone 或
  另一套权重覆盖入口。

## 2026-08-21：points-only eligibility 被误建模为重复 PromptSource pool

### 现象

- `line_context_reconstruction` pool 已经定义 `appearance/points/reconstruction`，但 points-only 数据 cohort
  又维护了一份 `line_context_points.v5.8.yaml`，重复定义同一个 `points` formulation 和 prompt。
- 框架要求 `formulation_sources` 与 pool formulations 全量 exact match，使只具备 points target 的真实 line
  数据无法直接复用 line reconstruction pool。

### 根因

- 混淆了 task formulation 与 dataset eligibility：`points` 是 line reconstruction 的既有 formulation，
  points-only 只是某个物理数据 cohort 的可选范围。
- 把“这个 dataset 只能选择哪些 formulation”错误放进新 pool，而没有把
  `formulation_sources` 的键集合视为 dataset 级 eligibility 真源。

### 影响范围

- 重复 prompt 会产生两份 wording/version 真源，后续修改容易漂移；同一 line reconstruction 子任务也会被
  错报为两个 PromptSource task/pool。
- 影响 PromptSource 配置、selection fingerprint、v5.8 preparation recipe 和数据准备文档；不改变 v5.7
  单 target SFT 行格式，也不是模型能力、eval、codec、metric 或 data label 问题。

### 修复方式

- `formulation_sources` 改为共享 pool formulations 的显式非空子集，其键集合唯一声明当前 dataset cohort 的
  eligibility；未知 id 和全零 eligible 权重 fail fast。
- selection、record validation、source loading 和 audit 只遍历 eligible 子集，子集内沿用 pool 的静态权重并
  重新归一化；resolver/record fingerprint 显式绑定 eligibility id 顺序。
- 删除重复的 `line_context_points.v5.8.yaml`。`line_context_reconstruction` 与 `line_context_points` 两个物理
  dataset cohort 都引用 `line_context_reconstruction.v5.8.yaml`；后者只配置现有 `points` source。

### 回归测试

- PromptSource 单测验证共享三 formulation pool 配置单元素 `points`/`b` 子集后，任意 draw 都复用原有 prompt
  和对应离线 target，且 full/subset resolver fingerprint 不同。
- DataCenter 主链测试验证 pool 中即使其它 formulation 权重更高，只提供一个 eligible source 时，planning 与
  runtime 仍选择同一个离线 target；未知 formulation source id 被拒绝。
- v5.8 配置测试验证两个 line cohort 的 pool path 完全相同，而 eligibility 分别为三 formulation 与
  `points` 单元素子集。
- focused config/data/PromptSource 回归全部通过；默认全量回归首次有 49 项统一失败于 shell 未设置
  `CUDA_HOME`，在 `worker-0` 显式启用用户级 CUDA 12.8 toolkit 后 `--lf` 49 项全部通过。
- `worker-0` GPU 0 上的 PromptSource 最短 SFT 训练 smoke 通过，确认 trainer 实际消费共享 pool 中选中的
  formulation prompt 与预写 target。

### 后续防线

- task/prompt 语义相同且 formulation 已存在时，不得为了数据 eligibility 新建或复制 pool/prompt。
- 不同 eligibility class 继续拆成 named dataset cohort，便于外层 mixing；每个 cohort 只通过
  `formulation_sources` 键选择共享 pool 子集，不增加 row-level allowlist。
- 文档和测试必须同时区分 pool 的完整 formulation 集合与 dataset 的 eligible formulation 子集。

## 2026-08-21：Grounding 被过度拆成 labels/boxes/objects formulations

### 现象

- v5.8 初版把 `grounding_layout` 拆成 `labels`、`boxes`、`objects` 三个 formulations，并要求分别物化三份
  target JSONL。
- 实际 grounding 任务始终需要完整的 `bbox_2d + label` objects；详细/简略只是等义 prompt wording，不是
  不同监督属性集合。

### 根因

- 把 shape/line 中确实存在的子属性请求模式泛化到所有任务，没有先判断 grounding 是否存在独立、稳定且有
  使用价值的 partial target 合同。
- 混淆了 formulation 与 prompt variant：前者必须改变请求属性集合和离线 target，后者只能改变措辞。

### 影响范围

- 会让 grounding builder、存储和对齐校验无意义地扩成三份数据，并把训练概率从 dataset mixing 进一步拆到
  formulation sampling。
- 影响 v5.8 grounding pool、catalog、preparation recipe、配置测试和数据准备文档；grounding 的 source truth、
  完整 objects target schema、图片与 split 均不改变。这不是模型能力或 eval/codec/metric 误判。

### 修复方式

- `grounding_layout.v5.8.yaml` 恢复顶层 `prompts`，只保留 `detailed/concise` 两个等义 variants；两者始终请求
  `[{"bbox_2d":[...],"label":"..."}]` 完整 objects。
- catalog 恢复普通 `data/grounding_layout/sft/train.jsonl` 与 `val.jsonl` source；training recipe 删除
  grounding 的 `formulation_sources`。
- shape/line 多 formulation 与 line points-only eligibility 设计保持不变。

### 回归测试

- v5.8 config test 验证 grounding pool 编译为非显式 `default` formulation、没有
  `formulation_sources`、使用普通 materialized dataset source，并继续提供 detailed/concise variants。
- focused config/PromptSource/DataCenter 回归 82 项通过；changed-file ruff 与 `git diff --check` 通过。

### 后续防线

- 只有请求属性集合和离线 target 都发生变化时才新增 formulation；纯 wording 变化只能作为 prompt variant。
- 数据准备前逐任务确认是否真的需要 partial target，不能仅因框架支持 formulation 就为所有任务机械拆分。

## 2026-08-21：compact raw 缺失 background 与 image type 真值，不能直接重建 v5.8 任务

### 现象

- 当前 active compact raw 的 19,797 份 JSON 均不含顶层 `background`，54,152 个 image instance 也都不含
  13 类 `image_type`；只读取当前 JSON 会把两个历史真实任务误判为不可恢复或诱发在线猜标签。
- 历史 v5.3 任务 bundle 仍保存完整 structured/SFT/task-local media，人工审核 background 标注也仍可追溯，
  但 image enriched raw 已不可用，二者的来源等级不同。

### 根因

- compact raw 收敛时只保留当前布局训练所需字段，没有承诺继续承载所有历史专项审核标签。
- 旧 derived bundle、review annotation 与 active raw 的真值边界未在 v5.8 数据合同中显式记录，容易把恢复产物
  误写回 raw 或把 derived image-type 数据冒充原始标注。
- 本问题属于 data source truth 与派生层级偏差，不是模型能力问题，也不是 eval、codec 或 metric 误判。

### 影响范围

- 影响 v5.8 `background` 与 `image_context_reconstruction` 的可复现构建、测试集排除、媒体冻结、catalog、
  PromptSource 绑定和 resume snapshot 语义；不改变 Shape/Line formulation 设计。
- 若直接沿用旧目录而不复验，可能混入测试样本、丢失 target/媒体一致性或在训练时引用不可追溯像素。

### 修复方式

- 新增显式输入的恢复脚本：background 以 reviewed annotation 为标签真值，并与历史 structured/SFT/media
  逐 ID、逐 target 对账；image type 标记为 `verified_recovered_v5_3_derived_bundle`，不提升为 raw truth。
- 历史审核文件与 split manifest 复制到独立 raw import sidecar，active compact raw 保持不变；task media 原像素
  hardlink/copy 到 staging，不执行 resize。
- raw sidecar 与两个 task report 冻结历史 selection/structured/SFT、审核标注、split manifest 和 PromptSource
  pool 的 SHA256，避免同名目录或同名配置变化后仍冒充同一 snapshot。
- 恢复过程先完成 PromptSource、schema、测试集 identity、媒体解码与声明尺寸校验，再原子发布。即使指定
  `--clean`，验证失败也不会先删除已有 task 目录。
- `background.v5.8` 只保留一个 detailed prompt，明确区分大面积不可编辑 backing、简单画布与局部 image
  object；catalog 和 preparation recipe 为 background 绑定普通 materialized SFT source。

### 回归测试

- 恢复脚本单测覆盖成功恢复、空 row prompt、materialized target、image formulation 路径，以及失败时已有输出
  不被 `--clean` 覆盖。
- 40 进程实际恢复并复验 38,443 条 background 与 21,184 条 image-type 数据；59,627 个媒体完整解码且尺寸
  一致，错误为 0。
- canonical `vlm.test.json` 当前/历史 manifest 均为 175 项且 SHA256 一致；grounding、background、image-type
  已物化训练源的 source identity overlap 均为 0。background 进一步排除三个历史 test manifest 的 313 项
  并证明恢复 ID 精确等于 reviewed annotation ID 减去该并集。

### 后续防线

- compact raw 缺字段时必须先定位可审核的上游/历史真值；不得从现有 target、prompt 参数或模型输出在线猜测。
- 恢复 derived bundle 必须显式标注来源等级、冻结 snapshot、保存输入 hash 并保持 raw sidecar 与 active raw
  解耦；只有可证明的原始审核标注才能称为 raw truth。
- 每个 v5.8 物化 task 发布前必须证明与 canonical 175 张测试集 source identity 重叠为 0；“合成来源”也不能
  替代正式去重审计。

## 2026-08-22：V9 reconstruction selection 与 formulation 物化语义未覆盖 v5.8 合同

### 现象

- 旧 selection 先抽取 full line，再从已选 line 中只按 segment count 取 points subset；这会丢掉未进入 full
  selection 的稀有多叉组合，也无法表达“完整 line 属性越稀有越优先”。
- context builder 只写一份 full `sft/train.jsonl`，不能在同一 crop pass 中生成 appearance、geometry/points、
  reconstruction 三份逐行对齐 target。
- shape/line 头部只有平方根配额，没有显式 rectangle/单段 line 占比合同；直接把 rectangle capacity 提前截断
  还会改变平方根权重，导致 rectangle 从合理降采样变成过度降采样。

### 根因

- v5.7 的单 target、二级 segment balancing 被直接沿用到 v5.8，没有把业务 formulation eligibility、稀有
  stratum 保护与离线 materialization 收口到同一数据准备流程。
- 把“限制最终占比”误实现为“在配额计算前改写自然 capacity”，混淆了抽样上限与分层权重真源。
- 本问题属于 data selection / derived target 语义偏差，不是模型能力问题，也不是 eval、codec 或 metric 误判。

### 影响范围

- 影响 v5.8 shape/line/line-points 的样本分布、PromptSource formulation stores 对齐、合成噪声审计与训练
  snapshot；不修改 V9 raw/gt_standard，也不改变 PromptSource 在线随机实现。
- 若不修复，points-only cohort 可能遗漏最稀有的箭头/分叉组合，full reconstruction 仍只能训练单一 target，
  训练配置中声明的 formulation sources 会指向不存在的数据。

### 修复方式

- 增加显式 `v5.8` selection profile：稀有 shape type 全保留；先按自然 stratum 分配，再只在超限时截断
  rectangle 并把溢出配额补给其它类型；直接删除 120 个包含连续重复点/零长度 segment 的坏 line instance，
  其余 118,996 个有效多叉 line 全保留，单段 line 最终最多 60%。
- synthetic points 从全部有效 V9 多叉 line 独立选择；完整 line-attribute stratum 数量不超过 256 的行全保留，
  再无放回补足 15,000 条，不依赖 full line selection。
- context builder 使用统一 PromptSource pool 编译入口识别显式 formulation，在同一 worker pass 复用一份
  crop/structured row，离线写出逐 formulation 标准 SFT；line points cohort 只声明共享 pool 的 `points`。
- builder 增加 `--preflight-only`：全量执行 source resolution、确定性 crop/量化、formulation projection 和
  augmentation plan，但不写媒体、不发布目录；避免单个晚序坏样本触发数十万 PNG staging 的昂贵回收。
- formulation task 的构建报告只保留 `reports/build_summary.json` 一个真源；普通 v5.7 单目标输出继续保留
  task 根目录旧路径，不为新目录复制两份相同 summary。
- shape/line/synthetic-points crop 统一记录并应用尺寸不变的 `synthetic_realism_v1`；120,744 条真实 points
  全保留且不加合成噪声，空 points 不补造。

### 回归测试

- selection 单测覆盖稀有 shape 保留、rectangle 最终 cap、多叉 line 全保留、单段 cap 与完整属性 rare-stratum
  保留；旧 v5.7 默认行为保持通过。
- builder 单测覆盖 card appearance/geometry exact projection、三 formulation 逐行 identity 对齐、相对媒体路径、
  单一 reports 真源与单 points eligibility；focused builder/selection 测试及 changed-file ruff 通过。
- 真实 V9 canary 覆盖 shape/line 各 2,000 条、合成 points 2,000 条和真实 points 2,000 条：合成样本全部记录
  `synthetic_realism_v1`，真实样本全部为 `none`，三 formulation 行数与 identity 对齐。
- 清洗后 297,489 条 full line 与 135,744 条真实+合成 points 全量 preflight 通过，量化后 segment collapse、
  source/index drift 和 formulation projection error 均为 0。
- 最终物化的 Shape 300,000、Line 297,489、Points 135,744 共 733,233 张 crop 全部重新解码且声明尺寸一致；
  三类 formulation store 的 exact target、唯一 ID 和逐行 identity 全量通过。
- 真实 v5.8 配置经 DataCenter 装配为 850,420 个 logical rows；首次建立 10 份 Arrow cache 后执行 6,000 次
  weighted draw，覆盖全部合法 formulation 与长短 variant，六个 cohort 的实际媒体读取及 execution/stream
  fingerprint 完整性通过。

### 后续防线

- selection 报告必须同时给出 available/selected 的 type、segment 与完整 attribute stratum 分布；不能用单一
  segment count 代替业务稀有度。
- 最终占比 cap 必须施加在初始自然配额之后；若截断头部，需要显式重分配溢出配额并测试最终比例。
- 多 formulation builder 必须单 pass 生成所有 eligible stores；训练在线阶段只能随机选已物化 target，不能从
  full target 截字段，也不能把 target 组合真值塞进 `prompt_args`。

## 2026-08-22：Qwen 产品 alias 共用架构但误共用 thinking 模板合同

### 现象

- `qwen35vl`、`qwen36vl` 共用 `qwen35vl` / `qwen35vl_thinking` 两个模板，新增 `qwen38vl` 时也会继承同一
  默认值；推理 policy 还通过模板名字符串独立猜测 thinking 开关。
- 标准 SFT message loader 会丢弃 `reasoning_content`，当前轮也只能留下 `target_text`，因此配置 thinking
  模板并不等于能正确监督 CoT。

### 根因

- 把 HF `qwen3_5` architecture 复用错误扩大成 chat-template 产品合同复用；Qwen3.5 只支持
  `enable_thinking`，Qwen3.6 增加 `preserve_thinking`，Qwen3.8 又增加三档 `reasoning_effort`。
- template、inference policy 与 SFT record 各自只实现了局部开关，没有统一的模板元数据真源和结构化
  reasoning target 合同。
- 本问题属于 template/data 语义偏差，不是模型能力问题，也不是 eval、codec 或 metric 误判。

### 影响范围

- 影响 Qwen3.5/3.6/3.8 本地 SFT prompt 渲染、CoT target 监督、历史 assistant reasoning 保留，以及
  OpenAI-compatible/vLLM 推理的 `chat_template_kwargs`。
- 非 thinking 的既有结构化任务仍保持关闭 thinking；旧配置若显式写 `template: qwen35vl`，继续按 3.5
  非 thinking 合同执行，不会被产品默认值迁移隐式改写。

### 修复方式

- `qwen35vl`、`qwen36vl`、`qwen38vl` 保留共享 loader/processor/sequence/sharding 实现，但分别使用同名
  非 thinking 默认模板；注册 3.5/3.6 thinking 模板及 Qwen3.8 `xhigh/medium/low` 三档模板。
- `TemplateMeta.chat_template_options` 成为本地 processor 与远端推理的单一选项真源，删除 inference policy
  的模板名判断。
- SFT record、JSONL normalization、Arrow cache、dataset 与 PromptSource formulation selection 全链路携带
  `target_reasoning_content`；历史 assistant 的 `reasoning_content` 保留在 messages。
- thinking 模板把 generation prompt 已打开的 `<think>`、推理正文、闭合标签和最终答案编译为 continuation；
  非 thinking 模板收到 reasoning，或 thinking 模板收到未结构化的普通答案时 fail closed。

### 回归测试

- 单测覆盖三代模板的精确 kwargs、Qwen3.8 reasoning effort、版本化模型默认模板与远端请求复用同一元数据。
- 数据测试覆盖末尾 assistant reasoning 提取、历史 reasoning 保留、Arrow roundtrip、dataset 暴露，以及
  PromptSource 同 formulation 的 answer/reasoning 联动选择与 fingerprint。
- supervision 测试覆盖结构化 reasoning 编译、非 thinking 拒绝 reasoning、thinking 拒绝未闭合普通 target；
  Qwen3.6/3.8 可选真实 processor integration gate 使用各自产品模板。

### 后续防线

- 新产品版本可以复用 HF architecture adapter，但必须独立审计官方 `chat_template.jinja` 的参数、默认值、
  历史 reasoning 与 generation prompt 行为；不得仅凭 `config.model_type` 相同就复用模板。
- chat-template 行为必须登记在 `TemplateMeta` 并同时驱动训练与推理；禁止在 backend policy 中维护模板名
  allowlist。
- CoT 数据必须显式区分 reasoning 与最终答案，loader/template 对不匹配的模板和标注 fail closed，不能静默
  丢弃 reasoning 或猜测 `<think>` 边界。

## 2026-08-22：分布式首次启动把 Arrow records cache 串行化到单个 rank

### 现象

- Banana v5.8 八卡启动长期停在 `startup.data/loading`，GPU 仅建立约 911 MiB CUDA context，模型尚未加载。
- 10 份 JSONL/formulation source 共 2,045,398 条、5.18 GB；每份 300K source 冷缓存约需 3 分 20 秒，
  全部串行预计 20–25 分钟。
- 8 个训练 rank 均已存活，但七个 rank 阻塞在同一 `~/.cache/shaft/records/*.lock`，只有一个 rank 写 Arrow。

### 根因

- 每个 rank 按相同 dataset/formulation 顺序调用 `from_jsonl()`；排他文件锁保证了原子发布，却也让所有 rank
  依次争用同一个任务，没有利用独立 source 之间天然可并行的关系。
- `data.num_workers` 只在正式 DataLoader 创建后生效，不能加速模型加载前的 JSON parse、record normalization、
  PromptSource render validation 和 Arrow serialization。
- 已发布 cache hit 仍先获取 builder 排他锁，使预热后的多 rank mmap 打开也发生短暂串行化。
- 本问题属于 data runtime/cache 编排与启动可观测性偏差，不是数据内容、模型能力或 eval/codec/metric 误判。

### 影响范围

- 影响 SFT、RL 与 OPD 共用 `ShaftDataCenter` 的多进程冷启动；单进程语义、cache fingerprint、训练样本顺序、
  PromptSource 选择及 checkpoint resume contract 不变。
- cache 已命中时主要是额外锁等待；cache 未命中且存在多份大 JSONL/formulation source 时 wall time 最明显。

### 修复方式

- DataCenter 将每个独立 JSONL/formulation store 暴露为指纹化 cache task；`ShaftRecordCachePlan` 对 task 去重，
  再按 source bytes largest-first 贪心分配到 `LOCAL_WORLD_SIZE` 个 shard。
- 每个 torchrun rank 在正式 records 装配前只预热自己的 local-rank shard；每个节点独立覆盖完整 task 集，兼容
  node-local cache，多节点共享 cache 继续由既有 `flock + os.replace` 保证单一原子发布。
- Arrow cache hit 增加 immutable fast path：先并发验证并打开已发布文件，只有缺失或损坏时才进入排他修复路径。
- rank 0 启动日志公布 task 数、source 总字节、各 shard 估算字节与 plan fingerprint；各 rank 记录自身 task、
  rows 和耗时，关闭 `rank_zero_only` 时可展开逐 rank 诊断。

### 回归测试

- 单测覆盖三个 materialized formulation source 形成三个独立任务、三 shard 无重叠覆盖、预热后正式装配不再
  调用 Arrow builder，以及 torchrun 环境自动采用 `LOCAL_RANK/LOCAL_WORLD_SIZE`。
- cache hit 测试禁止获取 `fcntl.LOCK_EX`，证明已发布 mmap 不再被 builder 锁串行化。
- 两进程真实 torchrun smoke 证明两个 local rank 各预热两份互不重叠 source，合并后覆盖四份 cache，最终两端
  均能加载完整 dataset。
- 真实 Banana v5.8 配置只读计划检查得到 10 tasks / 8 shards；六份 716.6–807.1 MB 大 source 独占 shard，
  四份小 source 合并为 292.9 MB 与 325.6 MB 两个 shard，总 source bytes 仍精确为 5.18 GB。

### 后续防线

- `num_workers`、DataLoader prefetch 与 record cache warmup 必须保持不同语义，禁止用一个字段同时控制两阶段。
- 新 data source 若使用可预热的 immutable cache，应通过 `BaseDataSource.record_cache_tasks()` 暴露原子任务；
  不支持预热的 source 返回空任务并沿用普通加载，不在 DataCenter 猜测 source 私有格式。
- cache plan 只优化物化顺序，不进入训练数据/采样语义；fingerprint、验证规则和原子发布必须继续由 record
  store 单一真源维护。

## 2026-08-22：项目命令反复因 shell 未继承 CUDA_HOME 而失败

### 现象

- CUDA 12.8 toolkit 已安装且 `nvcc` 可用，但新 shell 中没有 `CUDA_HOME`；依赖 DeepSpeed 的 smoke 在收集或
  导入阶段反复报 `MissingCUDAException: CUDA_HOME does not exist`。
- 同一套测试只要手工导出 toolkit 路径即可通过，导致环境准备需要在不同命令中重复补写。

### 根因

- toolkit 位于用户级缓存目录，不在系统默认 CUDA 路径；项目此前没有本地、可忽略且由正式入口加载的环境
  文件。
- 测试和训练依赖调用方 shell 偶然继承环境变量，项目环境真源缺失。

### 影响范围

- 影响会在导入期检测或编译 CUDA 扩展的训练与测试命令；不改变模型、数据、训练配置或 checkpoint 语义。
- `torchrun` 的 hostname `err=-3` 是独立的反向解析警告，不由 `CUDA_HOME` 引起。

### 修复方式

- 增加被 Git 忽略的项目本地 `.shaft.env` 与可提交的 `.shaft.env.example`；训练入口和 pytest 在导入训练栈前
  加载该文件。
- loader 只接受 `NAME=value` / `export NAME=value`，不执行 shell 展开或命令替换；调用方 shell 已显式设置的
  同名变量始终优先。

### 回归测试

- 从显式移除 `CUDA_HOME` 的子进程执行训练入口探针，确认自动恢复 CUDA 12.8 路径且 DeepSpeed 0.19.0 可导入。
- 同样移除 `CUDA_HOME` 后运行完整 smoke suite，全部通过；CLI 单测覆盖显式 shell 值优先与非法 assignment
  fail fast。

### 后续防线

- 非系统路径的 toolkit 统一登记在本地 `.shaft.env`，不再把个人绝对路径写入源码、公共配置或提交记录。
- 新增项目环境项时继续保持无 shell 求值、显式环境优先；需要跨机器共享的只提交 example，不提交实际值。

## 2026-08-22：Qwen3.5 首步 FLA Triton JIT 在共享缓存上丢失 metadata

### 现象

- Banana v5.8 Qwen3.5-4B 八卡训练完成 data、model 和 optimizer 初始化后，在 step 0 的首次 forward 失败。
- rank 0/1 均在 FLA `chunk_gated_delta_rule` 的 Triton autotune/compile 路径抛出
  `FileNotFoundError: [Errno 2] No such file or directory`，随后 torchrun 终止其余 ranks。

### 根因

- `TRITON_CACHE_DIR` 未设置，Triton 3.6.0 默认把 JIT cache 写入位于 CLOUDSTOR_FS 的
  `~/.triton/cache`。
- 多个 rank 首次编译时并发发布/替换 kernel metadata；读取方已经打开 metadata 后，在 `f.read()` 阶段遇到
  共享存储上的文件替换竞争并返回 ENOENT。
- 这是 Triton JIT cache 的运行时存储问题，不是模型能力、数据样本、CUDA_HOME、hostname、eval、codec 或
  metric 问题。

### 影响范围

- 影响首次触发 FLA/Triton kernel 编译的多进程训练；data cache、模型权重和训练输出合同不受影响。
- 已生成的部分 Triton cache 可能让原命令偶然越过同一 shape，但新 kernel/shape 仍可能再次触发。

### 修复方式

- 项目本地 `.shaft.env` 将 `SHAFT_TRITON_CACHE_ROOT` 指向 worker 节点本地 `/tmp`，不再使用共享 home cache。
- 训练入口在导入训练栈前按 `TORCHELASTIC_RUN_ID/LOCAL_RANK` 派生 rank 独立的 `TRITON_CACHE_DIR`；调用方
  显式 `TRITON_CACHE_DIR` 保持最高优先级。
- 不删除旧 `~/.triton/cache`；新训练通过新路径自然隔离历史残留和共享存储竞争。

### 回归测试

- CLI 单测覆盖 run id 安全化、local-rank 目录隔离、目录提前创建和显式 override 优先。
- 从未设置 `TRITON_CACHE_DIR` 的两进程 torchrun 环境在两张 A800 上并发编译、执行同一 Triton kernel；两个
  local rank 分别发布完整 metadata/PTX/cubin 到节点本地目录，数值结果与目录隔离检查均通过。

### 后续防线

- GPU JIT cache 不得默认落在 NFS/CLOUDSTOR_FS；多 rank 首次编译必须使用节点本地目录，必要时进一步按 rank
  隔离。
- 遇到 step 0 的裸 ENOENT 必须保留完整 traceback，先区分 input path 缺失与 compiler cache metadata 竞争，
  不能只依赖进度摘要中的异常字符串。

## 2026-08-22：Qwen3.5-4B 长序列在加权交叉熵阶段触发显存峰值 OOM

### 现象

- Banana v5.8 Qwen3.5-4B 八卡 DDP 已稳定运行到 step 157，随后 rank 5 在
  `training/loss.py::causal_lm_cross_entropy` 申请 6.85 GiB 失败；该卡只剩 6.36 GiB 空闲，PyTorch
  allocated 58.05 GiB、reserved-but-unallocated 12.87 GiB。
- 其他 rank 在 600 秒后报 NCCL ALLREDUCE timeout；这是 rank 5 提前 OOM 后的二次症状，不是 sampler
  卡间不均衡或 collective 首先失败。事故前 `rank_time_skew` 约 1.1%。

### 根因

- 原 loss 将完整 `[batch, sequence, vocabulary]` logits 连续化，再一次性执行 `reduction=none` CE。
  Qwen3.5 的 248,320 词表使长序列产生数 GiB 的 log-softmax/CE 工作区，并一直保留到 backward。
- 仅把普通 CE 按 token 循环切块仍会让每块 autograd 缓存同时存活，不能从结构上降低峰值；
  `expandable_segments` 只能缓解 12.87 GiB reserve 的碎片，不能消除 vocabulary-sized 临时张量。
- 本问题是训练 loss 内存实现缺陷，与 PromptSource、数据标注、token-average 归一化和模型能力无关。

### 影响范围

- 影响使用 Shaft 自定义加权/global-denominator causal LM loss 的大词表、长序列训练；序列越长、local
  batch 越大越易触发。Qwen3.8-27B 当前 BS1/GA8 ZeRO-3 配置也自动受益，但未因本事故调整其训练语义。
- 失败发生在 checkpoint 间隔内，没有产生可 resume 的 step 157 checkpoint；已完成的旧 checkpoint
  格式和模型导出格式不变。

### 修复方式

- `training/loss.py` 改为内存有界的 causal LM CE：forward 最多按 512 个扁平 token 计算一块且不保存
  vocabulary-sized autograd 中间量；backward 逐块重算 CE，并把梯度写入唯一的完整 logits gradient。
- 保留 `loss_scale`、ignore index、跨 GA/DP global denominator、逐样本 numerator/denominator 和
  token-average 的原有数学语义；低精度 logits 的 CE/recompute 统一使用 FP32。
- 4B v5.8 配置改为 fixed BS1/GA8，保持八卡 global sample batch 为 64；`max_length` 与
  `max_tokens_per_microbatch` 统一降为 8000。项目本地 `.shaft.env` 启用
  `PYTORCH_ALLOC_CONF=expandable_segments:True`，公共 example 同步记录该项。

### 回归测试

- loss 单测比较分块实现与原始 PyTorch CE 的 FP64 value、weighted component numerator/denominator 和
  logits gradient；同时覆盖全局 denominator 拆分不变性与 EOS shift。
- 工作区边界测试记录 forward/backward 的每次 CE 输入，确认均不超过指定 token cap，且 backward 确实
  逐块重算而非保留全部块的 log-softmax。
- 4B 配置加载和 SFT 装配 focused smoke 必须通过；真实八卡 8000-token 长训由本次重启继续验证，完成前
  不把该修改表述为已通过长程 GPU canary。

### 后续防线

- 大词表 loss 的内存验收必须同时检查 forward 临时工作区和 backward 保存张量；“循环调用普通 CE”不算
  内存有界实现。
- 单 rank CUDA OOM 后出现的 NCCL timeout 统一视为二次症状，排障先保留最早 rank traceback；batch planner
  skew 只能解释等待差异，不能解释 loss 内 6.85 GiB 的单次分配。
- 调整 BS/GA 时必须保持显式 global batch 语义；allocator 配置只作为碎片防线，不能替代框架级有界 loss。

## 2026-08-24：Offline KD vLLM scorer 移除 token-run 折叠与双侧 resize

### 现象

- 上一版为绕过 vLLM 图片 placeholder 双重展开，在 production path 中扫描本地 processor 已展开的连续
  image-token run，并将每段压缩成一个 token；同时把 `min_pixels/max_pixels` 传给本地 processor 和 vLLM。
- 该实现能通过特定 Qwen canary，但“连续 token 等于一张图”并不是 template/processor 的结构化合同；预算在
  两侧重复传递也不能证明两者使用了完全相同的 resized pixels。

### 根因

- producer 混用了 template rendered prompt（每图一个未展开 placeholder）和 processor scoring prompt（视觉
  token 已展开）两个层级，并用 token 值猜测结构，形成临时桥接代码。
- resize 的真源没有收敛到 producer：本地 HF processor 与 vLLM 各自收到预算，后端版本或 rounding policy
  变化时可能产生不同视觉网格。这是 Offline KD producer 输入合同问题，不是 teacher/student 模型能力问题。

### 影响范围

- 影响多模态 Offline KD 的 vLLM artifact 生产；JSONL+safetensors、top-K+tail 分布格式、artifact identity 与
  reader/trainer 合同不变。
- OPD、SFT、DPO、PPO、GRPO 等训练算法运行时不在本次修改范围。

### 修复方式

- producer 先调用既有模型 adapter `prepare_rollout_image()`，对每张图执行一次 Shaft smart resize；同一个
  resized PIL object 同时交给本地 processor 和 vLLM。本地 processor 不再接收 pixel budget，vLLM request
  也不再传 `min_pixels/max_pixels`。
- vLLM prompt 直接由 template 的 structured rendered token plan 和实际 target suffix 构造，每张图恰好一个
  未展开 placeholder；删除 production image-token run collapse。vLLM 返回的已展开 prompt IDs 必须逐 token
  严格等于 Shaft collated `input_ids`，不允许 decode/re-tokenize 或静默修补。

### 回归测试

- 新增单图、多图、processor 不接收二次 pixel budget、vLLM request 不携带预算以及 double-expansion 拒绝测试；
  fake engine 必须模拟 vLLM 展开后的返回 IDs，不能只回显未展开 request。
- Offline KD producer/artifact/pipeline focused suite 35/35 通过。真实 Qwen3.5 tokenizer/processor canary 验证：
  64×32 单图对应 vLLM 1 个 placeholder、本地 72 个 image tokens；两张 64×32/32×64 图片对应 2 个
  placeholder、本地 144 个 image tokens，且两条路径接收相同 PIL object。

### 后续防线

- 外部多模态引擎的未展开 request prompt 与本地展开 scoring prompt 必须由同一 structured plan 派生；禁止
  恢复 token-run 折叠、后端二次 resize 或按 token 数量容错。
- 真实 Qwen3.8 checkpoint-4000 仍需重跑新协议 canary 和 HF/vLLM 数值容差 parity；旧 200 条 artifact 证明的
  是上一版输入路径，不能替代新协议 release gate。
