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
