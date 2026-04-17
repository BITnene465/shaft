# Shaft 导出与合并

本文档描述 `src/shaft/export` 的设计边界和当前支持的 HF 兼容工具链。

## 1. 总原则

- 只处理 HF / PEFT 标准目录。
- 不生成自定义 metadata 目录。
- 不引入额外中间格式。
- full 模型训练结果本身就应是标准 HF export。
- adapter 训练结果本身就应是标准 PEFT export。

## 2. 当前能力

### 2.1 `inspect`

识别目录类型：

- `full`
- `adapter`
- `trainer_state_only`
- `unknown`

对应接口：

- `inspect_hf_artifact()`
- `inspect_checkpoint_layout()`

### 2.2 `validate`

校验目录是否符合当前 `finetune_mode` 的 HF/PEFT 规则。

对应接口：

- `validate_hf_artifact()`
- `ensure_hf_export_layout()`

### 2.3 `merge-peft`

把 adapter 合并为标准 HF full export。

对应接口：

- `merge_peft_adapter()`

输出产物要求：

- 包含标准 HF `config.json + model weights`
- 优先保留 adapter 侧 tokenizer / processor 资产
- 不额外生成 Shaft 自定义产物层

## 3. 目录类型约定

### 3.1 Full export

需要有：

- `config.json`
- `model.safetensors` 或等价 HF 权重文件

### 3.2 Adapter export

需要有：

- `adapter_config.json`
- `adapter_model.safetensors` 或等价 adapter 权重文件

说明：

- adapter export 仍然是标准 PEFT 目录，不是标准 full HF model 目录。
- adapter 目录可以包含 `modules_to_save` 对应的额外原始模块权重，这仍然属于 PEFT 语义。
- 这类目录应通过 `base model + PeftModel.from_pretrained(...)` 使用；若部署后端只接受 full HF model，应先 `merge-peft`。

### 3.3 Trainer state

需要有：

- `trainer_state.json`

说明：

- `resume_from_checkpoint` 依赖 trainer state。
- `init_from_checkpoint` 只关心模型或 adapter 权重本身。

## 4. 与训练状态的关系

### `init_from_checkpoint`

- 作用：初始化模型权重
- 可接受：
  - full checkpoint
  - adapter checkpoint
- adapter checkpoint 初始化时，会额外校验：
  - LoRA/DoRA/QLoRA 关键配置一致
  - `target_modules` 一致
  - `modules_to_save` 一致

### `resume_from_checkpoint`

- 作用：恢复 trainer 状态
- 需要：
  - `trainer_state.json`
  - 与当前 `finetune.mode` 匹配的权重布局

## 5. 文档化边界

### 允许

- 增加新的 HF 兼容检查
- 补充更严格的 adapter/full 校验
- 增加 merge 前后验证

### 禁止

- 设计自定义 checkpoint 布局
- 在导出层实现发布平台集成
- 把导出目录变成训练状态仓库

## 6. 何时不该进入主干

若需求只是：

- 为某个部署平台打包额外文件
- 上传到某个内网服务
- 写一次性转换脚本

那应该放到独立脚本或发布工具，不应污染 `src/shaft/export`。
