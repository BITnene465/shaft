# Shaft 开发流程

本文档定义当前仓库的标准开发流程。目标是：减少越界修改、减少返工、让文档、测试和代码保持同步。

## 1. 总流程

```mermaid
flowchart LR
    A[明确需求与边界] --> B[确认落在哪个模块]
    B --> C[先改测试或补测试]
    C --> D[实现代码]
    D --> E[更新文档]
    E --> F[执行回归测试]
    F --> G[代码审查与简化]
    G --> H[提交]
```

## 2. 第一步：先判断改动属于哪一层

### 进入 `config`

- 新增配置块
- 新增字段
- catalog 加载规则变化

### 进入 `data`

- 新数据源
- 新记录格式
- 新增强
- 新 mixing 策略
- 新 collator

### 进入 `model` / `template`

- 新模型族
- 模型专属 template
- processor / peft policy

### 进入 `algorithms`

- 新训练算法
- DPO/PPO 参数映射
- TRL/HF trainer 装配

### 进入 `pipeline`

- 训练主链编排变化
- 保存/评估/回调装配变化

### 进入 `training`

- 新 loss / optimizer / scheduler
- checkpoint 规则变化
- trainer 内核扩展

### 进入 `infer`

- 新推理后端
- 新 stage 执行规则
- 新 codec

### 进入 `export`

- HF 兼容检查
- adapter merge

## 3. 第二步：先写或先改测试

### 必须先补测试的场景

- 新增注册项
- 新增数据格式
- 新增训练配置字段
- 新增导出/恢复规则
- 修改推理 stage 或 codec 行为

### 推荐测试顺序

1. 单元测试
2. 轻量 smoke
3. integration/manual

## 4. 第三步：实现代码时的边界约束

### 一般原则

- `pipeline` 只编排，不承载任务语义。
- `data` 只产出样本，不碰训练状态。
- `model` 只收口模型差异，不读数据路径。
- `cli` 只做参数解析和调度。
- `scripts/*.py` 只做薄包装。

### 命名原则

- 框架级抽象统一 `Shaft*`
- 模型专属实现显式带模型名
- 配置/格式强绑定对象名称必须反映边界，如：
  - `TrainConfig`
  - `EvalConfig`
  - `DatasetSourceConfig`
  - `ShaftDatasetMeta`

## 5. 第四步：同步文档

新增或重构能力后，至少更新以下文档之一：

- `docs/architecture.md`
- `docs/module_reference.md`
- `docs/config_reference.md`
- `docs/extension_guide.md`

如果是用户可直接使用的能力，还应同步：

- `README.md`
- `docs/README.md`

如果本轮修复的是重复出现的 bug、指标误判、训练/评估语义偏差，必须同步更新：

- `docs/development_log.md`

开发日志需要记录现象、根因、影响范围、修复、回归测试和后续防线。不能只在聊天记录或临时排障输出中保留结论。

## 6. 第五步：回归测试

至少执行：

```bash
pytest -q
```

如果改动涉及真实模型加载、推理或后端：

```bash
pytest -q -m integration
```

如果改动涉及手工环境：

```bash
pytest -q -m manual
```

## 7. 第六步：提交前审查

提交前至少自查：

1. 有没有把模型专属逻辑写进通用层。
2. 有没有把业务脚本逻辑误塞进内核。
3. 文档是不是还在用旧命名。
4. 新增模块是否真的被主流程使用。
5. 是否引入了新的“平行实现”而不是复用现有入口。

## 8. 常见反模式

- 在 `scripts/*.py` 里写完整 argparse 业务逻辑
- 在 `pipeline` 里复制 mixing 或 template 逻辑
- 在 `training` 里读取 JSONL
- 在 `infer` 里固化某个单次离线任务
- 在 `export` 中发明新的目录格式
