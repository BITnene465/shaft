# Dense Qwen3.5 SFT：融合 linear CE 验证（2026-09-18）

## 结论与实现范围

复用 Liger 0.8.3 的融合 linear + CE，消除适用 SFT 训练路径上的完整 logits/grad_logits。
不自研 GPU kernel，不增加多后端注册体系，不更换模板、训练目标或输入预算。
`train.liger.fused_linear_ce: true` 为显式启用；依赖安装入口为 `.[train,fused-ce]`。
默认不开启，未适配模型不静默降级。详见 [配置](config_reference.md#trainliger)。
2026-09-19 配置已迁移；下文历史数值是 CE-only 验收，不包含新增 RMSNorm/SwiGLU 的性能结论。

单卡真实固定 batch 的显存/步时已接近 LF 使用的 Liger 模型执行路径。
**这不是完整 LF/Shaft 四卡 pipeline 配对验收，不能据此宣布正式 8K 训练可恢复。**

## 为什么需要这一改动

原 Shaft CE 虽然分块，但模型前向仍生成完整词表 logits，反向再创建完整 logits 梯度。
交接中的 OOM 就发生在 `zeros_like(logits)`，单次申请 28.33 GiB。Liger 将投影与 CE 融合，
无需保留这种形状的完整张量。模型参数只有 0.8B 并不能限制大词表输出的大小。

LF 本地参考入口为 `llamafactory/model/model_utils/liger_kernel.py`，其 Qwen3.5 分支调用
`apply_liger_kernel_to_qwen3_5`；融合 loss 实现在 `liger_kernel/transformers/model/qwen3_5.py`
和 `liger_kernel/ops/fused_linear_cross_entropy.py`。

Shaft 接入只保留两处职责：模型 policy 在被请求时执行 backbone 并交给 loss callable；训练层调用
Liger 并复用原 shift/mask/权重/分母。普通监督使用 sum，加权监督使用 none，DDP 平均补偿仍在 Trainer。
未修改 HF 类名、参数布局、持久化格式；推理/显式 logits 请求仍走上游 forward。

## 固定 batch 配对

- 同一 worker-1 A800 80GB GPU 0，按先后顺序运行，未使用其他正在工作的 GPU。
- 同一原始 Qwen3.5-0.8B 权重，852,985,920 参数全部训练；FP32 参数、BF16 autocast、SDPA、
  non-reentrant GC、fused AdamW、clip 1、每卡 batch 4，像素预算 500,000–2,000,000，max length 15,360。
- 两边均使用 **Shaft 原生模板和 collator**，直接重放相同处理后张量，不把 LF 模板差异混入算子比较。
- LF 参考臂调用其使用的 Liger 全模型 patch（含 RMSNorm/MLP 等），但使用 Shaft 环境的
  Transformers 5.10.1；不是重新运行原 Transformers 5.6.0 的完整 LF Trainer。
- Shaft 臂经过 `ShaftSFTTrainer.compute_loss`。此次单卡计时不含数据准备、DDP 通信、保存和评测。
- 各进程使用独立 `/tmp` 编译缓存，allocator 均为 expandable segments；没有复用热缓存隐藏启动成本。
- 每步同步 CUDA、重置峰值，测量 forward/backward/clip/optimizer；allocated 包含优化器状态。
  reserved 单独记录；本轮没有 NVML 采样，不能与历史 NVML 数值直接相减。
- 普通 batch 为最初 4 条真实记录，6 步诊断使用常数 LR 6e-5，**不是原 warmup 调度**。
  长尾 batch 使用原 stress 集前 4 条（覆盖该缓存最大长度），8 步保留 cosine 8000 / warmup 800。
  两组内部两臂调度一致；不得把普通短测用作原训练轨迹复现。

| 固定 batch | 执行路径 | 峰值 allocated GiB | 峰值 reserved GiB | 末三步平均秒 |
|---|---|---:|---:|---:|
| 普通，B=4，L=3791 | Shaft fused CE | 14.9996 | 15.7188 | 1.8761 |
| 同上 | LF 使用的 Liger 模型路径 | 14.9669 | 15.2891 | 1.9319 |
| 长尾，B=4，L=13214 | Shaft fused CE | 24.0123 | 25.2891 | 6.9472 |
| 同上 | LF 使用的 Liger 模型路径 | 23.2135 | 24.1563 | 6.8895 |

普通/长尾 allocated 比例分别为 100.22% / 103.44%，末三步吞吐比例分别约为 102.98% / 99.17%。
这些**局部指标**符合交接建议的 ≤110% 显存、≥90% 吞吐门槛；短窗口不等于完整稳态吞吐。
两组均无 OOM。未降低 batch、像素或序列上限；长尾实际长度不是配置上限 15360 的完整覆盖。

逐步时间（秒，包含所有编译慢步）：

```text
Shaft ordinary: 89.628, 1.883, 1.876, 1.875, 1.876, 1.876
Liger ordinary: 93.218, 1.936, 1.933, 1.932, 1.932, 1.932
Shaft long:     94.052, 8.922, 6.938, 6.935, 6.940, 6.959, 6.942, 6.940
Liger long:    103.399, 6.887, 6.869, 6.877, 6.886, 6.892, 6.889, 6.887
```

输入身份：

```text
ordinary rows: page-000000, page-000001, page-000002, page-000003
batch SHA256: 9e5b880b3a9f67e05181a76dd98a862895b8f0d8b82af0f4503ae107f456bf41
supervised tokens: 6247; pixel_values: [22664,1536]

long rows: page-032821, page-032822, page-032823, page-065882
batch SHA256: 036d1f9324ef65109d2ef7045ca149d47b1ae7e428309046109276358e0d1dde
supervised tokens: 42341; pixel_values: [29336,1536]
```

本地逐步 loss/grad norm、时间、allocated/reserved、命令脚本和日志位于
`temp/fused-ce-validation-20260918/`。该目录是一次性验证产物，不是正式训练入口。

## 数值与兼容性

- CPU focused 回归：599 passed，0 skipped，包括配置、SFT 装配、loss、模型注册、checkpoint，
  真实微型 Qwen3.5 图文模型的微批/GA 对照及 HF 保存/原生类重载。
  收口追加 head-only 拒绝测试后，`test_linear_ce.py` 的 12 项再次全部通过。
  另跑 config loader / RLHF config / offline KD / OPD 的兼容回归，全部通过；这些域未启用融合路径。
- CUDA 单卡：11 passed，覆盖 FP32/BF16、tied/untied、token 权重、GC、微批分组，及两步真实
  HF Trainer/Accelerate BF16 train loop（LM-head forward hook 禁止物化完整 logits）。
- 四卡 NCCL：1 passed，包含加权/非加权、不同 rank 监督 token 数、GA=2、一个 rank 全 ignore，
  全参数梯度对照通过。该项验证分布式数值合同，不是四卡真实模型性能 benchmark。
- FP32 的 loss、梯度和 AdamW 更新有严格对照。BF16 下归约及权重累积舍入可导致少数近零梯度变号，
  Adam 首步近似 sign(g)，会产生约 2×LR 的坐标差异；不能宣称逐 bit 或所有坐标更新一致。
  BF16 gate 检查 loss、梯度逐元素容差、梯度相对 L2 <1%，及显著梯度位置的更新；未用整体
  2×LR 参数容差掩盖问题。真实模型多步短测两臂 loss 接近，但不是严格轨迹等价证明。
- 原普通 logits loss 仍供未适配模型、显式输出和评测使用；融合开关不是这些路径的显存保证。
- 收口检查明确拒绝 backbone 不参与求导的 head-only 组合，避免 Liger 0.8.3 sum 路径对该组合的
  梯度限制；没有为了该低频模式增加自研补丁或额外后端。
- 没有修改共享 venv。Liger 通过 `uv pip --target` 在临时目录隔离验证。

## 剩余验收

完整四卡真实模型 pipeline 的同批次多步配对、更多长尾/配置长度边界、包含数据准备的长期吞吐，
以及 BF16 多步优化轨迹还需单独验收。此次没有启动或恢复正式 8000-step 训练。
