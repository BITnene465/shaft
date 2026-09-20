# 训练框架语义对照：2026-09-18

范围：Shaft `f868c97`、本地 LLaMA-Factory `436d26bc`（2026-04-12），以及本日读取的
ms-swift main / 4.6.0.dev0 文档。后者不是固定发布版本，不能外推到所有历史版本。
这是实现合同与小型数值实验，不是生产模型吞吐或效果榜单。

## Loss：token mean 与 microbatch mean 不同

设微批 token loss 总和为 S，有效 token 数为 N；忽略 prompt/padding，并使用 shifted labels。

| 路径 | 一个累积窗口的主损失 |
| --- | --- |
| Shaft，无额外权重 | sum(S) / sum(N)，生产参数跨 DP rank 汇总分母 |
| 本地 LF，多模态普通 SFT | 微批均值 S/N 再按累积步数平均；processor 存在时关闭 loss kwargs |
| ms-swift，普通 SFT 默认模板 | 使用 num_items_in_batch，将模型均值换算为累积窗口 token 均值 |

Shaft 适合 token-level next-token prediction；同一有效 batch 的划分不应改变目标。
LF 这条路径在各微批 N 相同时与其相同，N 不同则会改变相对权重，不能视为普遍更合理。
ms-swift 文档中 average_tokens_across_devices 默认 False；跨 rank token 数不同时，需显式对齐。
上述不覆盖 DFT、RLHF、label smoothing、特殊模板或自定义 compute_loss。

自定义 loss_scale 也须单独对齐：Shaft 为 sum(w*loss)/sum(w)，所读 ms-swift 自定义逐 token
加权路径为 sum(w*loss)/N。将全部权重翻倍，前者不变，后者翻倍。二者是不同的权重合同，
不是仅修改变量名就能兼容。任务采样比例也不等于监督 token 比例。

来源：

- 本仓库 `src/shaft/training/sft_trainer.py`、`training/loss.py`、`pipeline/training_args.py`。
- [LF 对应版本 Trainer](https://github.com/hiyouga/LLaMA-Factory/blob/436d26bc/src/llamafactory/train/sft/trainer.py)。
- [ms-swift 默认模板 loss](https://github.com/modelscope/ms-swift/blob/main/swift/template/base.py)。
- [ms-swift SFT Trainer](https://github.com/modelscope/ms-swift/blob/main/swift/trainers/seq2seq_trainer.py)。
- [ms-swift 参数](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html)。

## 模板：具体文本与监督边界

使用本地 Qwen3.5-0.8B tokenizer，user 为 `返回 JSON：{"ok":true}`，assistant 为 `{"ok":true}`，
无 system/tools；实测模板层编码如下。省略共同的 user 部分，`\n` 代表真实换行。

| 路径 | 不计 loss 的 assistant 前缀 | 受监督的 target |
| --- | --- | --- |
| Shaft qwen35vl，enable_thinking=False | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | `{"ok":true}<\|im_end\|>` |
| LF qwen3_5_nothink | `<\|im_start\|>assistant\n` | `{"ok":true}<\|im_end\|>\n` |
| LF qwen3_5，该版本默认 reasoning 模板 | `<\|im_start\|>assistant\n` | `<think>\n\n</think>\n\n{"ok":true}<\|im_end\|>\n` |

target token IDs：

```text
Shaft:      [4754, 547, 763, 1802, 92, 248046]
LF nothink:[4754, 547, 763, 1802, 92, 248046, 198]
LF qwen3_5:[248068, 271, 248069, 271, 4754, 547, 763, 1802, 92, 248046, 198]
```

这说明同为“不写推理内容”的样本，前缀和监督边界仍可能不同。Shaft 的做法与该模型 HF
non-thinking generation prefix 一致，不因此判错；不要在推理时改用无 thinking stub 的另一套前缀。
该文本例子不代替含图像的完整 processor/collator 对齐。

截断亦不同：若编码后的 prefix=950、target=200、max_length=1024，Shaft 保留950前缀，
target只剩74，截断时不强行补EOS；LF infer_seqlen 分配824前缀和200 target。
应测量实际数据的截断比例，不能为对齐参考框架而破坏图像token和聊天结构。

## 性能：确定的结构成本与待测的速度分开

Shaft CE 分块并在 backward 重算，可以减少完整 FP32 log-softmax 的驻留，但仍生成完整
logits 与同形梯度，并且不是 fused linear + CE。LF/ms-swift 提供可选 Liger 路径；
ms-swift 还具有 logits_to_keep 相关路径，但适用性取决于模型和 loss 配置。
不能由“参考框架支持”推断某次训练已启用或一定更快。

本地 Qwen3.5-0.8B vocab_size=248320。若 padded B=4、L=8000、logits 为 BF16：
仅 B*L*V*2 字节约14.80 GiB；同形 logits gradient 也约14.80 GiB，尚未计其他激活、权重和优化器。
这是尺寸推算，不是测得的峰值显存；FP32 logits 会翻倍。

此外，Shaft 每批在线进行 processor/tokenization，普通 AdamW 构建没有显式启用 fused=True。
这些需要分别以数据等待、LM head/CE、optimizer step 的 profiler 时间和峰值显存衡量，
不应把当前没有同硬件实测的差距宣称为具体吞吐倍数。

## CI 证据范围

本轮新增 required CPU numerics：真实微型 Qwen3.5、合成视觉输入、FP32/eager/CPU reference kernel，
比较 padded/单样本有效 logits、三种 BS/GA 划分的全参数梯度、AdamW 参数与状态；
有加权、无权重、完整与不完整窗口，且不允许 skip。参见 [testing.md](testing.md)。

它不代替真实 tokenizer/processor、BF16、GPU kernel、DDP/ZeRO 或模型效果验收。
现有 GPU isolation 用例仍未进入 required gate，不能用 CPU 绿灯外推。
