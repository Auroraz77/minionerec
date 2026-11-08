# MiniOneRec：SFT 与 SFT+RL 对比报告

日期：2026-11-07。

最终 RL 模型：`outputs/rl_rank_lr1e5/final_checkpoint`。候选依据完整验证集 NDCG@10 选择，测试集只在选定后用于结果对比。

已完成两组 RL 训练及完整测试。最终模型的 HR@10 为 15.5306%（SFT 为 14.8467%，相对变化 +4.61%）；NDCG@10 为 0.112451（SFT 为 0.105515，相对变化 +6.57%）。

## 测试结果

两种模型使用同一组 4,533 条测试记录、相同提示词和商品 SID 目录，均为确定性 50-beam 受限解码，max_new_tokens=8、length_penalty=0。

| K | SFT HR | SFT+RL HR | HR 变化（百分点） | SFT NDCG | SFT+RL NDCG | NDCG 相对变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.2579% | 7.7653% | +0.5074 | 0.072579 | 0.077653 | +6.99% |
| 3 | 9.9713% | 11.0964% | +1.1251 | 0.088081 | 0.096677 | +9.76% |
| 5 | 11.8023% | 12.7951% | +0.9927 | 0.095600 | 0.103654 | +8.43% |
| 10 | 14.8467% | 15.5306% | +0.6839 | 0.105515 | 0.112451 | +6.57% |
| 20 | 18.5308% | 19.1705% | +0.6398 | 0.114884 | 0.121627 | +5.87% |
| 50 | 24.6415% | 24.9504% | +0.3088 | 0.126963 | 0.133083 | +4.82% |

HR@10 相对变化：+4.61%；NDCG@10 相对变化：+6.57%。

SFT 非法/重复候选：0/0；RL 非法/重复候选：0/0。

## 验证集选模

训练阶段使用固定种子抽取的 1,024 条验证样本，每 150 次更新计算 50-beam NDCG@10，连续 4 次未改善早停；每组回载其最佳检查点。随后在全部 4,532 条验证记录上比较候选，选模记录先于测试评估写入 selection.json。

| 配置 | 学习率 | KL 系数 | 语义前缀奖励权重 | 最佳检查点 | 完整验证 HR@10 | 完整验证 NDCG@10 |
|---|---:|---:|---:|---|---:|---:|
| SFT 基线 | — | — | — | 原 SFT 权重 | 15.0044% | 0.111741 |
| rl_rank_lr1e5 | 1e-05 | 0.001 | 0 | checkpoint-1050 | 16.1297% | 0.122197 |
| rl_dense_lr3e6 | 3e-06 | 0.01 | 0.1 | checkpoint-1200 | 15.6222% | 0.117265 |

| 配置 | 停止步数 | 实际训练轮数 | 最佳模型选择 |
|---|---:|---:|---|
| rl_rank_lr1e5 | 1650 | 1.4553 | checkpoint-1050 |
| rl_dense_lr3e6 | 1800 | 1.5877 | checkpoint-1200 |

## RL 训练方法

输入固定为上一次已核验的 Qwen2.5-0.5B SFT 模型；训练数据为原训练划分的全部 36,259 条序列记录。采用 MiniOneRec ReReTrainer 的 GRPO 目标：16 个受限 beam 候选/组、组内标准化优势、正确命中奖励与排名惩罚，以及相对固定 SFT 参考模型的 KL 正则。第二组额外使用 0.1 权重的连续 SID 前缀奖励。

两组均为 BF16 全参数训练、micro batch 128、梯度累积 4，即每次更新 512 个候选、32 条训练记录；最多 2 轮，随机种子 42。未将正确目标强行注入生成候选。奖励逐条读取训练 target_sid，避免原脚本共享历史字典的目标覆盖和训练/验证标签混用。RL 训练提示词与最终评估一致；SFT 基线仍使用同一原始 SFT 权重及评估协议。

训练进程从完整保存的检查点继续时，保留优化器、调度器和随机状态。受限解码 GPU 索引写入及短响应 log-softmax 计算进行了批处理，分别核验了输出一致性和梯度一致性。

## 变化的不确定性

以下为按用户分组、2,000 次配对 bootstrap 的 95% 区间，差值均为 SFT+RL 减 SFT。区间仅反映当前测试集抽样波动，不包含不同训练随机种子的波动。

| 指标 | 绝对差值 | 95% 区间 |
|---|---:|---|
| HR@10 | +0.006839 | [+0.001331, +0.012841] |
| NDCG@10 | +0.006935 | [+0.003543, +0.010373] |
| HR@50 | +0.003088 | [-0.004085, +0.010591] |
| NDCG@50 | +0.006121 | [+0.002873, +0.009260] |

NDCG@10 的配对区间下界大于 0，本次测试集上的提升具有抽样统计支持；仍需多随机种子复验来判断训练稳定性。

HR@50 的区间包含 0，其点估计变化尚不能排除抽样波动。

## 独立复核

已从保存的预测逐条重算所有 HR/NDCG，核对原始测试行的目标与顺序，并遍历两种模型各 226,650 个测试候选，非法和重复候选均为 0。同时独立复核了两组完整验证预测。

原 SFT 权重 SHA256 保持不变，本次 SFT 重测与原报告在保存精度内完全一致。两组 RL 导出权重均与各自最佳检查点相同、与 SFT 不同，全部 290 个权重张量数值有限。测试结果写入时间晚于 selection.json 中的选模决定。

核验记录：`outputs/rl_comparison/audit.json`；复核入口：`scripts/audit_industrial_rl_comparison.py`。

## 交付与结果边界

- 原 SFT 模型：`/home/liujinhao/xxl/MiniOneRec/outputs/qwen2.5-0.5b-industrial-sft/final_checkpoint`。
- 选中 RL 模型：`outputs/rl_rank_lr1e5/final_checkpoint`。
- 完整对比 JSON：`outputs/rl_comparison/comparison.json`。
- 选模记录：`outputs/rl_comparison/selection.json`。
- SFT 测试预测：`outputs/rl_comparison/sft_test/predictions.json`。
- RL 测试预测：`outputs/rl_comparison/rl_test/predictions.json`。

本实验继续使用原有 SID：3,686 个商品对应 3,670 个唯一 SID，存在商品共用 SID 的情况，所以表中为 SID 级命中指标。两个配置都只运行了 seed=42；结果不能视为多随机种子均值，也不代表已穷尽 RL 超参数的最优值。

## 复现入口

在项目根目录使用 `/home/liujinhao/xxl/.venvs/minionerec-sft/bin/python`。以下命令从原 SFT 权重开始训练，复现时应为输出目录换一个新名称，避免覆盖现有结果。

```bash
CUDA_VISIBLE_DEVICES=1 /home/liujinhao/xxl/.venvs/minionerec-sft/bin/python scripts/train_industrial_rl.py --output-dir outputs/rl_rank_reproduce --learning-rate 1e-5 --beta 0.001 --epochs 2
CUDA_VISIBLE_DEVICES=0 /home/liujinhao/xxl/.venvs/minionerec-sft/bin/python scripts/train_industrial_rl.py --output-dir outputs/rl_dense_reproduce --learning-rate 3e-6 --beta 0.01 --prefix-weight 0.1 --epochs 2
/home/liujinhao/xxl/.venvs/minionerec-sft/bin/python scripts/compare_industrial_rl.py --trials outputs/rl_rank_reproduce outputs/rl_dense_reproduce --output-dir outputs/rl_comparison_reproduce --report reports/SFT_VS_SFT_RL_reproduce.md
```

完整超参数、固定验证子集行号及 SFT 权重 SHA256 见各训练目录的 run_config.json；实际停止步数和最佳检查点见 trainer_state.json 与 train_summary.json。训练存在检查点续训，train_summary 中 runtime_seconds 为最后一次启动后的耗时，不能当作全程耗时。
