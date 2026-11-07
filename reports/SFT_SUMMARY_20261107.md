# MiniOneRec SFT 训练总结报告

报告日期：2026-11-07。项目目录：`/home/liujinhao/xxl/MiniOneRec`。

## 1. 结果概览

已完成 Qwen2.5-0.5B 在 Amazon Reviews 2018 / Industrial_and_Scientific 数据上的全参数 SFT，并完成全部 4,533 条测试样本的受限 beam search 评估。

- 最佳模型：`outputs/qwen2.5-0.5b-industrial-sft/final_checkpoint/`。
- 最佳检查点：`checkpoint-1248`，对应第 4 轮，验证 loss 为 **1.5112187**。
- 训练在第 5.5 轮、1716 个优化步骤后触发早停，正常退出；计划上限为 10 轮。
- 测试结果：**HR@10 = 14.8467%，NDCG@10 = 0.105515；HR@50 = 24.6415%，NDCG@50 = 0.126963**。
- 独立复核了全部 **226,650** 个候选 SID：非法候选数为 **0**，每条样本内重复候选数为 **0**。

这是一组已经完成、可复核的 SFT 实验结果。本次仅运行了一组正式训练配置，按验证 loss 选模，尚不足以证明 HR/NDCG 已达到该模型和数据集的最佳水平。

## 2. 项目与数据

项目来源：`https://github.com/AkaliKong/MiniOneRec.git`。本地基准提交：`0cbd94de9ea58c59a6b05d4d1dbda7edf8d5b3b1`。

本次使用仓库随附的预处理 CSV、商品元数据和 SID 索引，文件名后缀为 `5_2016-10-2018-11`。没有重新从原始评论执行过滤与切分；以下规模以实际加载的文件为准。

| 数据部分 | 样本数 | 该部分的用户数 | 历史长度 |
|---|---:|---:|---|
| 训练集 | 36,259 | 7,034 | 1–10 个商品 |
| 验证集 | 4,532 | 2,011 | 1–10 个商品 |
| 测试集 | 4,533 | 1,987 | 1–10 个商品 |

各部分用户数分别统计，不能相加作为总用户数。完整检查了三个划分中目标商品及历史商品的 SID 映射，发现的映射错误数为 0。测试预测文件的行数、目标值及顺序也与测试 CSV 一致。

商品目录包含 **3,686 个商品、3,670 个唯一 SID**，每个 SID 由 3 个 token 组成，共使用 560 个 SID token。存在 15 组 SID 碰撞，涉及 31 个商品，即部分商品共享同一个 SID。

SFT 使用三类任务，合并后每轮共有 **79,834** 条训练样本：

| 训练任务 | 样本数 | 学习目标 |
|---|---:|---|
| SID 序列推荐 | 36,259 | 根据历史 SID 预测下一商品 SID |
| 商品与 SID 对齐 | 7,316 | 商品标题与 SID 双向映射 |
| 融合序列推荐 | 36,259 | 根据历史 SID 预测下一商品标题 |

验证 loss 仅针对 SID 序列推荐任务计算。训练 loss 来自混合任务，两者的数值不能直接比较。

## 3. 模型、环境与训练配置

基础模型为 **Qwen/Qwen2.5-0.5B 基础版**，从 ModelScope 下载到本地。添加 SID 词表后，保存模型的参数量为 **494,291,712**，约 4.94 亿；采用全参数训练，精度为 BF16。

| 配置 | 实际值 |
|---|---|
| GPU | NVIDIA A100 80GB × 1、NVIDIA A800 80GB × 1 |
| 分布式方式 | 2 卡 DDP |
| Python | 3.11.16 |
| PyTorch / CUDA | 2.6.0+cu124 / CUDA 12.4 |
| Transformers / Datasets | 4.57.1 / 4.2.0 |
| Accelerate / PyArrow | 1.10.1 / 21.0.0 |
| 优化器 | AdamW（PyTorch 实现） |
| 初始学习率 | 1e-4 |
| 学习率调度 | 线性衰减，warmup 20 步 |
| 每卡 micro batch | 8 |
| 梯度累积 | 16 步 |
| 有效总 batch | 8 × 16 × 2 = 256 |
| 最大序列长度 | 512 token |
| 最大训练轮数 | 10 |
| 验证与保存间隔 | 156 个优化步骤，约半轮 |
| 最佳模型选择 | 最低验证 loss |
| 早停规则 | 连续 3 次验证未刷新最佳 loss |
| 随机种子 | 42 |

环境位于 `/home/liujinhao/xxl/.venvs/minionerec-sft`。完整依赖快照见 [sft_environment_20261107.txt](sft_environment_20261107.txt)。学习率调度由保存的 TrainingArguments 确认为线性调度，代码中的辅助余弦函数未用于本次训练。

## 4. 训练过程与模型选择

| 训练轮次 | 验证 loss |
|---:|---:|
| 0.5 | 2.436689 |
| 1.0 | 2.017059 |
| 1.5 | 1.808024 |
| 2.0 | 1.675317 |
| 2.5 | 1.601829 |
| 3.0 | 1.553021 |
| 3.5 | 1.540685 |
| **4.0** | **1.511219** |
| 4.5 | 1.532662 |
| 5.0 | 1.521985 |
| 5.5 | 1.559299 |

验证 loss 在前 4 轮持续下降，第 4 轮达到最低点。随后 3 次验证均未改善，因此第 5.5 轮触发早停，Trainer 回载了第 4 轮的最佳权重并导出到 `final_checkpoint`。

日志记录的训练运行时长为 **2,295.4223 秒，约 38 分 15 秒**，包含训练期间的验证和保存。测试解码约 **5 分 26 秒**。此前回复中的约 70 分钟指本次任务从环境准备到训练、评估完成的总耗时，不是纯训练耗时。

已计算最佳 checkpoint 与最终导出权重的 SHA-256，两者完全一致：

```text
f975932b2c238511c44df7d204efb76cce5247db6e975e5f27d429a68e79e075
```

加载日志中的 `lm_head.weight` 缺失提示对应共享输入/输出 embedding 的保存方式：配置为 `tie_word_embeddings=true`。最终模型已成功执行完整测试集推理。

## 5. 测试集结果与口径

评估使用完整测试集，每条样本预测一个目标 SID。候选来自商品 SID 目录上的受限 beam search，`num_beams=50`、`num_return_sequences=50`、`do_sample=False`、`length_penalty=0.0`、`max_new_tokens=8`，评估 batch 为 16。评估没有采用随机抽取负样本的排序协议。

| K | 命中样本数 / 4,533 | HR@K | NDCG@K |
|---:|---:|---:|---:|
| 1 | 329 | 7.2579% | 0.072579 |
| 3 | 452 | 9.9713% | 0.088081 |
| 5 | 535 | 11.8023% | 0.095600 |
| 10 | 673 | **14.8467%** | **0.105515** |
| 20 | 840 | 18.5308% | 0.114884 |
| 50 | 1,117 | **24.6415%** | **0.126963** |

HR@K 为目标 SID 出现在前 K 个候选中的样本比例。每条样本只有一个目标，因此 NDCG@K 在命中时记为 `1 / log2(rank + 1)`，未命中记为 0，再对全部样本求平均；rank 从 1 开始。

指标先由仓库 `calc.py` 计算，再依据保存的预测结果独立重算，两者在原汇总精度内一致。本次独立复核遍历了所有候选，包括目标命中之后的候选，确认 226,650 个候选均属于目录，且各样本内 50 个 SID 互不重复。

## 6. 结果边界与后续优化优先级

**当前是 SID 级别评估。** 3,686 个商品仅对应 3,670 个唯一 SID，测试集中有 317 条样本的目标 SID 对应多个商品。因此现有指标不能直接解释为严格区分商品 ID 的推荐指标。后续应消除 SID 碰撞或添加消歧 token，再评估商品级命中。

**本次依据验证 loss 选模。** 没有对多个 checkpoint 的验证 HR/NDCG 进行比较，也没有完成学习率、任务比例和随机种子的系统搜索，因此不能宣称当前 checkpoint 的 HR/NDCG 最优。后续应优先按验证集 NDCG@10 或业务指定指标选模，测试集只用于最终结果报告。

**训练与评估提示词存在措辞差异。** `SidSFTDataset` 使用 “The user has interacted with items ... in chronological order ...”，`EvalSidDataset` 使用 “Can you predict ... given the following chronological interaction history ...”。本次沿用仓库行为，尚未验证这一差异对指标的影响；统一提示模板是后续可直接验证的改进项。

建议后续按以下顺序进行：统一提示模板并处理 SID 碰撞；以验证集 NDCG@10 选择 checkpoint；比较学习率及三类任务比例；对候选配置运行多个随机种子，报告均值和波动，再进行一次最终测试集评估。上述优化尚未执行，本报告表格仅包含本次实际完成的实验。

## 7. 交付文件与复现

以下路径均相对于项目根目录：

| 产物 | 路径 |
|---|---|
| 最终模型与 tokenizer | `outputs/qwen2.5-0.5b-industrial-sft/final_checkpoint/` |
| 最佳训练检查点与优化器状态 | `outputs/qwen2.5-0.5b-industrial-sft/checkpoint-1248/` |
| 训练日志 | `outputs/qwen2.5-0.5b-industrial-sft/console.log` |
| 训练配置摘要 | `outputs/qwen2.5-0.5b-industrial-sft/run_config.json` |
| 全部测试预测 | `outputs/qwen2.5-0.5b-industrial-sft/eval/test_result.json` |
| 指标 JSON / 原始日志 | `outputs/qwen2.5-0.5b-industrial-sft/eval/metrics.json` / `metrics.log` |
| 训练启动脚本 | `sft_qwen25_05b.sh` |
| 独立复核结果、数据及模型校验和 | `reports/sft_industrial_20261107.audit.json` |
| 完整依赖快照 | `reports/sft_environment_20261107.txt` |

本次代码调整包含添加 2 卡训练脚本、移除训练与评估入口中未使用的 `bitsandbytes` 导入，以及忽略模型和输出目录。

重新训练时使用新的输出目录，保留本次结果：

```bash
cd /home/liujinhao/xxl/MiniOneRec
source /home/liujinhao/xxl/.venvs/minionerec-sft/bin/activate
OUTPUT_DIR=./outputs/reproduce-sft-20261107 bash sft_qwen25_05b.sh
```

重新评估本次最终模型：

```bash
cd /home/liujinhao/xxl/MiniOneRec
source /home/liujinhao/xxl/.venvs/minionerec-sft/bin/activate
mkdir -p ./outputs/reproduce-eval-20261107
python evaluate.py \
  --base_model ./outputs/qwen2.5-0.5b-industrial-sft/final_checkpoint \
  --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
  --category Industrial_and_Scientific \
  --test_data_path ./data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv \
  --result_json_data ./outputs/reproduce-eval-20261107/test_result.json \
  --batch_size 16 --num_beams 50 --max_new_tokens 8 --length_penalty 0.0
python calc.py \
  --path ./outputs/reproduce-eval-20261107/test_result.json \
  --item_path ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```
