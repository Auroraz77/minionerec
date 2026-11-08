#!/usr/bin/env python3
"""Select RL by full validation NDCG@10, then evaluate SFT and RL identically."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from industrial_rl_utils import SFT, DATA, CATEGORY, write_json, KS


def write_report(comparison, output, report):
    selected = comparison["selection"]["selected"]
    baseline, rl = comparison["sft_test"], comparison["sft_rl_test"]
    lines = ["# MiniOneRec：SFT 与 SFT+RL 对比报告", "", "日期：2026-11-07。", "",
        f"最终 RL 模型：`{selected['model']}`。候选依据完整验证集 NDCG@10 选择，测试集只在选定后用于结果对比。", "",
        f"已完成两组 RL 训练及完整测试。最终模型的 HR@10 为 {rl['HR@10']*100:.4f}%"
        f"（SFT 为 {baseline['HR@10']*100:.4f}%，相对变化 {comparison['delta']['HR@10']['relative_percent']:+.2f}%）；"
        f"NDCG@10 为 {rl['NDCG@10']:.6f}"
        f"（SFT 为 {baseline['NDCG@10']:.6f}，相对变化 {comparison['delta']['NDCG@10']['relative_percent']:+.2f}%）。", "",
        "## 测试结果", "",
        "两种模型使用同一组 4,533 条测试记录、相同提示词和商品 SID 目录，均为确定性 50-beam 受限解码，max_new_tokens=8、length_penalty=0。", "",
        "| K | SFT HR | SFT+RL HR | HR 变化（百分点） | SFT NDCG | SFT+RL NDCG | NDCG 相对变化 |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for k in KS:
        hr, ndcg = f"HR@{k}", f"NDCG@{k}"
        lines.append(f"| {k} | {baseline[hr]*100:.4f}% | {rl[hr]*100:.4f}% | "
                     f"{comparison['delta'][hr]['absolute']*100:+.4f} | {baseline[ndcg]:.6f} | "
                     f"{rl[ndcg]:.6f} | {comparison['delta'][ndcg]['relative_percent']:+.2f}% |")
    lines.extend(["", f"HR@10 相对变化：{comparison['delta']['HR@10']['relative_percent']:+.2f}%；"
                  f"NDCG@10 相对变化：{comparison['delta']['NDCG@10']['relative_percent']:+.2f}%。", "",
        f"SFT 非法/重复候选：{baseline['invalid_sid_count']}/{baseline['duplicate_sid_count']}；"
        f"RL 非法/重复候选：{rl['invalid_sid_count']}/{rl['duplicate_sid_count']}。", "",
        "## 验证集选模", "",
        "训练阶段使用固定种子抽取的 1,024 条验证样本，每 150 次更新计算 50-beam NDCG@10，"
        "连续 4 次未改善早停；每组回载其最佳检查点。随后在全部 4,532 条验证记录上比较候选，"
        "选模记录先于测试评估写入 selection.json。", "",
        "| 配置 | 学习率 | KL 系数 | 语义前缀奖励权重 | 最佳检查点 | 完整验证 HR@10 | 完整验证 NDCG@10 |",
        "|---|---:|---:|---:|---|---:|---:|"])
    baseline_validation = output / "sft_valid/metrics.json"
    if baseline_validation.exists():
        baseline_valid = json.loads(baseline_validation.read_text())
        lines.append(f"| SFT 基线 | — | — | — | 原 SFT 权重 | "
                     f"{baseline_valid['HR@10']*100:.4f}% | {baseline_valid['NDCG@10']:.6f} |")
    for candidate in comparison["selection"]["candidates"]:
        config = json.loads((Path(candidate["trial"])/"run_config.json").read_text())
        summary = candidate["training_summary"]
        lines.append(f"| {Path(candidate['trial']).name} | {config['learning_rate']:g} | {config['beta']:g} | "
                     f"{config['prefix_weight']:g} | {Path(summary['best_checkpoint']).name} | "
                     f"{candidate['validation']['HR@10']*100:.4f}% | {candidate['validation']['NDCG@10']:.6f} |")
    lines.extend(["", "| 配置 | 停止步数 | 实际训练轮数 | 最佳模型选择 |",
                  "|---|---:|---:|---|"])
    for candidate in comparison["selection"]["candidates"]:
        state = json.loads((Path(candidate["trial"])/"trainer_state.json").read_text())
        lines.append(f"| {Path(candidate['trial']).name} | {state['global_step']} | {state['epoch']:.4f} | "
                     f"{Path(state['best_model_checkpoint']).name} |")
    lines.extend(["", "## RL 训练方法", "",
        "输入固定为上一次已核验的 Qwen2.5-0.5B SFT 模型；训练数据为原训练划分的全部 36,259 条序列记录。"
        "采用 MiniOneRec ReReTrainer 的 GRPO 目标：16 个受限 beam 候选/组、组内标准化优势、"
        "正确命中奖励与排名惩罚，以及相对固定 SFT 参考模型的 KL 正则。第二组额外使用 0.1 权重的连续 SID 前缀奖励。", "",
        "两组均为 BF16 全参数训练、micro batch 128、梯度累积 4，即每次更新 512 个候选、32 条训练记录；"
        "最多 2 轮，随机种子 42。未将正确目标强行注入生成候选。奖励逐条读取训练 target_sid，"
        "避免原脚本共享历史字典的目标覆盖和训练/验证标签混用。RL 训练提示词与最终评估一致；"
        "SFT 基线仍使用同一原始 SFT 权重及评估协议。", "",
        "训练进程从完整保存的检查点继续时，保留优化器、调度器和随机状态。受限解码 GPU 索引写入及"
        "短响应 log-softmax 计算进行了批处理，分别核验了输出一致性和梯度一致性。", "",
        "## 变化的不确定性", "",
        "以下为按用户分组、2,000 次配对 bootstrap 的 95% 区间，差值均为 SFT+RL 减 SFT。"
        "区间仅反映当前测试集抽样波动，不包含不同训练随机种子的波动。", "",
        "| 指标 | 绝对差值 | 95% 区间 |", "|---|---:|---|"])
    for key, row in comparison["paired_uncertainty"].items():
        lo, hi = row["paired_user_cluster_bootstrap_95ci"]
        lines.append(f"| {key} | {row['difference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    main_interval = comparison["paired_uncertainty"]["NDCG@10"]["paired_user_cluster_bootstrap_95ci"]
    if main_interval[0] > 0:
        lines.extend(["", "NDCG@10 的配对区间下界大于 0，本次测试集上的提升具有抽样统计支持；"
                      "仍需多随机种子复验来判断训练稳定性。"])
    elif main_interval[1] < 0:
        lines.extend(["", "NDCG@10 的配对区间整体小于 0，本次 RL 在该测试指标上退化；"
                      "不建议仅凭验证集改善替换 SFT 模型。"])
    else:
        lines.extend(["", "NDCG@10 的配对区间包含 0，当前差异尚不足以排除测试抽样波动；"
                      "需扩大评估或进行多随机种子复验。"])
    hr50_interval = comparison["paired_uncertainty"]["HR@50"]["paired_user_cluster_bootstrap_95ci"]
    if hr50_interval[0] <= 0 <= hr50_interval[1]:
        lines.extend(["", "HR@50 的区间包含 0，其点估计变化尚不能排除抽样波动。"])
    audit_path = output / "audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get("all_checks_passed"):
            lines.extend(["", "## 独立复核", "",
                "已从保存的预测逐条重算所有 HR/NDCG，核对原始测试行的目标与顺序，"
                "并遍历两种模型各 226,650 个测试候选，非法和重复候选均为 0。"
                "同时独立复核了两组完整验证预测。", "",
                "原 SFT 权重 SHA256 保持不变，本次 SFT 重测与原报告在保存精度内完全一致。"
                "两组 RL 导出权重均与各自最佳检查点相同、与 SFT 不同，全部 290 个权重张量数值有限。"
                "测试结果写入时间晚于 selection.json 中的选模决定。", "",
                f"核验记录：`{audit_path}`；复核入口：`scripts/audit_industrial_rl_comparison.py`。"])
    lines.extend(["", "## 交付与结果边界", "",
        f"- 原 SFT 模型：`{comparison['sft_model']}`。",
        f"- 选中 RL 模型：`{comparison['rl_model']}`。",
        f"- 完整对比 JSON：`{output/'comparison.json'}`。",
        f"- 选模记录：`{output/'selection.json'}`。",
        f"- SFT 测试预测：`{output/'sft_test/predictions.json'}`。",
        f"- RL 测试预测：`{output/'rl_test/predictions.json'}`。", "",
        "本实验继续使用原有 SID：3,686 个商品对应 3,670 个唯一 SID，存在商品共用 SID 的情况，"
        "所以表中为 SID 级命中指标。两个配置都只运行了 seed=42；结果不能视为多随机种子均值，"
        "也不代表已穷尽 RL 超参数的最优值。", "",
        "## 复现入口", "",
        "在项目根目录使用 `/home/liujinhao/xxl/.venvs/minionerec-sft/bin/python`。"
        "以下命令从原 SFT 权重开始训练，复现时应为输出目录换一个新名称，避免覆盖现有结果。", "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=1 /home/liujinhao/xxl/.venvs/minionerec-sft/bin/python scripts/train_industrial_rl.py --output-dir outputs/rl_rank_reproduce --learning-rate 1e-5 --beta 0.001 --epochs 2",
        "CUDA_VISIBLE_DEVICES=0 /home/liujinhao/xxl/.venvs/minionerec-sft/bin/python scripts/train_industrial_rl.py --output-dir outputs/rl_dense_reproduce --learning-rate 3e-6 --beta 0.01 --prefix-weight 0.1 --epochs 2",
        "/home/liujinhao/xxl/.venvs/minionerec-sft/bin/python scripts/compare_industrial_rl.py --trials outputs/rl_rank_reproduce outputs/rl_dense_reproduce --output-dir outputs/rl_comparison_reproduce --report reports/SFT_VS_SFT_RL_reproduce.md",
        "```", "",
        "完整超参数、固定验证子集行号及 SFT 权重 SHA256 见各训练目录的 run_config.json；"
        "实际停止步数和最佳检查点见 trainer_state.json 与 train_summary.json。"
        "训练存在检查点续训，train_summary 中 runtime_seconds 为最后一次启动后的耗时，不能当作全程耗时。", ""])
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def evaluate_checkpoint(checkpoint, split, output, device):
    # Isolate CUDA state and use each GPU for one independent evaluation job.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt = output/"evaluation_config.json"
    if receipt.exists() and (output/"metrics.json").exists() and (output/"predictions.json").exists():
        config = json.loads(receipt.read_text())
        digest = hashlib.sha256((Path(checkpoint)/"model.safetensors").read_bytes()).hexdigest()
        if config.get("split") == split and config.get("limit") == 0 and config.get("model_sha256") == digest:
            print("Reusing verified complete evaluation:", output, flush=True)
            return json.loads((output/"metrics.json").read_text())
    command = [sys.executable, str(Path(__file__).with_name("industrial_rl_utils.py")),
               "--model", str(checkpoint), "--split", split, "--output-dir", str(output)]
    with (output/"evaluate.log").open("w") as log:
        subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(device),
                                    "PYTHONUNBUFFERED": "1"}, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    return json.loads((output/"metrics.json").read_text())


def paired_intervals(sft_predictions, rl_predictions):
    rng = np.random.default_rng(42)
    result = {}
    assert len(sft_predictions) == len(rl_predictions)
    with (DATA/f"test/{CATEGORY}_5_2016-10-2018-11.csv").open() as stream:
        users = [row["user_id"] for row in csv.DictReader(stream)]
    user_ids = {user: index for index, user in enumerate(sorted(set(users)))}
    group_ids = np.array([user_ids[user] for user in users])
    counts = np.bincount(group_ids)
    for k in [10, 50]:
        values = []
        for sft, rl in zip(sft_predictions, rl_predictions):
            assert sft["row_id"] == rl["row_id"] and sft["output"] == rl["output"]
            pair = []
            for row in [sft, rl]:
                rank = row["predict"].index(row["output"]) + 1 if row["output"] in row["predict"] else k+1
                pair.append((float(rank <= k), 1/math.log2(rank+1) if rank <= k else 0.0))
            values.append(np.array(pair[1])-np.array(pair[0]))
        differences = np.array(values)
        # Resample users, preserving dependence between a user's test records.
        totals = np.stack([np.bincount(group_ids, weights=differences[:, column])
                           for column in range(2)], axis=1)
        means = []
        for _ in range(2000):
            sampled = rng.integers(0, len(user_ids), len(user_ids))
            means.append(totals[sampled].sum(0)/counts[sampled].sum())
        means = np.array(means)
        for column, name in enumerate(["HR", "NDCG"]):
            result[f"{name}@{k}"] = {"difference": float(differences[:, column].mean()),
                "paired_user_cluster_bootstrap_95ci": [float(x) for x in np.quantile(means[:, column], [.025, .975])]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", nargs="+", required=True)
    parser.add_argument("--output-dir", default="outputs/rl_comparison")
    parser.add_argument("--devices", nargs="+", default=["0", "1"])
    parser.add_argument("--report", default=str(Path(__file__).resolve().parents[1]/"reports/SFT_VS_SFT_RL_20261107.md"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        for offset in range(0, len(args.trials), len(args.devices)):
            batch = args.trials[offset:offset+len(args.devices)]
            futures = [executor.submit(evaluate_checkpoint, Path(trial)/"final_checkpoint", "valid",
                       output/f"{Path(trial).name}_valid", device)
                       for trial, device in zip(batch, args.devices)]
            for trial, future in zip(batch, futures):
                trial = Path(trial)
                summary = json.loads((trial/"train_summary.json").read_text())
                candidates.append({"trial": str(trial), "model": str(trial/"final_checkpoint"),
                                   "validation": future.result(), "training_summary": summary})
    selected = max(candidates, key=lambda row: (row["validation"]["NDCG@10"], row["validation"]["HR@10"]))
    selection = {"selection_metric": "full validation NDCG@10, tie-break HR@10",
                 "selected": selected, "candidates": candidates, "selected_at_unix": time.time(),
                 "model_sha256": hashlib.sha256((Path(selected["model"])/"model.safetensors").read_bytes()).hexdigest()}
    # Persist the decision before the test split is loaded or any test metric is computed.
    write_json(output / "selection.json", selection)

    if len(args.devices) >= 2:
        with ThreadPoolExecutor(max_workers=2) as executor:
            baseline_future = executor.submit(evaluate_checkpoint, SFT, "test", output/"sft_test", args.devices[0])
            rl_future = executor.submit(evaluate_checkpoint, selected["model"], "test", output/"rl_test", args.devices[1])
            baseline, rl = baseline_future.result(), rl_future.result()
    else:
        baseline = evaluate_checkpoint(SFT, "test", output/"sft_test", args.devices[0])
        rl = evaluate_checkpoint(selected["model"], "test", output/"rl_test", args.devices[0])
    comparison = {"sft_model": str(SFT), "rl_model": selected["model"],
                  "selection": selection, "sft_test": baseline, "sft_rl_test": rl,
                  "delta": {f"{name}@{k}": {"absolute": rl[f"{name}@{k}"]-baseline[f"{name}@{k}"],
                    "relative_percent": (rl[f"{name}@{k}"]/baseline[f"{name}@{k}"]-1)*100}
                    for name in ["HR", "NDCG"] for k in KS}}
    sft_predictions = json.loads((output/"sft_test/predictions.json").read_text())
    rl_predictions = json.loads((output/"rl_test/predictions.json").read_text())
    comparison["paired_uncertainty"] = paired_intervals(sft_predictions, rl_predictions)
    write_json(output / "comparison.json", comparison)
    write_report(comparison, output, args.report)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
