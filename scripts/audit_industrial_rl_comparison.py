#!/usr/bin/env python3
"""Independently verify saved predictions, model lineage and selection order."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from industrial_rl_utils import DATA, CATEGORY, INFO, KS, SFT, write_json


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/rl_comparison")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output_dir)
    comparison = json.loads((output / "comparison.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    assert selection == comparison["selection"]
    expected = max(selection["candidates"], key=lambda row: (
        row["validation"]["NDCG@10"], row["validation"]["HR@10"]))
    assert expected == selection["selected"]
    assert selection["model_sha256"] == digest(Path(expected["model"]) / "model.safetensors")
    catalog = {line.split("\t")[0].strip() for line in INFO.read_text().splitlines()}
    receipt = {"all_checks_passed": False, "catalog_unique_sids": len(catalog),
               "evaluation_checks": {}, "model_checks": {}}
    checks = [(output / "sft_test", "test", SFT),
              (output / "rl_test", "test", Path(expected["model"]))]
    checks.extend((output / f"{Path(row['trial']).name}_valid", "valid", Path(row["model"]))
                  for row in selection["candidates"])
    for folder, split, model in checks:
        config = json.loads((folder / "evaluation_config.json").read_text())
        metrics = json.loads((folder / "metrics.json").read_text())
        predictions = json.loads((folder / "predictions.json").read_text())
        with (DATA / split / f"{CATEGORY}_5_2016-10-2018-11.csv").open() as stream:
            source = list(csv.DictReader(stream))
        assert len(predictions) == len(source) == metrics["samples"]
        assert config["split"] == split and config["limit"] == 0
        assert config["model_sha256"] == digest(model / "model.safetensors")
        if split == "test":
            assert (folder / "metrics.json").stat().st_mtime > selection["selected_at_unix"]
        independent = {f"{name}@{k}": 0.0 for name in ("HR", "NDCG") for k in KS}
        for index, (row, original) in enumerate(zip(predictions, source)):
            assert row["row_id"] == index and row["output"] == original["item_sid"]
            candidates = row["predict"]
            assert len(candidates) == len(set(candidates)) == 50
            assert all(sid in catalog for sid in candidates)
            if row["output"] in candidates:
                rank = candidates.index(row["output"]) + 1
                for k in KS:
                    if rank <= k:
                        independent[f"HR@{k}"] += 1 / len(source)
                        independent[f"NDCG@{k}"] += 1 / (len(source) * math.log2(rank + 1))
        for key, value in independent.items():
            assert math.isclose(value, metrics[key], abs_tol=1e-12), (folder, key)
        assert metrics["invalid_sid_count"] == metrics["duplicate_sid_count"] == 0
        receipt["evaluation_checks"][str(folder)] = {
            "rows": len(source), "candidates_checked": 50 * len(source),
            "all_metrics_independently_recomputed": True,
            "targets_match_source_rows": True, "all_candidates_valid_and_unique": True,
            "model_hash_verified": True,
            "test_metrics_written_after_selection": split == "test"}
    sft_hash = digest(SFT / "model.safetensors")
    assert sft_hash == "f975932b2c238511c44df7d204efb76cce5247db6e975e5f27d429a68e79e075"
    for candidate in selection["candidates"]:
        trial = Path(candidate["trial"])
        state = json.loads((trial / "trainer_state.json").read_text())
        best = Path(state["best_model_checkpoint"]) / "model.safetensors"
        final = Path(candidate["model"]) / "model.safetensors"
        final_hash = digest(final)
        assert final_hash == digest(best) and final_hash != sft_hash
        config = json.loads((trial / "run_config.json").read_text())
        assert config["sft_weights_sha256"] == sft_hash and config["train_rows"] == 36259
        with safe_open(str(final), framework="pt", device="cpu") as tensors:
            names = list(tensors.keys())
            assert all(bool(torch.isfinite(tensors.get_tensor(name)).all()) for name in names)
        receipt["model_checks"][str(trial)] = {
            "sha256": final_hash, "final_equals_best_checkpoint": True,
            "differs_from_original_sft": True, "all_weight_tensors_finite": True,
            "tensor_count": len(names), "stopped_step": state["global_step"],
            "best_checkpoint": state["best_model_checkpoint"]}
    for name in ("HR", "NDCG"):
        for k in KS:
            key = f"{name}@{k}"
            delta = comparison["sft_rl_test"][key] - comparison["sft_test"][key]
            assert math.isclose(delta, comparison["delta"][key]["absolute"], abs_tol=1e-14)
    receipt["sft_sha256"] = sft_hash
    original_metrics = json.loads((SFT.parent / "eval/metrics.json").read_text())
    baseline_differences = {f"{name}@{k}": comparison["sft_test"][f"{name}@{k}"]
                            - original_metrics[f"{name}@{k}"]
                            for name in ("HR", "NDCG") for k in KS}
    receipt["sft_retest_difference_from_original_rounded_metrics"] = baseline_differences
    receipt["data_sha256"] = {
        str(path): digest(path) for path in [INFO] + [
            DATA / split / f"{CATEGORY}_5_2016-10-2018-11.csv"
            for split in ("train", "valid", "test")]}
    receipt["all_checks_passed"] = True
    write_json(output / "audit.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
