"""Read local SFT progress without loading the model or using the GPU."""

import ast
import json
import os
import re
from pathlib import Path


root = Path(__file__).resolve().parents[1]
output = root / "outputs/qwen2.5_0.5b_industrial_sft"
status = {"output_dir": str(output)}
pid_path = output / "run.pid"
if pid_path.exists():
    status["pid"] = int(pid_path.read_text())
    try:
        os.kill(status["pid"], 0)
        status["process_exists"] = True
    except ProcessLookupError:
        status["process_exists"] = False

exit_path = output / "exit_code"
status["exit_code"] = int(exit_path.read_text()) if exit_path.exists() else None
log_path = output / "train.log"
if log_path.exists():
    with log_path.open("rb") as stream:
        stream.seek(max(0, log_path.stat().st_size - 131072))
        tail = stream.read().decode(errors="replace")
    lines = [line.strip() for line in re.split(r"[\r\n]+", tail) if line.strip()]
    metrics = []
    for line in lines:
        if line.startswith("{") and line.endswith("}"):
            try:
                metrics.append(ast.literal_eval(line))
            except (SyntaxError, ValueError):
                pass
    status["recent_metrics"] = metrics[-2:]
    progress = [line for line in lines if re.search(r"\d+/6240\s", line)]
    if progress:
        status["training_progress"] = progress[-1]
    status["log_tail"] = lines[-3:]
    status["log_bytes"] = log_path.stat().st_size

checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
status["checkpoints"] = [p.name for p in checkpoints]
if checkpoints:
    state_path = checkpoints[-1] / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        status["saved_step"] = state.get("global_step")
        status["best_eval_loss"] = state.get("best_metric")
        status["best_checkpoint"] = state.get("best_model_checkpoint")
        status["validation_history"] = [
            entry for entry in state.get("log_history", []) if "eval_loss" in entry
        ]

for name in ["train_results.json", "eval_final_results.json"]:
    path = output / name
    if path.exists():
        status[name] = json.loads(path.read_text())
status["final_model_exists"] = (output / "final_checkpoint/model.safetensors").exists()
print(json.dumps(status, ensure_ascii=False, indent=2))
