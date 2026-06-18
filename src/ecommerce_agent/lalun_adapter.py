"""Adapter for the local LALUN research model.

The original LALUN code is a research training/evaluation package. To avoid
mixing its dependency stack with the LangGraph app, this adapter calls a small
English inference script in a separate Conda environment.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import get_settings


def _utf8_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


@dataclass
class LALUNStatus:
    root: str
    available: bool
    chinese_available: bool
    trained_models: list[str]
    distillation_model_found: bool
    english_inference_script_found: bool
    chinese_inference_script_found: bool
    chinese_model_found: bool
    chinese_tokenizer_found: bool
    lalun_python_found: bool
    note: str


def inspect_lalun(lalun_root: Path | None = None) -> dict:
    settings = get_settings()
    project_root = Path(__file__).resolve().parents[2]
    root = lalun_root or settings.lalun_root
    output_dir = root / "code" / "output"
    distillation_dir = root / "code" / "pretrained_model" / "distillation"
    english_inference_script = project_root / "scripts" / "lalun_infer_english.py"
    chinese_inference_script = project_root / "scripts" / "lalun_infer_chinese.py"

    trained_models = []
    if output_dir.exists():
        for model_file in output_dir.glob("*/*/model/model.safetensors"):
            trained_models.append(str(model_file.parent))

    distillation_model_found = (distillation_dir / "pytorch_model.bin").exists()
    english_inference_script_found = english_inference_script.exists()
    chinese_inference_script_found = chinese_inference_script.exists()
    chinese_model_found = (settings.lalun_chinese_model_dir / "model.safetensors").exists()
    chinese_tokenizer_found = (settings.lalun_chinese_tokenizer_dir / "vocab.txt").exists()
    lalun_python_found = settings.lalun_python.exists()
    available = (
        root.exists()
        and bool(trained_models)
        and distillation_model_found
        and english_inference_script_found
        and lalun_python_found
    )
    chinese_available = (
        root.exists()
        and chinese_inference_script_found
        and chinese_model_found
        and chinese_tokenizer_found
        and lalun_python_found
    )

    note = (
        "LALUN English wrapper and Chinese experimental fine-tuned model are available."
        if available and chinese_available
        else "LALUN is partially available; DeepSeek/rule sentiment remains the stable fallback."
    )

    return asdict(
        LALUNStatus(
            root=str(root),
            available=available,
            chinese_available=chinese_available,
            trained_models=trained_models,
            distillation_model_found=distillation_model_found,
            english_inference_script_found=english_inference_script_found,
            chinese_inference_script_found=chinese_inference_script_found,
            chinese_model_found=chinese_model_found,
            chinese_tokenizer_found=chinese_tokenizer_found,
            lalun_python_found=lalun_python_found,
            note=note,
        )
    )


class LALUNAdapter:
    """Subprocess wrapper around LALUN English ASTE inference."""

    def __init__(self, lalun_root: Path | None = None):
        self.settings = get_settings()
        self.project_root = Path(__file__).resolve().parents[2]
        self.lalun_root = lalun_root or self.settings.lalun_root
        self.inference_script = self.project_root / "scripts" / "lalun_infer_english.py"
        self.chinese_inference_script = self.project_root / "scripts" / "lalun_infer_chinese.py"
        self.status = inspect_lalun(self.lalun_root)

    def can_run(self) -> bool:
        return bool(self.status["available"])

    def can_run_chinese(self) -> bool:
        return bool(self.status["chinese_available"])

    def analyze_english(self, text: str, timeout: int = 180, dataset: str = "14res") -> dict:
        """Analyze one English sentence and return LALUN triplets."""
        if not self.can_run():
            return {
                "enabled": False,
                "input": text,
                "error": "LALUN runtime is not fully available.",
                "status": self.status,
            }

        command = [
            str(self.settings.lalun_python),
            str(self.inference_script),
            "--text",
            text,
            "--lalun-root",
            str(self.lalun_root),
            "--dataset",
            dataset,
        ]
        completed = subprocess.run(
            command,
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_utf8_subprocess_env(),
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            return {
                "enabled": False,
                "input": text,
                "error": completed.stderr.strip() or completed.stdout.strip(),
                "status": self.status,
            }

        json_line = next(
            (line for line in completed.stdout.splitlines() if line.strip().startswith("{")),
            "",
        )
        if not json_line:
            return {
                "enabled": False,
                "input": text,
                "error": "LALUN subprocess did not return JSON.",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "status": self.status,
            }

        result = json.loads(json_line)
        result["status"] = self.status
        return result

    def analyze_chinese_experimental(self, text: str, timeout: int = 180) -> dict:
        """Analyze one Chinese sentence with the fine-tuned experimental LALUN model."""
        if not self.can_run_chinese():
            return {
                "enabled": False,
                "input": text,
                "error": "Chinese LALUN runtime is not fully available.",
                "status": self.status,
            }

        command = [
            str(self.settings.lalun_python),
            str(self.chinese_inference_script),
            "--text",
            text,
            "--lalun-root",
            str(self.lalun_root),
            "--model-dir",
            str(self.settings.lalun_chinese_model_dir),
            "--tokenizer-dir",
            str(self.settings.lalun_chinese_tokenizer_dir),
        ]
        completed = subprocess.run(
            command,
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_utf8_subprocess_env(),
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            return {
                "enabled": False,
                "input": text,
                "error": completed.stderr.strip() or completed.stdout.strip(),
                "status": self.status,
            }

        json_line = next(
            (line for line in completed.stdout.splitlines() if line.strip().startswith("{")),
            "",
        )
        if not json_line:
            return {
                "enabled": False,
                "input": text,
                "error": "Chinese LALUN subprocess did not return JSON.",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "status": self.status,
            }

        result = json.loads(json_line)
        result["status"] = self.status
        return result
