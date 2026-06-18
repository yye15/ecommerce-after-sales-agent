"""Configuration loading for the e-commerce agent project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent


def load_project_env() -> None:
    """Load project-level env files without overwriting already-set values."""
    candidates = [
        PROJECT_ROOT / ".env",
        REPO_ROOT / ".env",
        REPO_ROOT.parent / ".env",
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)


@dataclass(frozen=True)
class Settings:
    llm_api_key: str | None
    llm_model_id: str | None
    llm_base_url: str | None
    llm_timeout: int = 60
    default_temperature: float = 0.2
    lalun_root: Path = Path("external/LALUN/delivery_105")
    lalun_python: Path = Path("python")
    lalun_chinese_model_dir: Path = Path(
        "external/LALUN/delivery_105/code/output/zh_ecommerce/model"
    )
    lalun_chinese_tokenizer_dir: Path = Path("models/hfl_chinese_macbert_base")
    use_lalun_english: bool = True

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_api_key and self.llm_model_id and self.llm_base_url)


def get_settings() -> Settings:
    load_project_env()
    timeout = int(os.getenv("LLM_TIMEOUT", "60"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    lalun_root = Path(os.getenv("LALUN_ROOT", "external/LALUN/delivery_105"))
    lalun_python = Path(os.getenv("LALUN_PYTHON", "python"))
    lalun_chinese_model_dir = Path(
        os.getenv(
            "LALUN_CHINESE_MODEL_DIR",
            "external/LALUN/delivery_105/code/output/zh_ecommerce/model",
        )
    )
    lalun_chinese_tokenizer_dir = Path(
        os.getenv("LALUN_CHINESE_TOKENIZER_DIR", "models/hfl_chinese_macbert_base")
    )
    use_lalun_english = os.getenv("USE_LALUN_ENGLISH", "1").strip().lower() not in {"0", "false", "no"}
    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_model_id=os.getenv("LLM_MODEL_ID"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_timeout=timeout,
        default_temperature=temperature,
        lalun_root=lalun_root,
        lalun_python=lalun_python,
        lalun_chinese_model_dir=lalun_chinese_model_dir,
        lalun_chinese_tokenizer_dir=lalun_chinese_tokenizer_dir,
        use_lalun_english=use_lalun_english,
    )
