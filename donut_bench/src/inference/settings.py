import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from inference.constants import MODEL_ID


class BenchSettings(BaseSettings):
    """Common settings for all bench scripts.

    Priority (highest first): CLI args → env vars (BENCH_ prefix) → defaults.
    """

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        env_prefix="BENCH_",
    )

    model: str = Field(default=MODEL_ID, description="HuggingFace model ID")
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Compute device: cuda or cpu",
    )
    warmup: int = Field(default=2, description="Warmup runs before measurement")
    no_patch: bool = Field(default=False, description="Skip attention mask patch")
    save: str | None = Field(default=None, description="Path to write JSON results")
