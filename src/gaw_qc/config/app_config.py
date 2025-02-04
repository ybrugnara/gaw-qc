from pathlib import Path
from typing import Any, Callable, Literal, Set
from dash_bootstrap_components import themes
from pydantic import (
    AliasChoices,
    AmqpDsn,
    BaseModel,
    Field,
    ImportString,
    PostgresDsn,
    RedisDsn,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    db_path: Path = Path("/data/test.db")
    assets_path: Path = Path("/assets")
    app_port: int = 8000
    cache_threshold: int = 50
    cache_dir: Path = Path("/tmp")
    cache_type: Literal["filesystem", "redis"] = "filesystem"
    theme: str = themes.MATERIA
    title: str = "GAW-QC"
