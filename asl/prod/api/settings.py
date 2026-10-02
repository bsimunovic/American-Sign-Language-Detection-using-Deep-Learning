from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings, read from ``ASL_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="ASL_", protected_namespaces=())

    model_dir: str = "artifacts/model"
    top_k: int = 3
    min_confidence: float = 0.0
    smoothing_window: int = 5
    max_image_bytes: int = 5 * 1024 * 1024
    enable_demo: bool = True
    cors_origins: list[str] = []
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
