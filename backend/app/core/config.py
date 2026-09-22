from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOXTUNE_")

    data_dir: Path = ROOT / "data"
    database_url: str = ""

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'voxtune.db'}"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def registry_dir(self) -> Path:
        return self.data_dir / "registry"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.datasets_dir, self.runs_dir, self.registry_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()


def model_source(hf_id: str) -> str:
    """Prefer a local mirror in data/models_cache (see scripts/fetch_model.py) over the hub."""
    local = settings.data_dir / "models_cache" / hf_id.replace("/", "--")
    return str(local) if (local / "COMPLETE").exists() else hf_id
