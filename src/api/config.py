"""
StockM v1.0 - Phase 9, Lesson 3
API Configuration & Settings Management
========================================

Env-driven configuration with dev / test / prod profiles, structured logging
setup, and API versioning. Follows the repo's existing conventions: env vars
(matching .env.example) + the existing configs/logging_config.yaml.

Design: a single ``Settings`` object (frozen, immutable) is built ONCE from the
environment and shared across the app. It is NOT a global mutable - changing
config means changing the environment and restarting. This makes every run
reproducible: the same env => the same settings => the same behavior (ties to
Phase 8 Lesson 13's reproducibility contract).

Environment profiles
--------------------
    STOCKM_ENV=development  -> reload on, debug logging, 1 worker, CORS open
    STOCKM_ENV=testing      -> test mode, debug logging, no file logging
    STOCKM_ENV=production   -> reload off, INFO logging, N workers, CORS locked

Env vars (see .env.example; all optional - sensible defaults):
    STOCKM_ENV        development | testing | production (default: development)
    API_HOST          bind host        (default 0.0.0.0)
    API_PORT          bind port        (default 8000)
    API_WORKERS       uvicorn workers  (default 1 dev / 4 prod)
    API_LOG_LEVEL     info | debug ... (default INFO)
    LOG_LEVEL         root log level   (default INFO)
    LOG_FORMAT        json | console   (default console dev / json prod)
    LOG_DIR           log directory    (default logs)
    API_KEY           if set, require this key for protected endpoints (Lesson 14)

No new dependencies: uses stdlib ``os`` + the existing ``pydantic`` for the
frozen model. ``pydantic-settings`` was intentionally NOT added (keep the dep
surface minimal); env parsing is explicit and small.
"""
from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("stockm.api.config")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGING_CONFIG_PATH = PROJECT_ROOT / "configs" / "logging_config.yaml"

Environment = Literal["development", "testing", "production"]


class Settings(BaseModel):
    """Immutable application settings, bound from the environment.

    Frozen via ``model_config`` so no handler mutates a shared settings object
    mid-flight. Built once at import time via :func:`get_settings`.
    """

    model_config = {"frozen": True}

    # --- Identity / versioning ---
    api_title: str = "StockM Prediction API"
    api_version: str = "1.0.0"

    # --- Environment profile ---
    environment: Environment = "development"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = True  # dev only; production overrides to False

    # --- Logging ---
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"
    log_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "logs")

    # --- Prediction defaults (passed to the service layer) ---
    default_split: str = "test"       # the held-out unseen split
    default_threshold: float = 0.0    # signal threshold on predicted return

    # --- Security (Lesson 14 wires enforcement; config lives here) ---
    api_key: str | None = None        # if set, protected endpoints require it
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- Feature flags ---
    enable_metrics: bool = True       # expose GET /metrics (Lesson 7)
    enable_request_id: bool = True    # middleware adds X-Request-ID (Lesson 9)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_testing(self) -> bool:
        return self.environment == "testing"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


def _get_env(name: str, default: str | None = None) -> str | None:
    """Read an env var, allowing an optional .env file (no python-dotenv dep).

    A .env file in the project root is parsed line-by-line for KEY=VALUE pairs
    and used as a FALLBACK for vars not already in the real environment. Real
    env vars always win (so Docker/k8s overrides take precedence).
    """
    val = os.environ.get(name)
    if val is not None:
        return val
    return _DOTENV_CACHE.get(name, default)


# Minimal .env parser (no python-dotenv dependency). Cached at import time.
_DOTENV_CACHE: dict[str, str] = {}


def _load_dotenv() -> None:
    """Populate _DOTENV_CACHE from project-root .env if present."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        _DOTENV_CACHE[key.strip()] = value.strip().strip('"').strip("'")


_load_dotenv()


def _build_settings() -> Settings:
    """Construct Settings from the environment + profile defaults."""
    env = (_get_env("STOCKM_ENV") or "development").lower()
    if env not in ("development", "testing", "production"):
        env = "development"

    # Profile-driven defaults (overridable by explicit env vars).
    if env == "production":
        workers = int(_get_env("API_WORKERS") or "4")
        reload = False
        log_format: Literal["json", "console"] = "json"
        cors = ["https://stockm.example.com"]  # locked down in prod
    elif env == "testing":
        workers = 1
        reload = False
        log_format = "console"
        cors = ["*"]
    else:  # development
        workers = int(_get_env("API_WORKERS") or "1")
        reload = True
        log_format = "console"
        cors = ["*"]

    log_dir = Path(_get_env("LOG_DIR") or str(PROJECT_ROOT / "logs"))
    return Settings(
        environment=env,  # type: ignore[arg-type]
        host=_get_env("API_HOST") or "0.0.0.0",
        port=int(_get_env("API_PORT") or "8000"),
        workers=workers,
        reload=reload,
        log_level=(_get_env("LOG_LEVEL") or _get_env("API_LOG_LEVEL") or "INFO").upper(),
        log_format=log_format,  # type: ignore[arg-type]
        log_dir=log_dir,
        api_key=_get_env("API_KEY") or None,
        cors_origins=cors,
    )


# Singleton: built once. Re-importing returns the same object (reproducible).
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings (built lazily on first call).

    A function (not a module-level constant) so tests can override via
    ``api.config._settings = test_settings`` without touching real env vars.
    """
    global _settings
    if _settings is None:
        _settings = _build_settings()
    return _settings


def reset_settings() -> None:
    """Clear the cached settings (tests use this to force a re-read)."""
    global _settings
    _settings = None


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(settings: Settings) -> None:
    """Configure structured logging from the existing configs/logging_config.yaml.

    Reuses the project's logging_config.yaml (so the API's logs land in the same
    logs/stockM.log with the same formatters as the rest of StockM). In
    production we force JSON structured logs (machine-parseable); in dev,
    human-readable console format. The log dir is created if missing.

    Idempotent: safe to call multiple times (e.g. in tests).
    """
    log_dir = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    if not LOGGING_CONFIG_PATH.exists():
        # Fallback: a minimal stdlib config if the YAML is absent.
        _configure_basic_logging(settings)
        return

    cfg = yaml.safe_load(LOGGING_CONFIG_PATH.read_text(encoding="utf-8"))

    # Apply the profile: set levels + format per environment.
    level = settings.log_level
    for logger_name in ("stockM", "stockM.api"):
        if logger_name in cfg.get("loggers", {}):
            cfg["loggers"][logger_name]["level"] = level
    # Point file handlers at the configured log dir + choose format.
    fmt = "json" if settings.log_format == "json" else "console"
    for handler in cfg.get("handlers", {}).values():
        handler["formatter"] = fmt
        fn = handler.get("filename")
        if fn:
            # The YAML uses relative paths (logs/...); resolve under log_dir.
            handler["filename"] = str(log_dir / Path(fn).name)

    try:
        logging.config.dictConfig(cfg)
    except (ValueError, ImportError, TypeError) as e:
        # The YAML may reference optional libs (e.g. structlog) that aren't
        # installed. NEVER let logging setup crash the server — fall back to a
        # stdlib basicConfig so logs still flow. The error is logged to stderr.
        _configure_basic_logging(settings)
        logging.getLogger("stockm.api.config").warning(
            "logging_config.yaml could not be applied (%s); using stdlib fallback.", e,
        )
        return
    logger.info(
        "logging configured: env=%s level=%s format=%s dir=%s",
        settings.environment, level, settings.log_format, log_dir,
    )


def _configure_basic_logging(settings: Settings) -> None:
    """Minimal stdlib logging fallback (no external deps, always works)."""
    log_dir = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.handlers.RotatingFileHandler(
                log_dir / "stockM.log", maxBytes=10_485_760, backupCount=5,
            ),
        ],
        force=True,  # replace any prior basicConfig (idempotent)
    )
