import logging
from os import getenv
from pathlib import Path

from flask_caching import Cache, Flask
from gaw_qc.app_factory import create_app
from gaw_qc.config.app_config import AppConfig
from sqlalchemy import Engine, create_engine
import logging


settings = AppConfig()
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    encoding="utf-8",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO,
)


def start() -> Flask:
    logger.debug(f"Settings: {settings}")
    logger.debug("Creating cache")
    cache = Cache(
        config={  # note that filesystem cache doesn't work on systems with ephemeral filesystems like Heroku
            "CACHE_TYPE": settings.cache_type,
            "CACHE_DIR": settings.cache_dir,
            "CACHE_THRESHOLD": settings.cache_threshold,  # should be equal to maximum number of users on the app at a single time
        }
    )

    db_path: Path = settings.db_path
    assets_path: Path = settings.assets_path
    theme: str = settings.theme
    title: str = settings.title
    logger.info(f"Using database at {db_path}")
    prefix = "//" if db_path.is_absolute() else "/"
    logger.debug(f"Creating engine for database at {db_path}")
    engine = create_engine(f"sqlite://{prefix}{str(db_path)}", pool_size=10)
    dash = create_app(engine, cache, assets_path, theme, title)
    application = dash.server
    return application


app = start()
