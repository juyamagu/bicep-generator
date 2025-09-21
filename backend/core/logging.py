import logging
import os
from datetime import datetime
from backend.core.config import DEBUG_LOG

logger = logging.getLogger("bicep-generator")
logger.setLevel(logging.DEBUG)

_log_level = logging.DEBUG if DEBUG_LOG else logging.INFO
_has_console = any(
    isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers
)
if not _has_console:
    _console = logging.StreamHandler()
    _console.setLevel(_log_level)
    _console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(funcName)s] %(message)s"))
    logger.addHandler(_console)
else:
    for _h in logger.handlers:
        if isinstance(_h, logging.StreamHandler) and not isinstance(_h, logging.FileHandler):
            _h.setLevel(_log_level)

_log_filename = datetime.now().strftime("%Y%m%d.log")
_log_file_path = os.getenv("APP_LOG_FILE", os.path.join(os.path.dirname(__file__), "..", "logs", _log_filename))
try:
    os.makedirs(os.path.dirname(_log_file_path), exist_ok=True)
except Exception:
    pass

if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    _file = logging.FileHandler(_log_file_path, encoding="utf-8")
    _file.setLevel(logging.DEBUG)
    _file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s:%(funcName)s] %(message)s"))
    logger.addHandler(_file)

logger.propagate = False

__all__ = ["logger"]
