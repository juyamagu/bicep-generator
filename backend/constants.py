from enum import Enum


class Phase(str, Enum):
    HEARING = "hearing"
    SUMMARIZING = "summarizing"
    CODE_GENERATING = "code_generating"
    CODE_VALIDATING = "code_validating"
    COMPLETED = "completed"


AUTO_PROGRESS_PHASES = {Phase.SUMMARIZING, Phase.CODE_GENERATING, Phase.CODE_VALIDATING, Phase.COMPLETED}

# Default language settings
DEFAULT_LANGUAGE = "en"
