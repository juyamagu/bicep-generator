from enum import Enum


class Phase(str, Enum):
    HEARING = "hearing"
    SUMMARIZING = "summarizing"
    CODE_GENERATION = "code_generation"
    CODE_VALIDATION = "code_validation"
    COMPLETED = "completed"


AUTO_PROGRESS_PHASES = {Phase.SUMMARIZING, Phase.CODE_GENERATION, Phase.CODE_VALIDATION}

# Default language settings
DEFAULT_LANGUAGE = "en"
