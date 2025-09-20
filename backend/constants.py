from enum import Enum


class Phase(str, Enum):
    HEARING = "hearing"
    SUMMARIZING = "summarizing"
    CODE_GENERATING = "code_generating"
    CODE_VALIDATING = "code_validating"
    COMPLETED = "completed"


REQUIRE_USER_INPUT_PHASES = {Phase.HEARING}


class Language(str, Enum):
    EN = "en"
    JA = "ja"


DEFAULT_LANGUAGE = Language.EN
