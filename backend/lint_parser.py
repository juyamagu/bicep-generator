from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, List


@dataclass(slots=True)
class BicepLintMessage:
    path: Path
    line: int
    column: int
    severity: str
    code: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        return f"({self.line},{self.column}) : {self.severity} {self.code} : {self.message}"  # noqa: E501


_LINT_LINE_PATTERN = re.compile(
    r"^(?P<path>.+?)\((?P<line>\d+),(?P<col>\d+)\)\s*:\s*(?P<severity>\w+)\s+"  # path, position, severity
    r"(?P<code>[A-Za-z0-9\-]+):\s+"  # code
    r"(?P<message>.+?)$"  # message (exclude trailing url)
)


def parse_bicep_lint_output(text: str | Iterable[str]) -> List[BicepLintMessage]:
    if not isinstance(text, str):
        lines = list(text)
    else:
        lines = text.splitlines()

    results: List[BicepLintMessage] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _LINT_LINE_PATTERN.match(line)
        if not m:
            continue  # スキップ (必要ならログなど)
        path_str = m.group("path").strip()
        try:
            path = Path(path_str)
        except OSError:
            # パスが異常でも文字列として保持
            path = Path(path_str.replace("\\", "/"))
        results.append(
            BicepLintMessage(
                path=path,
                line=int(m.group("line")),
                column=int(m.group("col")),
                severity=m.group("severity"),
                code=m.group("code"),
                message=m.group("message").rstrip(),
            )
        )
    return results


__all__ = ["BicepLintMessage", "parse_bicep_lint_output"]
