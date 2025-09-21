from __future__ import annotations

import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class BicepCodeBlock:
    text: str
    start_line: int
    end_line: int
    kind: Optional[str] = None


_QUICK_TOP_LEVEL_START_RE = re.compile(
    r"^\s*(?:@|metadata\b|targetScope\b|type\b|func\b|param\b|var\b|resource\b|module\b|output\b)",
    re.IGNORECASE,
)

_QUOTED_RE = re.compile(r"('(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\")")

_FIRST_KEYWORD_RE = re.compile(
    r"^\s*(metadata|targetScope|type|func|param|var|resource|module|output)\b",
    re.IGNORECASE,
)

_CANONICAL_KIND = {
    "metadata": "metadata",
    "targetscope": "targetScope",
    "type": "type",
    "func": "func",
    "param": "param",
    "var": "var",
    "resource": "resource",
    "module": "module",
    "output": "output",
}


def _remove_quoted_parts(s: str) -> str:
    return _QUOTED_RE.sub("''", s)


def parse_bicep_blocks(text: str) -> List[BicepCodeBlock]:
    lines = text.splitlines()
    blocks: List[BicepCodeBlock] = []
    current: List[str] = []
    brace_depth = 0
    current_start_idx: Optional[int] = None

    def finalize_current() -> None:
        nonlocal current
        nonlocal current_start_idx
        if not current:
            return
        block_text = "\n".join(current).strip("\n \t")
        if block_text:
            assert current_start_idx is not None
            start_line = current_start_idx + 1
            end_line = current_start_idx + len(current)
            kind: Optional[str] = None
            for ln in current:
                s = ln.strip()
                if not s:
                    continue
                if s.startswith("@"):
                    continue
                m = _FIRST_KEYWORD_RE.match(ln)
                if m:
                    key = m.group(1).lower()
                    kind = _CANONICAL_KIND.get(key, key)
                break
            blocks.append(BicepCodeBlock(text=block_text, start_line=start_line, end_line=end_line, kind=kind))
        current = []
        current_start_idx = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "":
            if brace_depth == 0 and current:
                current_has_top_level = any(_FIRST_KEYWORD_RE.match(ln) for ln in current)
                if current_has_top_level:
                    finalize_current()
                else:
                    current.append(line)
            else:
                if current:
                    current.append(line)
            continue
        is_top_level = _QUICK_TOP_LEVEL_START_RE.match(line)
        is_decorator = stripped.startswith("@")
        if is_top_level and not is_decorator and current and brace_depth == 0:
            current_has_top_level = any(_FIRST_KEYWORD_RE.match(ln) for ln in current)
            if current_has_top_level:
                finalize_current()
                current_start_idx = idx
                current.append(line)
            else:
                current.append(line)
        elif not current and is_top_level:
            current_start_idx = idx
            current.append(line)
        else:
            if not current:
                current_start_idx = idx
                current.append(line)
            else:
                current.append(line)
        cleaned = _remove_quoted_parts(line)
        brace_depth += cleaned.count("{") - cleaned.count("}")
    if current:
        finalize_current()
    return blocks


__all__ = ["BicepCodeBlock", "parse_bicep_blocks"]
