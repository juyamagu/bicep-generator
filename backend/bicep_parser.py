"""
A lightweight Bicep parser helper.

Provides parse_bicep_blocks(text: str) -> list[str], which splits a Bicep file
into top-level blocks (including preceding decorators). The parser is
intended for simple structural splitting (by blank lines and brace depth)
and is resilient to braces appearing inside string literals.

This is not a full parser for the Bicep language, but it's sufficient for
splitting common top-level constructs (metadata, targetScope, param, var,
resource, module, type, func, output) into separate text blocks.
"""

from __future__ import annotations

import re
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class BicepCodeBlock:
    """Represents a top-level Bicep block.

    Attributes:
        text: The block text (preserves internal formatting, trimmed).
        start_line: 1-based index of the first line of the block in the
            original input.
        end_line: 1-based index of the last line of the block in the
            original input.
    """

    text: str
    start_line: int
    end_line: int
    # One of: metadata, targetScope, type, func, param, var, resource, module, output
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
    """Remove quoted string literals from a line so braces inside strings
    won't affect brace depth counting.
    """
    return _QUOTED_RE.sub("''", s)


def parse_bicep_blocks(text: str) -> List[BicepCodeBlock]:
    """
    Split Bicep source text into top-level blocks and return a list of
    strings, each representing a block. Blocks preserve internal
    formatting but are trimmed of leading/trailing blank lines.

    Heuristics used:
    - Top-level blocks are typically separated by blank lines when not
      inside braces.
    - Brace depth is tracked using '{' and '}' occurrences outside of
      string literals.
    - A block starts when a line begins with a decorator '@' or one of
      the known top-level keywords (metadata, targetScope, type, func,
      param, var, resource, module, output).

    This function is intentionally conservative: it does not attempt to
    validate syntax, and it will return reasonable blocks for typical
    Bicep files used in this project.
    """
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
            # compute 1-based start/end line numbers
            assert current_start_idx is not None
            start_line = current_start_idx + 1
            end_line = current_start_idx + len(current)
            # Determine kind by scanning for the first non-decorator, non-empty line
            kind: Optional[str] = None
            for ln in current:
                s = ln.strip()
                if not s:
                    continue
                if s.startswith("@"):
                    # decorator line -> skip
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

        # Blank line handling: if not inside braces, blank line separates blocks
        # However, if the current block so far represents only decorators
        # (possibly with multi-line decorator arguments) and does not yet
        # include a top-level keyword (param/var/resource/etc.), we should
        # not finalize here so the decorator remains attached to its
        # following declaration even across blank lines.
        if stripped == "":
            if brace_depth == 0 and current:
                # Determine whether current already contains a top-level keyword
                current_has_top_level = any(_FIRST_KEYWORD_RE.match(ln) for ln in current)
                if current_has_top_level:
                    finalize_current()
                else:
                    # keep decorator (and its argument lines) until the
                    # next top-level declaration arrives; preserve blank line
                    current.append(line)
            else:
                if current:
                    current.append(line)
            continue

        # Start new block / attach to current depending on context.
        is_top_level = _QUICK_TOP_LEVEL_START_RE.match(line)
        is_decorator = stripped.startswith("@")

        if is_top_level and not is_decorator and current and brace_depth == 0:
            # If the current block already contains a top-level keyword,
            # it is a completed block and we should finalize it before
            # starting a new one. If the current block so far is only
            # decorators (possibly with multi-line args), then the current
            # non-decorator top-level line belongs to the same block and
            # should be appended.
            current_has_top_level = any(_FIRST_KEYWORD_RE.match(ln) for ln in current)
            if current_has_top_level:
                finalize_current()
                # start new block
                current_start_idx = idx
                current.append(line)
            else:
                # attach this top-level declaration to the existing
                # decorator block
                current.append(line)
        elif not current and is_top_level:
            # starting new block: record its start index
            current_start_idx = idx
            current.append(line)
        else:
            # If current is empty but line does not look like a top-level
            # start, we still start a block (covers e.g. stray comments or
            # non-standard constructs). Otherwise append to current.
            if not current:
                current_start_idx = idx
                current.append(line)
            else:
                current.append(line)

        # Update brace depth using a version of the line without quoted parts
        cleaned = _remove_quoted_parts(line)
        brace_depth += cleaned.count("{") - cleaned.count("}")

        # If brace depth returns to zero and the next non-empty line starts a
        # new top-level construct, we could finalize here, but we already
        # rely on blank-line separators in the input which is the most
        # common and predictable delimiter. Still, if brace depth is zero and
        # the current line itself looks like a single-line construct, we can
        # finalize on the next blank line or EOF.

    # End of file: finalize any remaining block
    if current:
        finalize_current()

    return blocks
