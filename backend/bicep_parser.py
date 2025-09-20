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


_QUICK_TOP_LEVEL_START_RE = re.compile(
    r"^\s*(?:@|metadata\b|targetScope\b|type\b|func\b|param\b|var\b|resource\b|module\b|output\b)",
    re.IGNORECASE,
)

_QUOTED_RE = re.compile(r"('(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\")")


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
            blocks.append(BicepCodeBlock(text=block_text, start_line=start_line, end_line=end_line))
        current = []
        current_start_idx = None

    for idx, line in enumerate(lines):
        stripped = line.strip()

        # Blank line handling: if not inside braces, blank line separates blocks
        if stripped == "":
            if brace_depth == 0 and current:
                finalize_current()
            else:
                if current:
                    current.append(line)
            continue

        # Start new block if current is empty and line looks like a top-level start
        # If the incoming line is a top-level start and it's NOT a decorator
        # (i.e. a normal top-level declaration like `var`, `param`, etc.)
        # and we already have a finished top-level block (brace_depth == 0)
        # then finalize the current block so adjacent single-line
        # declarations don't get merged into one block. However, if the
        # current block only contains decorator lines, do not finalize: the
        # decorator should remain attached to the following construct.
        is_top_level = _QUICK_TOP_LEVEL_START_RE.match(line)
        is_decorator = stripped.startswith("@")
        if is_top_level and not is_decorator and current and brace_depth == 0:
            current_has_non_decorator = any((ln.strip() and not ln.strip().startswith("@")) for ln in current)
            if current_has_non_decorator:
                finalize_current()

        if not current and is_top_level:
            # starting new block: record its start index
            current_start_idx = idx
            current.append(line)
        else:
            # If current is empty but line does not look like a top-level
            # start, we still start a block (covers e.g. stray comments or
            # non-standard constructs).
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
