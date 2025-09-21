import logging
from typing import Optional
import requests  # type: ignore
import re

try:
    from bs4 import BeautifulSoup
    from bs4.element import Tag, Comment, NavigableString
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore
    Tag = None  # type: ignore
    Comment = None  # type: ignore
    NavigableString = None  # type: ignore

logger = logging.getLogger(__name__)


USER_AGENT = "bicep-generator/1.0"


def fetch_url_content(
    url: str,
    timeout: float = 10.0,
    max_bytes: int = 200_000,
    query_selector: Optional[str] = None,
    compressed: bool = False,
) -> str:
    """
    Fetch text content from the specified URL and return it as a string.

    Behavior:
    - Times out after the specified number of seconds (timeout)
    - Returns an empty string if the response Content-Type is not text-like (text/html, application/json, etc.)
    - Truncates content that exceeds max_bytes
    - On error, returns an empty string and emits debug logs

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        max_bytes: Maximum number of bytes to read (truncated if exceeded)
        query_selector: CSS selector for an HTML element (e.g. "div#content", "p.class_name")

    Returns:
        The fetched text ("" on failure)
    """
    try:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/*, application/json, application/xml, */*;q=0.1",
        }
        with requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as resp:
            resp.raise_for_status()

            content_type = (resp.headers.get("Content-Type") or "").lower()
            # Do not fetch non-text content
            if not any(t in content_type for t in ("text", "html", "json", "xml")):
                logger.debug("fetch_url_content: unsupported content-type %s for %s", content_type, url)
                return ""

            parts = []
            read_bytes = 0
            # With decode_unicode=True the iter_content yields text chunks directly
            for chunk in resp.iter_content(chunk_size=8192, decode_unicode=True):
                if not chunk:
                    continue
                if isinstance(chunk, bytes):
                    # If chunk is bytes, decode using the response encoding
                    chunk = chunk.decode(resp.encoding or "utf-8", errors="replace")
                chunk_len = len(chunk)
                if read_bytes + chunk_len > max_bytes:
                    parts.append(chunk[: max_bytes - read_bytes])
                    break
                parts.append(chunk)
                read_bytes += chunk_len

            text = "".join(parts)

            # If a selector is requested and BeautifulSoup is available and the content is HTML
            if query_selector and BeautifulSoup is not None and "html" in content_type:
                try:
                    soup = BeautifulSoup(text, "html.parser")
                    el = soup.select_one(query_selector)
                    if not el:
                        logger.debug("fetch_url_content: selector %s not found in %s", query_selector, url)
                        return ""
                    # Optionally compress the selected element's content
                    if compressed:
                        # Remove structural attributes and comments from the selected subtree
                        for tag in el.find_all(True):
                            # manipulate attrs dict safely via getattr to satisfy static analysis
                            attrs = getattr(tag, "attrs", None)
                            if not isinstance(attrs, dict):
                                continue
                            # remove class, id, and data-* attributes
                            PRESERVE_ATTRS = ()
                            for attr in list(attrs.keys()):
                                if attr not in PRESERVE_ATTRS:
                                    attrs.pop(attr, None)
                        # remove comments
                        if Comment is not None:
                            comments = [s for s in el.find_all(string=True) if isinstance(s, Comment)]
                            for c in comments:
                                c.extract()
                        # normalize whitespace in text nodes, skipping pre/code/style/script
                        if NavigableString is not None:
                            for s in el.find_all(string=True):
                                if isinstance(s, NavigableString):
                                    parent = s.parent
                                    if parent and parent.name not in ("pre", "code", "script", "style"):
                                        new = re.sub(r"\s+", " ", str(s))
                                        s.replace_with(new)

                    # get inner HTML of the element
                    inner = el.decode_contents()
                    # ensure max_bytes is respected
                    if len(inner) > max_bytes:
                        return inner[:max_bytes]
                    return inner
                except Exception as exc:  # pragma: no cover - parsing errors
                    logger.debug("fetch_url_content: error parsing HTML for selector %s: %s", query_selector, exc)
                    return ""

            # If no selector but compressed requested for full page
            if compressed and BeautifulSoup is not None and "html" in content_type:
                try:
                    soup = BeautifulSoup(text, "html.parser")
                    # remove attributes and comments site-wide
                    for tag in soup.find_all(True):
                        attrs = getattr(tag, "attrs", None)
                        if not isinstance(attrs, dict):
                            continue
                        for attr in list(attrs.keys()):
                            if attr == "class" or attr == "id" or attr.startswith("data-"):
                                attrs.pop(attr, None)
                    if Comment is not None:
                        comments = [s for s in soup.find_all(string=True) if isinstance(s, Comment)]
                        for c in comments:
                            c.extract()
                    if NavigableString is not None:
                        for s in soup.find_all(string=True):
                            if isinstance(s, NavigableString):
                                parent = s.parent
                                if parent and parent.name not in ("pre", "code", "script", "style"):
                                    new = re.sub(r"\s+", " ", str(s))
                                    s.replace_with(new)
                    cleaned = str(soup)
                    if len(cleaned) > max_bytes:
                        return cleaned[:max_bytes]
                    return cleaned
                except Exception as exc:  # pragma: no cover - parsing errors
                    logger.debug("fetch_url_content: error compressing HTML for %s: %s", url, exc)
                    return ""

            return text

    except requests.RequestException as exc:
        logger.debug("fetch_url_content: request error for %s: %s", url, exc)
        return ""
    except Exception:
        logger.exception("fetch_url_content: unexpected error for %s", url)
        return ""


__all__ = ["fetch_url_content"]
