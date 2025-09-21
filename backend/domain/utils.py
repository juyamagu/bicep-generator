import logging
from typing import Optional
import requests  # type: ignore
import re

try:
    from bs4 import BeautifulSoup
    from bs4.element import Tag, Comment, NavigableString
except Exception:  # pragma: no cover
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
    try:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/*, application/json, application/xml, */*;q=0.1",
        }
        with requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as resp:
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if not any(t in content_type for t in ("text", "html", "json", "xml")):
                logger.debug("fetch_url_content: unsupported content-type %s for %s", content_type, url)
                return ""
            parts = []
            read_bytes = 0
            for chunk in resp.iter_content(chunk_size=8192, decode_unicode=True):
                if not chunk:
                    continue
                if isinstance(chunk, bytes):
                    chunk = chunk.decode(resp.encoding or "utf-8", errors="replace")
                chunk_len = len(chunk)
                if read_bytes + chunk_len > max_bytes:
                    parts.append(chunk[: max_bytes - read_bytes])
                    break
                parts.append(chunk)
                read_bytes += chunk_len
            text = "".join(parts)
            if query_selector and BeautifulSoup is not None and "html" in content_type:
                try:
                    soup = BeautifulSoup(text, "html.parser")
                    el = soup.select_one(query_selector)
                    if not el:
                        logger.debug("fetch_url_content: selector %s not found in %s", query_selector, url)
                        return ""
                    if compressed:
                        for tag in el.find_all(True):
                            attrs = getattr(tag, "attrs", None)
                            if not isinstance(attrs, dict):
                                continue
                            PRESERVE_ATTRS = ()
                            for attr in list(attrs.keys()):
                                if attr not in PRESERVE_ATTRS:
                                    attrs.pop(attr, None)
                        if Comment is not None:
                            comments = [s for s in el.find_all(string=True) if isinstance(s, Comment)]
                            for c in comments:
                                c.extract()
                        if NavigableString is not None:
                            for s in el.find_all(string=True):
                                if isinstance(s, NavigableString):
                                    parent = s.parent
                                    if parent and parent.name not in ("pre", "code", "script", "style"):
                                        new = re.sub(r"\s+", " ", str(s))
                                        s.replace_with(new)
                    inner = el.decode_contents()
                    if len(inner) > max_bytes:
                        return inner[:max_bytes]
                    return inner
                except Exception as exc:  # pragma: no cover
                    logger.debug("fetch_url_content: error parsing HTML for selector %s: %s", query_selector, exc)
                    return ""
            if compressed and BeautifulSoup is not None and "html" in content_type:
                try:
                    soup = BeautifulSoup(text, "html.parser")
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
                except Exception as exc:  # pragma: no cover
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
