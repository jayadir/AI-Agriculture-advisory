import re
from tavily import TavilyClient

_client = TavilyClient()

_BASE64_RE = re.compile(r"!\[.*?\]\(data:[^)]+\)")
_DATA_URI_RE = re.compile(r"data:[a-zA-Z0-9+/]+;base64,[A-Za-z0-9+/=]+")
_IMG_MD_RE = re.compile(r"!\[.*?\]\(https?://[^)]+\)")
_MIN_USEFUL_LENGTH = 200


def _clean_raw_content(text):
    if not text:
        return ""
    text = _BASE64_RE.sub("", text)
    text = _DATA_URI_RE.sub("", text)
    text = _IMG_MD_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _format_results(raw_results):
    """Convert raw Tavily results into our standard format."""
    formatted = []
    for r in raw_results:
        raw = _clean_raw_content(r.get("raw_content", ""))
        snippet = r.get("content", "")

        if len(raw) >= _MIN_USEFUL_LENGTH:
            content = raw[:4000]
        elif snippet:
            content = snippet
        else:
            continue

        url = r.get("url", "")
        formatted.append({
            "title": r.get("title", ""),
            "url": url,
            "content": content,
        })
    return formatted


def search_with_content(query, max_results=5, search_depth="advanced", include_domains=None):
    """Basic search — optionally restricted to specific domains."""
    try:
        kwargs = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_raw_content": "markdown",
        }
        if include_domains:
            kwargs["include_domains"] = include_domains

        response = _client.search(**kwargs)
        return _format_results(response.get("results", []))
    except Exception as e:
        print(f"Tavily search error: {e}")
        return []


def search_prioritized(query, max_results=5):
    """Search the open web (wide and unrestricted) and return results with content."""
    results = search_with_content(query, max_results=max_results)
    print(f"Tavily: {len(results)} total results")
    
    return results


def format_search_results(results):
    """Format results for LLM consumption (no trust tagging)."""
    if not results:
        return ""
    parts = []
    for r in results:
        rel = r.get("relevance")
        cred = r.get("credibility")
        meta = ""
        if rel is not None or cred is not None:
            meta = f"Relevance: {rel if rel is not None else 'N/A'} | Credibility: {cred if cred is not None else 'N/A'}\n"
        parts.append(
            f"Title: {r['title']}\n"
            f"Source: {r['url']}\n"
            f"{meta}"
            f"{r['content']}"
        )
    return "\n---\n".join(parts)
