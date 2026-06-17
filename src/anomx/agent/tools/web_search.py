"""Web-search tool."""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from anomx.agent.base.tools import BaseTool, ToolExecutionContext, object_schema, statement_property
from anomx.agent.tools.web_fetch import plain_web_text

WEB_SEARCH_MAX_RESULTS = 8


class WebSearchTool(BaseTool):
    def __init__(self, *, statement_description: str) -> None:
        super().__init__(
            name="web_search",
            description="Search the web for relevant pages.",
            parameters=object_schema(
                {
                    "statement": statement_property(statement_description),
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results.",
                    },
                },
                ["statement", "query", "limit"],
            ),
            aliases=("websearch",),
        )

    def execute(self, arguments: dict[str, Any], context: ToolExecutionContext) -> str:
        context.emit_operator_statement(self.name, arguments)
        query = str(arguments.get("query", "")).strip()
        if not query:
            return context.json_result({"error": "web_search requires a query."})
        limit = min(context.positive_int(arguments.get("limit"), WEB_SEARCH_MAX_RESULTS), 10)
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://duckduckgo.com/html/?{encoded}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AnomxAgent/0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                html_text = response.read(80_000).decode("utf-8", errors="replace")
        except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as error:
            return context.json_result({"error": str(error), "query": query})
        return context.json_result(
            {
                "query": query,
                "results": duckduckgo_results(html_text, limit),
            }
        )


def duckduckgo_results(html_text: str, limit: int) -> list[dict[str, str]]:
    """Extract DuckDuckGo HTML result titles and target URLs."""

    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(html_text):
        href = html.unescape(match.group("href"))
        title = plain_web_text(match.group("title"))
        url = duckduckgo_result_url(href)
        if not title or not url:
            continue
        results.append({"title": title, "url": url})
        if len(results) >= limit:
            break
    return results


def duckduckgo_result_url(href: str) -> str:
    """Return the original result URL from a DuckDuckGo redirect href."""

    parsed = urllib.parse.urlparse(href)
    query = urllib.parse.parse_qs(parsed.query)
    uddg = query.get("uddg")
    if uddg:
        return urllib.parse.unquote(uddg[0])
    return href
