"""Deep research agent built with Pydantic AI.

Takes a free-text question or a stock ticker and writes a structured report
from a fixed research plan:

1. Detect whether the input is a ticker or a general query, and resolve a
   ticker to a company plus the context around it.
2. Run one DuckDuckGo discovery search. A resolved ticker is searched with
   its company name so the results are about that company.
3. Turn the top snippets into 3-4 non-overlapping research angles.
4. Search DuckDuckGo once per angle, in parallel, then fetch the best pages.
5. Extract claims with the source that supports each one, and synthesize a report.

Search uses ``pydantic_ai.common_tools.duckduckgo.DuckDuckGoSearchTool``.
Page fetches use ``pydantic_ai.common_tools.web_fetch.WebFetchLocalTool``.

Requires ``OPENAI_API_KEY``. Run with:

    py -3 research_agent.py NVDA
    py -3 research_agent.py "solid-state battery manufacturing"
"""

from __future__ import annotations as _annotations

import asyncio
import os
import re
import sys
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from ddgs import DDGS
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.common_tools.duckduckgo import DuckDuckGoResult, DuckDuckGoSearchTool
from pydantic_ai.common_tools.web_fetch import WebFetchLocalTool
from pydantic_ai.messages import BinaryContent

try:
    import logfire
except ImportError:  # pragma: no cover - optional, same as the docs' if-token-present mode
    logfire = None
else:
    logfire.configure(send_to_logfire='if-token-present')
    logfire.instrument_pydantic_ai()

MODEL = os.getenv('RESEARCH_MODEL', 'openai:gpt-5-mini')
DISCOVERY_RESULTS = 8
ANGLE_RESULTS = 6
PAGES_PER_ANGLE = 3
PAGE_CHARS = 6_000

REPUTABLE_HOSTS = {
    'reuters.com',
    'apnews.com',
    'bloomberg.com',
    'wsj.com',
    'ft.com',
    'cnbc.com',
    'nytimes.com',
    'washingtonpost.com',
    'bbc.com',
    'bbc.co.uk',
    'economist.com',
    'fortune.com',
    'barrons.com',
    'marketwatch.com',
    'theinformation.com',
    'nikkei.com',
    'nature.com',
    'science.org',
    'ieee.org',
}
SKIP_HOSTS = {
    'facebook.com',
    'instagram.com',
    'tiktok.com',
    'pinterest.com',
    'x.com',
    'twitter.com',
    'youtube.com',
    'reddit.com',
}


class QueryIntent(BaseModel):
    """Whether the user passed a ticker or a free-text question."""

    kind: Literal['ticker', 'general']
    subject: str = Field(description='Company name for a ticker, otherwise a short restatement of the query.')
    ticker: str | None = Field(description='Uppercase ticker when kind is ticker, otherwise null.')
    context: list[str] = Field(
        description='For a ticker: industry, products, and themes, e.g. semiconductors, GPUs, AI. Otherwise themes from the query.'
    )


class ResearchAngle(BaseModel):
    keyword: str = Field(description='Short name of this research angle.')
    search_query: str = Field(description='DuckDuckGo query that includes the subject and this angle.')


class ResearchPlan(BaseModel):
    angles: list[ResearchAngle] = Field(min_length=3, max_length=4)


class Evidence(BaseModel):
    claim: str
    source_title: str
    url: str


class SourceRecord(BaseModel):
    title: str
    url: str
    snippet: str
    tier: Literal['primary', 'reputable', 'other']
    page_excerpt: str | None = None


class AngleFindings(BaseModel):
    keyword: str
    key_findings: list[str] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    uncertainties: list[str] = Field(
        description='Gaps, caveats, or claims the sources disagree on for this angle.'
    )


class Synthesis(BaseModel):
    title: str
    executive_summary: str = Field(description='Several sentences covering the whole report, with no new facts.')
    risks_and_uncertainties: list[str] = Field(min_length=1)
    conflicting_information: list[str] = Field(
        description='Places sources disagree. Empty only when the notes show no conflict.'
    )
    what_to_watch_next: list[str] = Field(min_length=1)


class ReportSection(BaseModel):
    keyword: str
    key_findings: list[str]
    evidence: list[Evidence]
    uncertainties: list[str]


class ResearchReport(BaseModel):
    title: str
    query: str
    intent: Literal['ticker', 'general']
    subject: str
    ticker: str | None
    context: list[str]
    executive_summary: str
    sections: list[ReportSection]
    risks_and_uncertainties: list[str]
    conflicting_information: list[str]
    what_to_watch_next: list[str]


intent_agent = Agent(
    MODEL,
    output_type=QueryIntent,
    instructions=(
        'Classify the user input. Use kind="ticker" only when the entire input is a '
        'stock ticker, optionally with an exchange prefix such as NASDAQ:NVDA or a '
        'share-class suffix such as BRK.B. Anything else is kind="general", even if '
        'it mentions a company or a ticker.\n'
        'When kind is ticker, set ticker to the bare uppercase symbol, subject to the '
        'company name, and context to the industry, products, and themes a reader '
        'needs (for NVDA: NVIDIA, semiconductors, GPUs, AI). Do not invent a company '
        'for a symbol you do not know; use the symbol as subject and say the company '
        'is unresolved in context.\n'
        'When kind is general, ticker is null, subject restates the question, and '
        'context lists the main themes.'
    ),
)

angle_agent = Agent(
    MODEL,
    output_type=ResearchPlan,
    instructions=(
        'You plan web research. From the discovery snippets, propose 3 or 4 angles. '
        'Each angle must cover different ground: do not split one topic into two '
        'near-duplicate keywords.\n'
        'For a stock, prefer angles such as SWOT, last-12-months stock performance, '
        'competition and market positioning, and latest quarterly results with forward '
        'guidance, but replace any of those when the snippets show a more important '
        'and still non-overlapping question.\n'
        'For a general query, derive the angles from what the snippets actually discuss.\n'
        'search_query must contain the subject plus the angle, and must be specific '
        'enough for DuckDuckGo. keyword is a short section title.'
    ),
)

extraction_agent = Agent(
    MODEL,
    output_type=AngleFindings,
    instructions=(
        'Extract facts for one research angle from the sources below. Use only those '
        'sources. Every evidence item is one checkable claim, number, or attributed '
        'statement, and it must cite a url and source title copied from the sources. '
        'Prefer primary sources for financial figures: SEC filings, earnings releases, '
        'and investor-relations pages. Prefer Reuters, AP, Bloomberg, FT, WSJ, and '
        'similar outlets for news. If sources disagree, keep both claims and describe '
        'the disagreement in uncertainties. Do not fill gaps with outside knowledge. '
        'If a page excerpt is missing, you may use the snippet and must say the page '
        'was not fetched.'
    ),
)

synthesis_agent = Agent(
    MODEL,
    output_type=Synthesis,
    instructions=(
        'Write the narrative parts of a research report from the section notes. '
        'Do not introduce facts, numbers, or sources that are not in the notes. '
        'The executive summary should stand alone. risks_and_uncertainties covers '
        'what is weakly sourced or could change. conflicting_information quotes the '
        'disagreements already present in the notes. what_to_watch_next is a short '
        'list of concrete upcoming events, filings, or open questions from the notes.'
    ),
)


@angle_agent.output_validator
async def angles_do_not_overlap(ctx: RunContext[None], plan: ResearchPlan) -> ResearchPlan:
    """Reject a plan whose keywords are the same topic twice."""
    seen: list[set[str]] = []
    for angle in plan.angles:
        tokens = _keywords(angle.keyword)
        for previous in seen:
            union = tokens | previous
            if union and len(tokens & previous) / len(union) > 0.6:
                raise ModelRetry(
                    f'"{angle.keyword}" overlaps an earlier angle. '
                    'Return 3 or 4 angles that do not cover the same topic.'
                )
        seen.append(tokens)
    return plan


def _clean_excerpt(content: str) -> str:
    """Drop style and script leftovers that page-to-markdown conversion sometimes keeps."""
    content = re.sub(r'(?is)<style\b[^>]*>.*?</style>', ' ', content)
    content = re.sub(r'(?is)<script\b[^>]*>.*?</script>', ' ', content)
    content = re.sub(r'(?is)(?:html|body)\s*\{[^{}]*\}', ' ', content)
    content = re.sub(r'[ \t]{2,}', ' ', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


def _keywords(text: str) -> set[str]:
    return {word for word in re.findall(r'[a-z0-9]+', text.lower()) if len(word) > 3}


def _normalize_url(url: str) -> str:
    return url.strip().rstrip('/')


def _host(url: str) -> str:
    host = (urlparse(url).hostname or '').lower().removeprefix('www.')
    return host[:-1] if host.endswith('.') else host


def source_tier(url: str) -> Literal['primary', 'reputable', 'other']:
    """Rank filings and investor pages above news, and news above everything else."""
    host = _host(url)
    path = urlparse(url).path.lower()
    if host.endswith('sec.gov') or host.startswith('ir.') or 'investor' in host or 'investor' in path:
        return 'primary'
    if host in REPUTABLE_HOSTS or any(host.endswith('.' + name) for name in REPUTABLE_HOSTS):
        return 'reputable'
    return 'other'


def rank_results(results: list[DuckDuckGoResult]) -> list[DuckDuckGoResult]:
    """Keep useful hits, with primary and reputable sources first."""
    kept = [item for item in results if _host(item['href']) not in SKIP_HOSTS]
    order = {'primary': 0, 'reputable': 1, 'other': 2}
    return sorted(kept, key=lambda item: order[source_tier(item['href'])])


async def search_duckduckgo(query: str, max_results: int) -> list[DuckDuckGoResult]:
    """Run one DuckDuckGo text search through the Pydantic AI search tool.

    The default client timeout is 5 seconds. Engines often need longer, and a
    short wait comes back as "No results found" even when a later engine would
    have answered.
    """
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with DDGS(timeout=20) as client:
                tool = DuckDuckGoSearchTool(client=client, max_results=max_results)
                return await tool(query)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


_fetcher = WebFetchLocalTool(
    max_content_length=PAGE_CHARS,
    allow_local_urls=False,
    timeout=20,
)


async def fetch_page(url: str) -> str | None:
    """Fetch one page as markdown. Missing or binary pages contribute no excerpt."""
    try:
        fetched = await _fetcher(url)
    except Exception:
        return None
    if isinstance(fetched, BinaryContent):
        return None
    content = _clean_excerpt(str(fetched.get('content', '')))
    return content[:PAGE_CHARS] or None


def _format_discovery(results: list[DuckDuckGoResult]) -> str:
    lines: list[str] = []
    for index, item in enumerate(results, start=1):
        lines.append(
            f'{index}. {item["title"]}\n   {item["href"]}\n   {item["body"]}'
        )
    return '\n'.join(lines) if lines else '(no results)'


def _format_sources(sources: list[SourceRecord]) -> str:
    blocks: list[str] = []
    for source in sources:
        excerpt = source.page_excerpt or '(page not fetched; use the snippet only)'
        blocks.append(
            f'Title: {source.title}\n'
            f'URL: {source.url}\n'
            f'Tier: {source.tier}\n'
            f'Snippet: {source.snippet}\n'
            f'Page:\n{excerpt}'
        )
    return '\n\n---\n\n'.join(blocks)


async def discover(query: str) -> list[DuckDuckGoResult]:
    results = await search_duckduckgo(query, DISCOVERY_RESULTS)
    return rank_results(results)


async def research_angle(angle: ResearchAngle) -> tuple[AngleFindings, list[SourceRecord]]:
    """Search once for this angle, fetch the best pages, and extract cited claims."""
    raw_results = rank_results(await search_duckduckgo(angle.search_query, ANGLE_RESULTS))
    chosen = raw_results[:PAGES_PER_ANGLE]
    excerpts = await asyncio.gather(*(fetch_page(item['href']) for item in chosen))
    sources = [
        SourceRecord(
            title=item['title'],
            url=item['href'],
            snippet=item['body'],
            tier=source_tier(item['href']),
            page_excerpt=excerpt,
        )
        for item, excerpt in zip(chosen, excerpts)
    ]
    if not sources:
        findings = AngleFindings(
            keyword=angle.keyword,
            key_findings=['No DuckDuckGo results were returned for this angle.'],
            evidence=[
                Evidence(
                    claim='This section has no supporting source.',
                    source_title='(none)',
                    url='',
                )
            ],
            uncertainties=['The search returned nothing, so this section is incomplete.'],
        )
        return findings, sources

    result = await extraction_agent.run(
        f'Angle: {angle.keyword}\nSearch query: {angle.search_query}\n\n'
        f'Sources:\n{_format_sources(sources)}'
    )
    findings = result.output
    findings.keyword = angle.keyword
    allowed = {_normalize_url(source.url) for source in sources}
    grounded = [item for item in findings.evidence if _normalize_url(item.url) in allowed]
    if grounded:
        findings.evidence = grounded
    else:
        findings.uncertainties.append(
            'The extracted claims did not cite one of the retrieved URLs, so treat this section as unverified.'
        )
    return findings, sources


async def research(query: str) -> ResearchReport:
    """Run the full research plan and return a structured report."""
    text = query.strip()
    if not text:
        raise ValueError('query must not be empty')

    intent = (await intent_agent.run(text)).output
    # One discovery search. A bare ticker such as NVDA also matches unrelated
    # projects, so include the resolved company name when we have one.
    discovery_query = text
    if intent.kind == 'ticker' and intent.ticker and intent.subject.casefold() != intent.ticker.casefold():
        discovery_query = f'{intent.ticker} {intent.subject}'
    discovery = await discover(discovery_query)
    plan = (
        await angle_agent.run(
            f'Input: {text}\n'
            f'Kind: {intent.kind}\n'
            f'Subject: {intent.subject}\n'
            f'Ticker: {intent.ticker or "(none)"}\n'
            f'Context: {", ".join(intent.context) or "(none)"}\n\n'
            f'Discovery results:\n{_format_discovery(discovery)}'
        )
    ).output

    dives = await asyncio.gather(*(research_angle(angle) for angle in plan.angles))
    sections = [
        ReportSection(
            keyword=findings.keyword,
            key_findings=findings.key_findings,
            evidence=findings.evidence,
            uncertainties=findings.uncertainties,
        )
        for findings, _sources in dives
    ]
    notes = _format_sections(sections)
    synthesis = (
        await synthesis_agent.run(
            f'Query: {text}\nSubject: {intent.subject}\nContext: {", ".join(intent.context)}\n\n'
            f'Section notes:\n{notes}'
        )
    ).output
    return ResearchReport(
        title=synthesis.title,
        query=text,
        intent=intent.kind,
        subject=intent.subject,
        ticker=intent.ticker,
        context=intent.context,
        executive_summary=synthesis.executive_summary,
        sections=sections,
        risks_and_uncertainties=synthesis.risks_and_uncertainties,
        conflicting_information=synthesis.conflicting_information,
        what_to_watch_next=synthesis.what_to_watch_next,
    )


def _format_sections(sections: list[ReportSection]) -> str:
    chunks: list[str] = []
    for section in sections:
        evidence = '\n'.join(
            f'- {item.claim} ({item.source_title}: {item.url})' for item in section.evidence
        )
        findings = '\n'.join(f'- {item}' for item in section.key_findings)
        uncertainties = '\n'.join(f'- {item}' for item in section.uncertainties) or '- (none noted)'
        chunks.append(
            f'## {section.keyword}\nFindings:\n{findings}\nEvidence:\n{evidence}\n'
            f'Uncertainties:\n{uncertainties}'
        )
    return '\n\n'.join(chunks)


def render_report(report: ResearchReport) -> str:
    """Render a report as markdown for the terminal."""
    identity = report.subject
    if report.ticker:
        identity = f'{report.ticker} → {report.subject}'
    context = ', '.join(report.context) if report.context else '(none)'
    lines = [
        f'# {report.title}',
        '',
        f'Query: {report.query}',
        f'Intent: {report.intent} ({identity})',
        f'Context: {context}',
        '',
        '## Executive summary',
        report.executive_summary,
        '',
    ]
    for section in report.sections:
        lines.append(f'## {section.keyword}')
        lines.append('')
        lines.append('Key findings:')
        lines.extend(f'- {item}' for item in section.key_findings)
        lines.append('')
        lines.append('Evidence:')
        for item in section.evidence:
            lines.append(f'- {item.claim} ({item.source_title}: {item.url})')
        if section.uncertainties:
            lines.append('')
            lines.append('Uncertainties in this section:')
            lines.extend(f'- {item}' for item in section.uncertainties)
        lines.append('')
    lines.append('## Risks and uncertainties')
    lines.extend(f'- {item}' for item in report.risks_and_uncertainties)
    lines.append('')
    lines.append('## Conflicting information')
    if report.conflicting_information:
        lines.extend(f'- {item}' for item in report.conflicting_information)
    else:
        lines.append('- None recorded in the sourced notes.')
    lines.append('')
    lines.append('## What to watch next')
    lines.extend(f'- {item}' for item in report.what_to_watch_next)
    lines.append('')
    return '\n'.join(lines)


async def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: py -3 research_agent.py "<query or ticker>"')
    report = await research(' '.join(sys.argv[1:]))
    print(render_report(report))


if __name__ == '__main__':
    asyncio.run(main())
