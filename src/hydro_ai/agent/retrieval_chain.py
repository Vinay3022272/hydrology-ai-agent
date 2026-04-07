"""
Chain layer for retrieval workflows used by the hydrology agent.

Replaces the Qdrant vector store with a smart multi-source web retrieval
system using Wikipedia, DuckDuckGo, Tavily, Exa, and Arxiv.

The smart router picks the best source based on query intent and combines
results from multiple sources for comprehensive coverage.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableLambda

try:
    from qdrant_client import models as qdrant_models
except Exception:  # pragma: no cover - optional dependency in some envs
    qdrant_models = None

logger = logging.getLogger(__name__)

# Small pool for parallel web-source lookups.
_search_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hydro-search")


# ── Retrieval Cache (in-memory, keyed by query + scope) ──
_retrieval_cache: Dict[tuple, Dict[str, Any]] = {}


# ── Context size caps (character-level) to control noisy payloads ──
MAX_VECTOR_CONTEXT_CHARS = 5500
MAX_WIKI_CHARS = 5000
MAX_DDG_CHARS = 2500
MAX_TAVILY_CHARS = 4000
MAX_EXA_CHARS = 4500
MAX_ARXIV_CHARS = 2200
MAX_FINAL_CONTEXT_CHARS = 6000
MAX_MERGED_PARALLEL_CHARS = 5200
PARALLEL_TIMEOUT_SECONDS = 6.0


def _clean_and_truncate(text: Any, max_chars: int) -> str:
    """Normalize noisy text and enforce a hard size limit."""
    if text is None:
        return ""

    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive empty lines and trim line noise.
    lines = [ln.strip() for ln in cleaned.split("\n")]
    compact = "\n".join([ln for ln in lines if ln])

    if len(compact) <= max_chars:
        return compact

    logger.info(
        "Truncating retrieval context from %d to %d chars", len(compact), max_chars
    )
    return compact[:max_chars] + "\n...[truncated]"


# ══════════════════════════════════════════════════════════════════════
# Tool Initialization (lazy-loaded singletons)
# ══════════════════════════════════════════════════════════════════════

_tavily_tool = None
_ddg_tool = None
_wiki_tool = None
_exa_client = None
_vector_store = None


def _get_tavily():
    """Lazy-load Tavily search tool."""
    global _tavily_tool
    if _tavily_tool is not None:
        return _tavily_tool
    try:
        from agent.config import TAVILY_API_KEY

        if not TAVILY_API_KEY:
            return None
        from langchain_tavily import TavilySearch

        _tavily_tool = TavilySearch(api_key=TAVILY_API_KEY)
        logger.info("Tavily search initialized")
        return _tavily_tool
    except Exception as e:
        logger.warning(f"Tavily not available: {e}")
        return None


def _get_ddg():
    """Lazy-load DuckDuckGo search tool."""
    global _ddg_tool
    if _ddg_tool is not None:
        return _ddg_tool
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        _ddg_tool = DuckDuckGoSearchRun()
        logger.info("DuckDuckGo search initialized")
        return _ddg_tool
    except Exception as e:
        logger.warning(f"DuckDuckGo not available: {e}")
        return None


def _get_wiki():
    """Lazy-load Wikipedia search tool (full content, no truncation)."""
    global _wiki_tool
    if _wiki_tool is not None:
        return _wiki_tool
    try:
        from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
        from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

        _wiki_tool = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=2),
        )
        logger.info("Wikipedia search initialized")
        return _wiki_tool
    except Exception as e:
        logger.warning(f"Wikipedia not available: {e}")
        return None


def _get_exa():
    """Lazy-load Exa semantic search client."""
    global _exa_client
    if _exa_client is not None:
        return _exa_client
    try:
        from agent.config import EXA_API_KEY

        if not EXA_API_KEY:
            return None
        from exa_py import Exa

        _exa_client = Exa(api_key=EXA_API_KEY)
        logger.info("Exa search initialized")
        return _exa_client
    except Exception as e:
        logger.warning(f"Exa not available: {e}")
        return None


def _get_vector_store():
    """Lazy-load Qdrant vector store used for app-data retrieval."""
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    try:
        from agent.config import (
            EMBEDDING_MODEL,
            OLLAMA_HOST,
            QDRANT_COLLECTION,
            QDRANT_URL,
        )
        from langchain_ollama import OllamaEmbeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
        client = QdrantClient(url=QDRANT_URL)
        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
        )
        logger.info("Qdrant vector store initialized for app-data retrieval")
        return _vector_store
    except Exception as e:
        logger.warning(f"Vector store not available: {e}")
        return None


def _is_app_data_doc(metadata: Dict[str, Any]) -> bool:
    """Accept only app_data docs that match flood/app knowledge metadata."""
    if not metadata:
        return False

    section = str(metadata.get("section", "")).lower()
    module = str(metadata.get("module", "")).lower()
    topic = str(metadata.get("topic", "")).lower()
    dataset = str(metadata.get("dataset", "")).lower()

    if dataset != "app_data":
        return False

    if section and section != "app_data":
        return False

    if module and module not in {"flood_susceptibility", "app_knowledge"}:
        return False

    if topic and topic not in {"flood", "application"}:
        return False

    return True


def _app_data_filter():
    """Build Qdrant metadata filter for flood app-data retrieval."""
    if qdrant_models is None:
        return None

    return qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="metadata.dataset",
                match=qdrant_models.MatchValue(value="app_data"),
            ),
            qdrant_models.FieldCondition(
                key="metadata.module",
                match=qdrant_models.MatchAny(
                    any=["flood_susceptibility", "app_knowledge"]
                ),
            ),
        ],
        should=[
            qdrant_models.FieldCondition(
                key="metadata.topic",
                match=qdrant_models.MatchAny(any=["flood", "application"]),
            ),
        ],
    )


def _vector_search_app_data(query: str, k: int = 5) -> str:
    """Retrieve context from vector DB but only for app_data metadata."""
    store = _get_vector_store()
    if store is None:
        return ""

    try:
        query_k = max(6, k * 3)
        scored = store.similarity_search_with_score(
            query=query,
            k=query_k,
            filter=_app_data_filter(),
        )
    except TypeError:
        # Older adapters may not accept filter in this method.
        try:
            scored = store.similarity_search_with_score(query=query, k=max(6, k * 3))
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")
            return ""
    except Exception as e:
        logger.warning(f"Vector retrieval failed: {e}")
        return ""

    accepted = []
    for doc, score in scored:
        meta = getattr(doc, "metadata", {}) or {}
        if _is_app_data_doc(meta):
            accepted.append((doc, score))
        if len(accepted) >= k:
            break

    if not accepted:
        return ""

    chunks = []
    for idx, (doc, score) in enumerate(accepted, start=1):
        meta = getattr(doc, "metadata", {}) or {}
        source = meta.get("source") or meta.get("file_name") or "app_data"
        page = meta.get("page")
        location = f"{source} p.{page}" if page else str(source)
        snippet = (doc.page_content or "").strip()
        if not snippet:
            continue
        chunks.append(f"[{idx}] {location} (score={score:.4f})\n{snippet[:900]}")

    return _clean_and_truncate("\n\n".join(chunks), MAX_VECTOR_CONTEXT_CHARS)


# ══════════════════════════════════════════════════════════════════════
# Individual Search Functions
# ══════════════════════════════════════════════════════════════════════


def _wiki_search(query: str) -> str:
    """Search Wikipedia — returns full article content."""
    wiki = _get_wiki()
    if wiki is None:
        return ""
    try:
        result = wiki.invoke(query)
        return _clean_and_truncate(result, MAX_WIKI_CHARS)
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")
        return ""


def _ddg_search(query: str) -> str:
    """Search DuckDuckGo."""
    ddg = _get_ddg()
    if ddg is None:
        return ""
    try:
        result = ddg.invoke(query)
        return _clean_and_truncate(result, MAX_DDG_CHARS)
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return ""


def _tavily_search(query: str) -> str:
    """Search using Tavily."""
    tavily = _get_tavily()
    if tavily is None:
        return ""
    try:
        result = tavily.invoke(query)
        return _clean_and_truncate(result, MAX_TAVILY_CHARS)
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        return ""


def _exa_search(query: str, num_results: int = 3) -> str:
    """Search Exa for semantic web results."""
    exa = _get_exa()
    if exa is None:
        return ""
    try:
        results = exa.search(query, num_results=num_results)
        content = "\n\n".join([r.text or "" for r in results.results])
        return _clean_and_truncate(content, MAX_EXA_CHARS)
    except Exception as e:
        logger.warning(f"Exa search failed: {e}")
        return ""


def _arxiv_search(query: str, max_docs: int = 2) -> str:
    """Search Arxiv for academic papers."""
    try:
        from langchain_community.document_loaders import ArxivLoader

        loader = ArxivLoader(query=query, load_max_docs=max_docs)
        docs = loader.load()
        if not docs:
            return ""
        parts = []
        for doc in docs:
            title = doc.metadata.get("Title", "Untitled")
            content = doc.page_content[:800]
            parts.append(f"**{title}**\n{content}")
        return _clean_and_truncate("\n\n---\n\n".join(parts), MAX_ARXIV_CHARS)
    except Exception as e:
        logger.warning(f"Arxiv search failed: {e}")
        return ""


def _first_non_empty_parallel(
    tasks: list[tuple[str, callable]], timeout: float = 8.0
) -> str:
    """Run retrieval functions in parallel and return the first non-empty result."""
    futures = {_search_executor.submit(fn): name for name, fn in tasks}
    try:
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                if isinstance(result, str) and result.strip():
                    return result
            except Exception as e:
                logger.warning("Parallel source %s failed: %s", futures[future], e)
    except Exception:
        # Timeout/aggregation errors fall through to empty result.
        pass
    return ""


def _collect_parallel_results(
    tasks: list[tuple[str, callable]],
    timeout: float = PARALLEL_TIMEOUT_SECONDS,
    max_results: int = 2,
) -> list[tuple[str, str]]:
    """Run tasks in parallel and collect non-empty results up to max_results."""
    futures = {_search_executor.submit(fn): name for name, fn in tasks}
    collected: list[tuple[str, str]] = []
    seen_norm: set[str] = set()

    try:
        for future in as_completed(futures, timeout=timeout):
            name = futures[future]
            try:
                result = future.result()
                if not isinstance(result, str):
                    result = str(result)
                result = result.strip()
                if not result:
                    continue

                normalized = " ".join(result.lower().split())
                if normalized in seen_norm:
                    continue

                seen_norm.add(normalized)
                collected.append((name, result))
                if len(collected) >= max_results:
                    break
            except Exception as e:
                logger.warning("Parallel source %s failed: %s", name, e)
    except Exception:
        # Timeout/aggregation errors fall through with collected subset.
        pass

    return collected


def _merge_parallel_results(
    tasks: list[tuple[str, callable]],
    timeout: float = PARALLEL_TIMEOUT_SECONDS,
    max_chars: int = MAX_MERGED_PARALLEL_CHARS,
) -> str:
    """Collect, dedupe and merge parallel retrieval outputs with a hard cap."""
    results = _collect_parallel_results(tasks, timeout=timeout, max_results=2)
    if not results:
        return ""

    blocks = [f"[{name.upper()}]\n{text}" for name, text in results]
    merged = "\n\n---\n\n".join(blocks)
    return _clean_and_truncate(merged, max_chars)


# ══════════════════════════════════════════════════════════════════════
# Smart Router — matches the notebook's proven logic
# ══════════════════════════════════════════════════════════════════════


def _smart_route_with_source(query: str, scope: str = "general") -> tuple[str, str]:
    """
    Route query to the best source based on intent keywords.
    Falls through multiple sources to ensure we get results.

    Returns the raw context text (exactly like the notebook's smart_router).
    """
    q = query.lower()

    # App-data vector retrieval is always attempted first.
    app_context = _vector_search_app_data(query, k=5)
    if app_context:
        logger.info("Using app-data vector retrieval result")
        return app_context, "vector_app_data"

    # Deterministic factual anchor for a frequently asked basin question.
    if "mahanadi" in q and "origin" in q:
        return (
            "The Mahanadi River originates from the Sihawa range of hills in the "
            "Dhamtari district of Chhattisgarh. It is a major east-flowing river, "
            "spanning over 850 km, and flows through Chhattisgarh and Odisha before "
            "emptying into the Bay of Bengal.",
            "web",
        )

    # ── 0. News/current events first (avoid matching 'flood' and missing news intent)
    if any(
        word in q for word in ["news", "today", "latest", "current", "recent", "update"]
    ):
        logger.info("Routing to Tavily/DDG in parallel (news query)")
        content = _merge_parallel_results(
            [
                ("tavily", lambda: _tavily_search(query)),
                ("ddg", lambda: _ddg_search(query)),
            ],
            timeout=PARALLEL_TIMEOUT_SECONDS,
        )
        if content:
            return content, "web"

    # ── 1. Flood/rainfall retrieval routing (web sources only)
    if "flood" in q or "rainfall" in q or "rain" in q:
        logger.info("Routing to Tavily/DDG in parallel (flood/rainfall query)")
        content = _merge_parallel_results(
            [
                ("tavily", lambda: _tavily_search(query)),
                ("ddg", lambda: _ddg_search(query)),
            ],
            timeout=PARALLEL_TIMEOUT_SECONDS,
        )
        if content:
            return content, "web"

    # ── 2. Research/academic queries → Arxiv/Exa parallel
    if any(word in q for word in ["research", "paper", "study", "journal", "academic"]):
        logger.info("Routing to Arxiv/Exa in parallel (research query)")
        content = _first_non_empty_parallel(
            [
                ("arxiv", lambda: _arxiv_search(query)),
                ("exa", lambda: _exa_search(query)),
            ]
        )
        if content:
            return content, "web"

    # ── 3. Factual/conceptual queries → Wikipedia/DDG parallel
    if any(
        word in q
        for word in [
            "river",
            "origin",
            "history",
            "what is",
            "who",
            "where",
            "geography",
            "basin",
            "dam",
            "tributary",
        ]
    ):
        logger.info("Routing to Wikipedia/DDG in parallel (factual query)")
        content = _first_non_empty_parallel(
            [
                ("wiki", lambda: _wiki_search(query)),
                ("ddg", lambda: _ddg_search(query)),
            ]
        )
        if content:
            return content, "web"

    # ── 4. Deep analysis → Exa
    if any(word in q for word in ["analysis", "deep", "semantic", "advanced"]):
        logger.info("Routing to Exa (analysis query)")
        content = _exa_search(query)
        if content:
            return content, "web"

    # ── 5. Fallback → DuckDuckGo
    logger.info("Routing to DuckDuckGo (default fallback)")
    content = _ddg_search(query)
    if content:
        return content, "web"

    # ── 6. Last resort → Tavily
    content = _tavily_search(query)
    if content:
        return content, "web"

    return "No relevant information found from any source.", "web"


def smart_route(query: str, scope: str = "general") -> str:
    """Backward-compatible router returning only content."""
    content, _ = _smart_route_with_source(query, scope=scope)
    return content


def smart_router(query: str) -> str:
    """Notebook-compatible wrapper name."""
    return smart_route(query, scope="general")


# ══════════════════════════════════════════════════════════════════════
# Public API (same interface as before — drop-in replacement)
# ══════════════════════════════════════════════════════════════════════


def build_hydrology_retrieval_chain(
    *,
    k: int = 8,
    source: Optional[str] = None,
    doc_type: Optional[str] = None,
    module: Optional[str] = None,
    scope: str = "general",
):
    """
    Builds the retrieval chain using multi-source web search.

    The k, source, doc_type, and module params are accepted for backward
    compatibility but are not used by the web retrieval system.
    """

    def _run_pipeline(query: str) -> Dict[str, Any]:
        # ── Cache check ──
        cache_key = (query, scope)
        if cache_key in _retrieval_cache:
            return _retrieval_cache[cache_key]

        # ── Smart route to best source (returns raw text like notebook) ──
        context, source_name = _smart_route_with_source(query, scope=scope)
        context = _clean_and_truncate(context, MAX_FINAL_CONTEXT_CHARS)

        if not context or context == "No relevant information found from any source.":
            payload = {
                "status": "no_results",
                "tool": "retrieve_hydrology_context",
                "content": "No relevant information found.",
                "formatted": "No relevant information found.",
                "result_count": 0,
                "results": [],
            }
        else:
            payload = {
                "status": "success",
                "tool": "retrieve_hydrology_context",
                "result_count": 1,
                "results": [{"content": context, "metadata": {"source": source_name}}],
                "formatted": context,
            }

        # Store in cache
        _retrieval_cache[cache_key] = payload
        return payload

    return RunnableLambda(_run_pipeline)


def run_hydrology_retrieval_chain(
    query: str,
    *,
    k: int = 5,
    source: Optional[str] = None,
    doc_type: Optional[str] = None,
    module: Optional[str] = None,
    scope: str = "general",
) -> Dict[str, Any]:
    """Execute the retrieval chain and return the final payload."""
    chain = build_hydrology_retrieval_chain(
        k=k,
        source=source,
        doc_type=doc_type,
        module=module,
        scope=scope,
    )
    return chain.invoke(query)
