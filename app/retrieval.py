import asyncio
import logging
from typing import Iterable, List, Optional, Sequence

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from .config import COLLECTION_NAME, RERANKER_ENABLED
from .qdrant import hybrid_search

logger = logging.getLogger(__name__)


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> List[str]:
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [item for item, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


_CROSS_ENCODER: Optional[CrossEncoder] = None


def _get_cross_encoder() -> CrossEncoder:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER


def preload_reranker() -> None:
    if RERANKER_ENABLED:
        _get_cross_encoder()


def _normalize_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def _result_id(item: dict) -> Optional[str]:
    entity_id = item.get("entity_id")
    if entity_id:
        return str(entity_id)
    link = item.get("link")
    if link:
        return str(link)
    name = item.get("name")
    return str(name) if name else None


def _doc_text(item: dict) -> str:
    name = str(item.get("name", "")).strip()
    description = str(item.get("description", "")).strip()
    return f"{name} {description}".strip()


def _normalize_adaptive(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"yes", "y", "true", "1"}:
            return "yes"
        if lowered in {"no", "n", "false", "0"}:
            return "no"
        return value.strip() or None
    return str(value).strip() or None


def _pipeline_constraint(
    client: QdrantClient,
    requirement_summary: str,
    seniority: Optional[Iterable[str]],
    test_type: Optional[Iterable[str]],
    languages: Optional[Iterable[str]],
    adaptive: Optional[str],
    limit: int,
) -> List[dict]:
    logger.info("pipeline_constraint start limit=%s", limit)
    results = hybrid_search(
        client=client,
        collection_name=COLLECTION_NAME,
        query=requirement_summary,
        limit=limit,
        job_levels=seniority,
        keys=test_type,
        languages=languages,
        adaptive=adaptive,
    )
    logger.info("pipeline_constraint done results=%s", len(results))
    return results


def _pipeline_jd(client: QdrantClient, query_text: str, limit: int) -> List[dict]:
    logger.info("pipeline_jd start limit=%s query_len=%s", limit, len(query_text or ""))
    results = hybrid_search(
        client=client,
        collection_name=COLLECTION_NAME,
        query=query_text,
        limit=limit,
    )
    logger.info("pipeline_jd done results=%s", len(results))
    return results


async def pipeline_constraint(
    client: QdrantClient,
    requirement_summary: str,
    seniority: Optional[Iterable[str]],
    test_type: Optional[Iterable[str]],
    languages: Optional[Iterable[str]],
    adaptive: Optional[str],
    limit: int = 20,
) -> List[dict]:
    seniority_list = _normalize_list(seniority)
    test_type_list = _normalize_list(test_type)
    languages_list = _normalize_list(languages)
    adaptive_value = _normalize_adaptive(adaptive)
    return await asyncio.to_thread(
        _pipeline_constraint,
        client,
        requirement_summary or "",
        seniority_list or None,
        test_type_list or None,
        languages_list or None,
        adaptive_value,
        limit,
    )


async def pipeline_jd(
    client: QdrantClient, query_text: str, limit: int = 20
) -> List[dict]:
    query = (query_text or "").strip()
    return await asyncio.to_thread(_pipeline_jd, client, query, limit)


def merge_results(
    results_a: List[dict], results_b: List[dict], limit: int = 20
) -> List[dict]:
    logger.info(
        "merge_results start a=%s b=%s limit=%s", len(results_a), len(results_b), limit
    )
    ranking_a: List[str] = []
    for item_id in map(_result_id, results_a):
        if item_id:
            ranking_a.append(item_id)

    ranking_b: List[str] = []
    for item_id in map(_result_id, results_b):
        if item_id:
            ranking_b.append(item_id)

    fused_ids = rrf_fuse([ranking_a, ranking_b])

    merged: List[dict] = []
    seen = set()
    index = {(_result_id(item) or ""): item for item in results_a + results_b}
    for item_id in fused_ids:
        if item_id in seen:
            continue
        item = index.get(item_id)
        if item:
            merged.append(item)
            seen.add(item_id)
        if len(merged) >= limit:
            break
    logger.info("merge_results done merged=%s", len(merged))
    return merged


def merge_many_results(result_sets: List[List[dict]], limit: int = 20) -> List[dict]:
    logger.info("merge_many_results start sets=%s limit=%s", len(result_sets), limit)
    rankings: List[List[str]] = []
    index = {}
    for results in result_sets:
        ranking: List[str] = []
        for item in results:
            item_id = _result_id(item)
            if item_id:
                ranking.append(item_id)
                index[item_id] = item
        rankings.append(ranking)

    fused_ids = rrf_fuse(rankings)
    merged: List[dict] = []
    seen = set()
    for item_id in fused_ids:
        if item_id in seen:
            continue
        item = index.get(item_id)
        if item:
            merged.append(item)
            seen.add(item_id)
        if len(merged) >= limit:
            break
    logger.info("merge_many_results done merged=%s", len(merged))
    return merged


def rerank_results(query_text: str, items: List[dict], top_k: int = 10) -> List[dict]:
    if not items:
        return []
    logger.info("rerank_results start items=%s top_k=%s", len(items), top_k)
    cross_encoder = _get_cross_encoder()
    pairs = [(query_text, _doc_text(item)) for item in items]
    scores = cross_encoder.predict(pairs)
    ranked = [
        item
        for item, _ in sorted(
            zip(items, scores), key=lambda pair: pair[1], reverse=True
        )
    ]
    results = ranked[:top_k]
    logger.info("rerank_results done results=%s", len(results))
    return results
