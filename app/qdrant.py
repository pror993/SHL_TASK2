import logging
from typing import Iterable, List, Optional

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import QDRANT_KEY, QDRANT_URL

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

logger = logging.getLogger(__name__)

_DENSE_EMBEDDER = TextEmbedding(model_name=DENSE_MODEL_NAME)
_SPARSE_EMBEDDER = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY or None)


def _to_list(values) -> List[float]:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _build_filter(
    job_levels: Optional[Iterable[str]],
    keys: Optional[Iterable[str]],
    adaptive: Optional[str],
    base_filter: Optional[models.Filter],
) -> Optional[models.Filter]:
    must = list(base_filter.must) if base_filter and base_filter.must else []
    should = list(base_filter.should) if base_filter and base_filter.should else []
    must_not = list(base_filter.must_not) if base_filter and base_filter.must_not else []

    if adaptive is not None and str(adaptive).strip() != "":
        must.append(
            models.FieldCondition(
                key="adaptive",
                match=models.MatchValue(value=str(adaptive).strip()),
            )
        )

    if job_levels:
        should.append(
            models.FieldCondition(
                key="job_levels",
                match=models.MatchAny(any=list(job_levels)),
            )
        )

    if keys:
        should.append(
            models.FieldCondition(
                key="keys",
                match=models.MatchAny(any=list(keys)),
            )
        )

    if not must and not should and not must_not:
        return None

    return models.Filter(must=must or None, should=should or None, must_not=must_not or None)


def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    limit: int = 10,
    payload_filter: Optional[models.Filter] = None,
    job_levels: Optional[Iterable[str]] = None,
    keys: Optional[Iterable[str]] = None,
    adaptive: Optional[str] = None,
) -> List[dict]:
    logger.info("hybrid_search start limit=%s query_len=%s", limit, len(query or ""))
    dense_vector = next(_DENSE_EMBEDDER.embed([query]))
    sparse_vector = next(_SPARSE_EMBEDDER.embed([query]))
    qdrant_filter = _build_filter(job_levels, keys, adaptive, payload_filter)
    logger.info("hybrid_search filter=%s", "set" if qdrant_filter else "none")

    prefetch = [
        models.Prefetch(
            query=_to_list(dense_vector),
            using=DENSE_VECTOR_NAME,
            limit=limit,
            filter=qdrant_filter,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=_to_list(sparse_vector.indices),
                values=_to_list(sparse_vector.values),
            ),
            using=SPARSE_VECTOR_NAME,
            limit=limit,
            filter=qdrant_filter,
        ),
    ]

    response = client.query_points(
        collection_name=collection_name,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        prefetch=prefetch,
        with_payload=True,
        limit=limit,
    )
    results = [point.payload for point in response.points]
    logger.info("hybrid_search done results=%s", len(results))
    return results
