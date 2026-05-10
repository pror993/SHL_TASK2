import json
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Iterable, List

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models

BATCH_SIZE = 50
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env(base_dir: Path) -> Dict[str, str]:
    load_dotenv(base_dir / ".env")
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_key = os.getenv("QDRANT_KEY", "").strip()
    collection_name = os.getenv("COLLECTION_NAME", "").strip()
    if not qdrant_url:
        raise ValueError("QDRANT_URL is required in .env")
    if not collection_name:
        raise ValueError("COLLECTION_NAME is required in .env")
    return {
        "QDRANT_URL": qdrant_url,
        "QDRANT_KEY": qdrant_key,
        "COLLECTION_NAME": collection_name,
    }


def _load_catalog(catalog_path: Path) -> List[dict]:
    with catalog_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("catalog.json must contain a list of assessments")
    return data


def _chunked(items: List[dict], size: int) -> Iterable[List[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _to_list(values) -> List:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _entity_id(item: dict) -> str:
    name = str(item.get("name", "")).strip()
    link = str(item.get("link", "")).strip()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{name}|{link}"))


def _collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        return client.collection_exists(name)
    except AttributeError:
        try:
            client.get_collection(name)
        except Exception:
            return False
        return True


def _ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    if _collection_exists(client, name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams()
        },
    )


def _embedding_size(model: TextEmbedding) -> int:
    for vector in model.embed(["size probe"]):
        return len(vector)
    raise RuntimeError("Failed to infer embedding size")


def _build_payload(item: dict, entity_id: str) -> dict:
    return {
        "name": item.get("name"),
        "link": item.get("link"),
        "description": item.get("description"),
        "job_levels": item.get("job_levels", []),
        "keys": item.get("keys", []),
        "languages": item.get("languages", []),
        "adaptive": item.get("adaptive"),
        "duration": item.get("duration"),
        "entity_id": entity_id,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    base_dir = _project_root()
    env = _load_env(base_dir)
    catalog_path = base_dir / "data" / "catalog.json"
    catalog = _load_catalog(catalog_path)

    client = QdrantClient(url=env["QDRANT_URL"], api_key=env["QDRANT_KEY"] or None)
    dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    vector_size = _embedding_size(dense_model)

    _ensure_collection(client, env["COLLECTION_NAME"], vector_size)

    total = len(catalog)
    if total == 0:
        logger.info("No catalog items found.")
        return

    processed = 0
    for batch in _chunked(catalog, BATCH_SIZE):
        documents = [
            f"{item.get('name', '')} {item.get('description', '')}".strip()
            for item in batch
        ]
        dense_vectors = list(dense_model.embed(documents))
        sparse_vectors = list(sparse_model.embed(documents))

        points = []
        for item, dense_vector, sparse_vector in zip(
            batch, dense_vectors, sparse_vectors
        ):
            entity_id = _entity_id(item)
            payload = _build_payload(item, entity_id)
            sparse_payload = models.SparseVector(
                indices=_to_list(sparse_vector.indices),
                values=_to_list(sparse_vector.values),
            )
            points.append(
                models.PointStruct(
                    id=entity_id,
                    vector={
                        DENSE_VECTOR_NAME: _to_list(dense_vector),
                        SPARSE_VECTOR_NAME: sparse_payload,
                    },
                    payload=payload,
                )
            )

        client.upsert(collection_name=env["COLLECTION_NAME"], points=points)
        processed += len(batch)
        logger.info("Upserted %s/%s catalog items", processed, total)

    logger.info("Indexing complete.")


if __name__ == "__main__":
    main()
