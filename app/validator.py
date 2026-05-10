from typing import List, Set

from .catalog import CatalogMeta

_VALID_URLS: Set[str] = set()


def init_validator(catalog_meta: CatalogMeta) -> None:
    global _VALID_URLS
    _VALID_URLS = set(catalog_meta.valid_urls)


def _get_valid_urls() -> Set[str]:
    return _VALID_URLS


def validate_recommendations(recommendations: List[dict], intent: str) -> List[dict]:
    valid_set = _get_valid_urls()
    filtered = [rec for rec in recommendations if rec.get("url") in valid_set]
    if intent in {"clarify", "refuse"}:
        filtered = []
    filtered = filtered[:10]
    return filtered
