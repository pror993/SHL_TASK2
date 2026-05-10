import json
from pathlib import Path
from typing import Iterable, List, Set

from rapidfuzz import fuzz, process

TEST_TYPE_MAP = {
    "knowledge & skills": "K",
    "personality & behavior": "P",
    "ability & aptitude": "A",
    "competencies": "C",
    "biodata & situational judgment": "B",
    "biodata & situational judgement": "B",
    "development & 360": "D",
    "assessment exercises": "E",
    "simulations": "S",
}


class CatalogMeta:
    def __init__(self, catalog_path: str) -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog = self._load_catalog(self.catalog_path)

        self.job_levels = self._unique_values("job_levels")
        self.keys = self._unique_values("keys")
        self.languages = self._unique_values("languages")
        self.adaptive = self._unique_values("adaptive")
        self.valid_urls = self._collect_urls()

    def _load_catalog(self, path: Path) -> List[dict]:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("catalog.json must contain a list of assessments")
        return data

    def _normalize_list(self, value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return [str(value).strip()] if str(value).strip() else []

    def _unique_values(self, field: str) -> Set[str]:
        values: Set[str] = set()
        for item in self.catalog:
            values.update(self._normalize_list(item.get(field)))
        return values

    def _collect_urls(self) -> Set[str]:
        urls: Set[str] = set()
        for item in self.catalog:
            link = item.get("link")
            if isinstance(link, str) and link.strip():
                urls.add(link.strip())
        return urls

    def _fuzzy_match(self, value: str, choices: Iterable[str]) -> List[str]:
        if not value:
            return []
        choice_list = sorted(choices)
        results = process.extract(
            value,
            choice_list,
            scorer=fuzz.token_set_ratio,
            limit=3,
            score_cutoff=55,
        )
        return [result[0] for result in results]

    def match_job_levels(self, value: str) -> List[str]:
        return self._fuzzy_match(value, self.job_levels)

    def match_keys(self, value: str) -> List[str]:
        return self._fuzzy_match(value, self.keys)
