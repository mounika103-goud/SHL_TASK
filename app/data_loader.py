import json
import os
from typing import List, Dict, Any, Optional, Set
from rank_bm25 import BM25Okapi
import re

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "catalog.json")


class CatalogStore:
    def __init__(self):
        self.assessments: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_tokens: List[List[str]] = []

        # Fast O(1) validation lookups
        self._name_to_item: Dict[str, Dict[str, Any]] = {}   # lowercased name -> item
        self._url_set: Set[str] = set()                        # set of valid URLs

        self._load_data()

    def _load_data(self):
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                self.assessments = json.load(f, strict=False)

            for item in self.assessments:
                # Build BM25 corpus
                text = (
                    f"{item.get('name', '')} "
                    f"{item.get('description', '')} "
                    f"{' '.join(item.get('keys', []))} "
                    f"{' '.join(item.get('job_levels', []))}"
                )
                self.corpus_tokens.append(self._tokenize(text))

                # Build validation indexes
                name_key = item.get("name", "").strip().lower()
                if name_key:
                    self._name_to_item[name_key] = item

                url = item.get("link", "").strip()
                if url:
                    self._url_set.add(url)

            if self.corpus_tokens:
                self.bm25 = BM25Okapi(self.corpus_tokens)

        except Exception as e:
            print(f"[catalog] Failed to load: {e}")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """BM25 search over name, description, keys, and job levels."""
        if not self.bm25 or not query:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.assessments[i] for i in top_indices if scores[i] > 0]

    def search_multi(self, queries: List[str], top_k_each: int = 10) -> List[Dict[str, Any]]:
        """
        Run multiple BM25 searches and merge results de-duplicated by entity_id.
        Useful for building a diverse candidate pool covering multiple test types.
        """
        seen_ids: Set[str] = set()
        results: List[Dict[str, Any]] = []
        for q in queries:
            for item in self.search(q, top_k=top_k_each):
                eid = item.get("entity_id", item.get("name", ""))
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    results.append(item)
        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def is_valid_name(self, name: str) -> bool:
        """Return True iff name matches a catalog entry (case-insensitive)."""
        return name.strip().lower() in self._name_to_item

    def is_valid_url(self, url: str) -> bool:
        """Return True iff url exactly matches a catalog entry URL."""
        return url.strip() in self._url_set

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Return the catalog item for the given name (case-insensitive), or None."""
        return self._name_to_item.get(name.strip().lower())

    def validate_recommendation(self, name: str, url: str) -> Optional[Dict[str, Any]]:
        """
        Return the canonical catalog item if both name and url are valid and match
        the same record. Otherwise return None.
        """
        item = self.get_by_name(name)
        if item is None:
            return None
        if item.get("link", "").strip() != url.strip():
            return None
        return item

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    def format_for_prompt(self, items: List[Dict[str, Any]], *, numbered: bool = True) -> str:
        """
        Format a list of catalog items for injection into a prompt.
        If numbered=True, each item gets an index (used for selection).
        """
        lines = []
        for idx, item in enumerate(items, start=1):
            prefix = f"[{idx}] " if numbered else "- "
            lines.append(
                f"{prefix}Name: {item.get('name')}\n"
                f"   URL: {item.get('link')}\n"
                f"   Type: {', '.join(item.get('keys', []))}\n"
                f"   Job Levels: {', '.join(item.get('job_levels', []))}\n"
                f"   Duration: {item.get('duration', 'N/A')}\n"
                f"   Description: {item.get('description', '')[:200]}"
            )
        return "\n\n".join(lines)


# Singleton
store = CatalogStore()
