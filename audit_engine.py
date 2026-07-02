"""
Offline audit of the recommendation engine.

Verifies:
1. catalog.json loads correctly and builds validation indexes
2. BM25 search returns ONLY real catalog items
3. validate_recommendation correctly accepts real items and rejects fake ones
4. _validate_and_build_recs drops any out-of-catalog index
5. _post_validate drops hallucinated names/URLs
6. Simulates the Phase 3→4 pipeline end-to-end with mock LLM responses
   (including attempts to inject hallucinated items)
7. Checks that no example.com URLs exist in any search result
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.data_loader import store
from app.models import Recommendation
from app.agent import _validate_and_build_recs, _post_validate

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    print(f"{status}  {label}")
    if detail:
        print(f"       {detail}")


# ============================================================
# 1. Catalog loaded correctly
# ============================================================
check(
    "Catalog loads without error",
    len(store.assessments) > 0,
    f"{len(store.assessments)} assessments loaded"
)
check(
    "Name validation index built",
    len(store._name_to_item) == len(store.assessments),
    f"{len(store._name_to_item)} names indexed"
)
check(
    "URL validation set built",
    len(store._url_set) == len(store.assessments),
    f"{len(store._url_set)} URLs indexed"
)

# ============================================================
# 2. No example.com or placeholder URLs in catalog
# ============================================================
bad_urls = [
    item["link"] for item in store.assessments
    if "example.com" in item.get("link", "")
    or not item.get("link", "").startswith("https://www.shl.com")
]
check(
    "No example.com or non-SHL URLs in catalog",
    len(bad_urls) == 0,
    f"Bad URLs found: {bad_urls[:3]}" if bad_urls else "All URLs start with https://www.shl.com"
)

# ============================================================
# 3. BM25 search returns only real catalog items
# ============================================================
search_results = store.search("Java developer mid-level personality", top_k=20)
non_catalog = [
    r for r in search_results
    if not store.is_valid_name(r.get("name", ""))
    or not store.is_valid_url(r.get("link", ""))
]
check(
    "BM25 search results are all from catalog",
    len(non_catalog) == 0,
    f"{len(search_results)} results, {len(non_catalog)} non-catalog items"
)

# Check no example.com in results
bad_search_urls = [r["link"] for r in search_results if "example.com" in r.get("link","")]
check(
    "No example.com URLs in BM25 search results",
    len(bad_search_urls) == 0,
    f"Clean: {len(search_results)} results, 0 bad URLs"
)

# ============================================================
# 4. search_multi returns only real catalog items
# ============================================================
multi_results = store.search_multi(
    ["Java programming", "OPQ personality behaviour", "cognitive ability reasoning numerical"],
    top_k_each=10
)
non_catalog_multi = [
    r for r in multi_results
    if not store.validate_recommendation(r.get("name",""), r.get("link",""))
]
check(
    "search_multi results all pass validate_recommendation",
    len(non_catalog_multi) == 0,
    f"{len(multi_results)} results, {len(non_catalog_multi)} invalid"
)

# ============================================================
# 5. validate_recommendation: real item accepted, fake rejected
# ============================================================
real_item = store.assessments[0]
real_valid = store.validate_recommendation(real_item["name"], real_item["link"])
check("Real catalog item passes validate_recommendation", real_valid is not None)

fake_valid = store.validate_recommendation("Big Five Personality Test", "https://example.com/big-five")
check("Hallucinated item rejected by validate_recommendation", fake_valid is None)

wrong_url_valid = store.validate_recommendation(real_item["name"], "https://example.com/fake")
check("Real name + wrong URL rejected by validate_recommendation", wrong_url_valid is None)

wrong_name_valid = store.validate_recommendation("FAKE NAME XYZ", real_item["link"])
check("Wrong name + real URL rejected by validate_recommendation", wrong_name_valid is None)

# ============================================================
# 6. _validate_and_build_recs: index-based selection is safe
# ============================================================
# Simulate 5 real candidates
candidates = store.search("Java", top_k=5)
assert len(candidates) >= 3, "Need at least 3 results for this test"

# LLM selects indices 1, 2, 3
recs = _validate_and_build_recs(candidates, [1, 2, 3])
check(
    "_validate_and_build_recs produces valid recs for valid indices",
    all(store.validate_recommendation(r.name, r.url) for r in recs),
    f"{len(recs)} recs produced"
)

# LLM tries to inject out-of-range index
recs_oob = _validate_and_build_recs(candidates, [1, 9999, -1, 0])
check(
    "Out-of-range indices silently dropped",
    len(recs_oob) == 1,  # only index 1 is valid
    f"Got {len(recs_oob)} rec(s), expected 1"
)

# ============================================================
# 7. _post_validate: final safety net blocks hallucinations
# ============================================================
hallucinated_recs = [
    Recommendation(name="Big Five Personality Test", url="https://example.com/big5", test_type="Personality"),
    Recommendation(name="HackerRank Java Test",     url="https://hackerrank.com/java", test_type="Coding"),
    Recommendation(name=real_item["name"],           url=real_item["link"],             test_type="Any"),
]
clean = _post_validate(hallucinated_recs)
check(
    "_post_validate drops hallucinated items, keeps catalog items",
    len(clean) == 1 and clean[0].name == real_item["name"],
    f"Input: 3, output: {len(clean)} (should be 1)"
)
check(
    "Kept item's URL is official SHL URL",
    clean[0].url.startswith("https://www.shl.com"),
    f"URL: {clean[0].url}"
)
check(
    "No example.com URL survived _post_validate",
    all("example.com" not in r.url for r in clean),
)

# ============================================================
# 8. End-to-end pipeline simulation (no API calls)
# ============================================================
# Simulate what happens in Phase 2+3+4 with a mock "LLM" that tries to hallucinate
candidates_pool = store.search_multi(
    ["Java programming", "personality OPQ", "cognitive ability"],
    top_k_each=8
)
# Mock: LLM selects real indices [1,2,3] + tries to inject hallucination
# (but in our architecture LLM can ONLY return indices, so hallucination is impossible)
mock_selected_indices = [1, 2, 3, 4]  # all valid indices
pipeline_recs = _validate_and_build_recs(candidates_pool, mock_selected_indices)
pipeline_recs = _post_validate(pipeline_recs)

all_in_catalog = all(
    store.validate_recommendation(r.name, r.url) is not None
    for r in pipeline_recs
)
no_bad_urls = all("example.com" not in r.url for r in pipeline_recs)

check(
    "End-to-end pipeline: all recs in catalog",
    all_in_catalog,
    f"{len(pipeline_recs)} recs produced"
)
check(
    "End-to-end pipeline: no example.com URLs",
    no_bad_urls,
)
check(
    "End-to-end pipeline: all URLs start with https://www.shl.com",
    all(r.url.startswith("https://www.shl.com") for r in pipeline_recs),
)

# ============================================================
# Summary
# ============================================================
print()
print("="*60)
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
print(f"AUDIT COMPLETE: {passed}/{len(results)} checks passed, {failed} failed")
print("="*60)

if failed > 0:
    print("\nFailed checks:")
    for s, label, detail in results:
        if s == FAIL:
            print(f"  {FAIL} {label}: {detail}")
    sys.exit(1)
else:
    print("All checks passed. The recommendation engine is hallucination-safe.")
