"""Score a QueryResult against a Query's ground truth."""

from pathlib import Path

from ..wrappers.base import Query, QueryResult


def _normalize(p: str) -> str:
    """Strip leading ./ and normalise slashes."""
    p = p.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def _paths_match(returned: str, truth: str) -> bool:
    """Fuzzy path match: exact after normalise, suffix match, or basename fallback."""
    r = _normalize(returned)
    t = _normalize(truth)

    if r == t:
        return True

    # Absolute/deep path contains the relative truth as a suffix
    if r.endswith("/" + t) or t.endswith("/" + r):
        return True

    # Basename match (last resort — only when basename is non-trivial)
    r_name = Path(r).name
    t_name = Path(t).name
    if r_name and t_name and r_name == t_name and "." in r_name:
        return True

    return False


def score_query(result: QueryResult, query: Query) -> dict:
    """Score result files against query ground truth.

    Returns a dict with: query_id, tool_name, mode, run_number,
    file_recall, file_precision, f1, keyword_coverage,
    optional_hits, anti_file_hits.
    """
    returned = result.files_returned
    truth = query.ground_truth

    # --- file recall ---
    matched_truth_count = 0
    for t in truth:
        if any(_paths_match(r, t) for r in returned):
            matched_truth_count += 1
    file_recall = matched_truth_count / len(truth) if truth else 1.0

    # --- file precision ---
    if returned:
        matched_returned_count = sum(
            1 for r in returned if any(_paths_match(r, t) for t in truth)
        )
        file_precision = matched_returned_count / len(returned)
    else:
        file_precision = 0.0 if truth else 1.0

    # --- f1 ---
    if file_recall + file_precision > 0:
        f1 = 2 * file_recall * file_precision / (file_recall + file_precision)
    else:
        f1 = 0.0

    # --- keyword coverage ---
    answer_lower = result.answer.lower()
    if query.keywords:
        found = sum(1 for kw in query.keywords if kw.lower() in answer_lower)
        keyword_coverage = found / len(query.keywords)
    else:
        keyword_coverage = 1.0

    # --- optional hits ---
    optional_hits = sum(
        1 for opt in query.optional_files
        if any(_paths_match(r, opt) for r in returned)
    )

    # --- anti-file hits (penalty indicator) ---
    anti_file_hits = sum(
        1 for anti in query.anti_files
        if any(_paths_match(r, anti) for r in returned)
    )

    return {
        "query_id": result.query_id,
        "tool_name": result.tool_name,
        "mode": result.mode,
        "run_number": result.run_number,
        "file_recall": file_recall,
        "file_precision": file_precision,
        "f1": f1,
        "keyword_coverage": keyword_coverage,
        "optional_hits": optional_hits,
        "anti_file_hits": anti_file_hits,
    }
