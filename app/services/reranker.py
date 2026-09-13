import math

import httpx


def rerank_documents(settings, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
    """DashScope text reranking. Invalid results fail explicitly, without lexical fallback."""
    if not documents or top_k <= 0:
        return []
    if not settings.openai_api_key:
        raise ValueError("Model reranking requires an API key")
    count = min(top_k, len(documents))
    response = httpx.post(
        settings.rerank_url,
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={"model": settings.rerank_model,
              "input": {"query": query, "documents": documents},
              "parameters": {"top_n": count}},
        timeout=settings.rerank_timeout_seconds,
    )
    response.raise_for_status()
    rows = response.json().get("output", {}).get("results", [])
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Reranker returned an unexpected result count")
    results = []
    seen = set()
    for row in rows:
        index = row.get("index")
        score = float(row["relevance_score"])
        if type(index) is not int or not 0 <= index < len(documents) or index in seen or not math.isfinite(score):
            raise ValueError("Reranker returned an invalid index or score")
        seen.add(index)
        results.append((index, score))
    return sorted(results, key=lambda item: item[1], reverse=True)
