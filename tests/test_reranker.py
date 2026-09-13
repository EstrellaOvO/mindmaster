import unittest
from unittest.mock import patch

import httpx

from app.core.config import Settings
from app.services.reranker import rerank_documents


class RerankerTests(unittest.TestCase):
    def settings(self):
        return Settings(_env_file=None, openai_api_key="test", rerank_url="https://example.invalid/rerank")

    def response(self, rows):
        return httpx.Response(200, request=httpx.Request("POST", "https://example.invalid"),
                              json={"output": {"results": rows}})

    def test_maps_indices_and_sorts_scores(self):
        rows = [{"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9}]
        with patch("app.services.reranker.httpx.post", return_value=self.response(rows)) as post:
            self.assertEqual(rerank_documents(self.settings(), "q", ["a", "b"], 4), [(1, 0.9), (0, 0.1)])
            self.assertEqual(post.call_args.kwargs["json"]["parameters"]["top_n"], 2)

    def test_invalid_results_fail_instead_of_falling_back(self):
        cases = [[], [{"index": 9, "relevance_score": 1}], [{"index": 0, "relevance_score": float("nan")}]]
        for rows in cases:
            # Mock the JSON method so nonfinite values can be tested without JSON serialization.
            with patch("app.services.reranker.httpx.post") as post:
                post.return_value.json.return_value = {"output": {"results": rows}}
                with self.assertRaises(ValueError):
                    rerank_documents(self.settings(), "q", ["a"], 1)

    def test_duplicate_indices_are_rejected(self):
        rows = [{"index": 0, "relevance_score": 0.9}] * 2
        with patch("app.services.reranker.httpx.post", return_value=self.response(rows)):
            with self.assertRaises(ValueError):
                rerank_documents(self.settings(), "q", ["a", "b"], 2)
