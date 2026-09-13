import unittest
from unittest.mock import patch

import httpx

from app.core.config import Settings
from app.services.vector_store import ChromaKnowledgeStore, VectorStoreUnavailable


class EmbeddingBatchTests(unittest.TestCase):
    def store(self):
        store = object.__new__(ChromaKnowledgeStore)
        store.settings = Settings(_env_file=None, embedding_batch_size=2, openai_base_url="https://example.invalid/v1/")
        return store

    def test_batch_limit_and_output_order(self):
        requests = []

        def post(url, *, headers, json, timeout):
            requests.append(json)
            self.assertEqual(url, "https://example.invalid/v1/embeddings")
            self.assertEqual(json["encoding_format"], "float")
            rows = [{"index": i, "embedding": [float(text)]} for i, text in enumerate(json["input"])]
            return httpx.Response(200, request=httpx.Request("POST", url), json={"data": rows[::-1]})

        with patch("app.services.vector_store.httpx.post", side_effect=post):
            self.assertEqual(self.store()._embed(["1", "2", "3", "4", "5"]), [[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.assertEqual([len(r["input"]) for r in requests], [2, 2, 1])

    def test_missing_embedding_aborts(self):
        response = httpx.Response(200, request=httpx.Request("POST", "https://example.invalid"), json={"data": []})
        with patch("app.services.vector_store.httpx.post", return_value=response):
            with self.assertRaises(VectorStoreUnavailable):
                self.store()._embed(["1"])


if __name__ == "__main__":
    unittest.main()
