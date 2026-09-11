"""
Retriever module: indexes historical @AmazonHelp brand resolutions
and retrieves the most relevant exemplars using BM25 and intent filtering.
"""
import json
import os
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.config import AgentConfig


class HistoricalExemplarRetriever:
    def __init__(self, corpus_path: Optional[str] = None, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        if corpus_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            corpus_path = os.path.join(base_dir, "data", "raw_brand_corpus.json")
        self.corpus_path = corpus_path
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load_and_index()

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def _load_and_index(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(f"Corpus file not found at {self.corpus_path}")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.tokenized_corpus = [
            self._tokenize(f"{doc.get('query_summary', '')} {doc.get('historical_reply', '')} {doc.get('resolution_category', '')}")
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, intent: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k historical resolutions, with priority given to matching intent.
        """
        tokenized_query = self._tokenize(query)
        if not tokenized_query or self.bm25 is None:
            return self.documents[:top_k]

        scores = self.bm25.get_scores(tokenized_query)

        # Apply intent boost
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        # First check matching intent
        for idx in ranked_indices:
            doc = self.documents[idx]
            if intent is None or doc.get("intent") == intent:
                results.append({
                    **doc,
                    "bm25_score": float(scores[idx])
                })
                if len(results) >= top_k:
                    break

        # If not enough matches within same intent, backfill from top BM25
        if len(results) < top_k:
            for idx in ranked_indices:
                doc = self.documents[idx]
                if doc["id"] not in [r["id"] for r in results]:
                    results.append({
                        **doc,
                        "bm25_score": float(scores[idx])
                    })
                    if len(results) >= top_k:
                        break

        return results
