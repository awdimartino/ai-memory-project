"""Embedding via LM Studio's OpenAI-compatible /embeddings endpoint.

Uses asymmetric task prefixes (nomic-embed style: "search_query:" for the
incoming message, "search_document:" for stored facts). This directly targets
the v1 perspective-mismatch bug, where a first-person query embedded poorly
against a third-person stored memory.
"""
import logging

logger = logging.getLogger(__name__)

_QUERY_PREFIX = "search_query: "
_DOC_PREFIX = "search_document: "


class Embedder:
    def __init__(self, client, model: str):
        self.client = client  # shared AsyncOpenAI
        self.model = model

    async def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        resp = await self.client.embeddings.create(
            model=self.model, input=[prefix + t for t in texts]
        )
        return [d.embedding for d in resp.data]

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], _QUERY_PREFIX))[0]

    async def embed_document(self, text: str) -> list[float]:
        return (await self._embed([text], _DOC_PREFIX))[0]
