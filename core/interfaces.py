"""Structural interfaces the core managers depend on.

These are typing.Protocol definitions — a store satisfies them by shape, no
inheritance required. They make the persistence layer an explicit swap point
(e.g. swapping storage engines, or a fake store in tests) rather than an
implicit one.
"""
from typing import Protocol

from core.models import MemoryRecord


class MemoryStoreProtocol(Protocol):
    """What MemoryManager needs from a memory store."""

    def setup(self) -> None: ...

    def store_memory(self, record: MemoryRecord) -> bool: ...

    def memory_exists(self, query_embedding, threshold: float = ...) -> bool: ...

    def fetch_memories(
        self, query_embedding, threshold: float = ..., limit: int = ...
    ) -> list[MemoryRecord]: ...
