"""Agent memory system — working, episodic, semantic, shared, procedural.

Each memory type serves a different purpose and has different
retention characteristics. The manager provides a unified
interface for storing, retrieving, and consolidating memories.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    key: str
    content: Any
    memory_type: str  # working, episodic, semantic, shared, procedural
    agent_id: str
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600


class WorkingMemory:
    """Short-term memory for the current task. Cleared after task completes."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._store: dict[str, MemoryEntry] = {}

    def set(self, key: str, value: Any):
        self._store[key] = MemoryEntry(
            key=key, content=value, memory_type="working",
            agent_id=self.agent_id, ttl_seconds=3600,  # 1 hour default
        )

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry and not entry.expired:
            entry.accessed_at = time.time()
            entry.access_count += 1
            return entry.content
        return None

    def clear(self):
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


class EpisodicMemory:
    """Records past interactions and their outcomes. Decays over weeks."""

    def __init__(self, agent_id: str, max_episodes: int = 1000):
        self.agent_id = agent_id
        self.max_episodes = max_episodes
        self._episodes: list[MemoryEntry] = []

    def store(self, interaction_id: str, context: dict, outcome: str,
              contact_id: str = "", metadata: dict = None):
        entry = MemoryEntry(
            key=interaction_id,
            content={
                "context": context, "outcome": outcome,
                "contact_id": contact_id,
            },
            memory_type="episodic",
            agent_id=self.agent_id,
            ttl_seconds=14 * 86400,  # 2 weeks default
            metadata=metadata or {},
        )
        self._episodes.append(entry)

        # Prune if over limit
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]

    def recall(self, contact_id: str = None, limit: int = 10) -> list[dict]:
        """Recall recent non-expired episodes, optionally filtered by contact."""
        valid = [e for e in self._episodes if not e.expired]
        if contact_id:
            valid = [e for e in valid if e.content.get("contact_id") == contact_id]

        # Most recent first
        valid.sort(key=lambda e: e.created_at, reverse=True)
        return [e.content for e in valid[:limit]]

    def successful_episodes(self, limit: int = 50) -> list[dict]:
        """Recall episodes with positive outcomes — used for pattern extraction."""
        positive = ["replied_positive", "meeting_scheduled", "deal_closed",
                     "followed", "clicked", "invested"]
        valid = [
            e for e in self._episodes
            if not e.expired and e.content.get("outcome") in positive
        ]
        valid.sort(key=lambda e: e.created_at, reverse=True)
        return [e.content for e in valid[:limit]]

    @property
    def size(self) -> int:
        return len([e for e in self._episodes if not e.expired])


class SemanticMemory:
    """Long-term learned facts about contacts, companies, preferences.
    Persists until explicitly updated or contradicted."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._facts: dict[str, MemoryEntry] = {}

    def learn(self, key: str, fact: Any, confidence: float = 0.8,
              source: str = ""):
        existing = self._facts.get(key)
        if existing:
            # Update if new info has higher confidence
            old_conf = existing.metadata.get("confidence", 0)
            if confidence >= old_conf:
                existing.content = fact
                existing.metadata["confidence"] = confidence
                existing.metadata["source"] = source
                existing.accessed_at = time.time()
        else:
            self._facts[key] = MemoryEntry(
                key=key, content=fact, memory_type="semantic",
                agent_id=self.agent_id,
                metadata={"confidence": confidence, "source": source},
            )

    def recall(self, key: str) -> Optional[Any]:
        entry = self._facts.get(key)
        if entry:
            entry.accessed_at = time.time()
            entry.access_count += 1
            return entry.content
        return None

    def search(self, query_terms: set[str]) -> list[tuple[str, Any, float]]:
        """Simple term-matching search over semantic memory."""
        results = []
        for key, entry in self._facts.items():
            content_str = str(entry.content).lower()
            key_lower = key.lower()
            matches = sum(1 for t in query_terms if t in content_str or t in key_lower)
            if matches > 0:
                score = matches / len(query_terms) if query_terms else 0
                results.append((key, entry.content, round(score, 3)))
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    @property
    def size(self) -> int:
        return len(self._facts)


class SharedMemory:
    """Cross-agent memory bus. Agents publish; other agents subscribe.
    No agent can modify another agent's private memory."""

    def __init__(self):
        self._store: dict[str, MemoryEntry] = {}

    def publish(self, key: str, data: Any, publisher: str,
                available_to: list[str] = None, ttl_hours: float = 24):
        self._store[key] = MemoryEntry(
            key=key, content=data, memory_type="shared",
            agent_id=publisher,
            ttl_seconds=ttl_hours * 3600,
            metadata={"available_to": available_to or [], "publisher": publisher},
        )

    def read(self, key: str, reader: str) -> Optional[Any]:
        entry = self._store.get(key)
        if not entry or entry.expired:
            return None
        allowed = entry.metadata.get("available_to", [])
        if allowed and reader not in allowed:
            log.warning(f"shared memory access denied: {reader} not in {allowed}")
            return None
        entry.accessed_at = time.time()
        return entry.content

    def list_keys(self, reader: str) -> list[str]:
        return [
            k for k, e in self._store.items()
            if not e.expired and (
                not e.metadata.get("available_to") or reader in e.metadata["available_to"]
            )
        ]


class ProceduralMemory:
    """Learned patterns and templates from successful interactions.
    These are reusable "how to do X" patterns extracted from episodic memory."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._patterns: dict[str, MemoryEntry] = {}

    def store_pattern(self, name: str, pattern: dict, success_count: int = 1):
        existing = self._patterns.get(name)
        if existing:
            existing.metadata["success_count"] = existing.metadata.get("success_count", 0) + success_count
            existing.content = pattern  # Update with latest version
        else:
            self._patterns[name] = MemoryEntry(
                key=name, content=pattern, memory_type="procedural",
                agent_id=self.agent_id,
                metadata={"success_count": success_count},
            )

    def get_pattern(self, name: str) -> Optional[dict]:
        entry = self._patterns.get(name)
        return entry.content if entry else None

    def best_patterns(self, limit: int = 5) -> list[tuple[str, dict, int]]:
        """Return patterns sorted by success count."""
        items = [
            (k, e.content, e.metadata.get("success_count", 0))
            for k, e in self._patterns.items()
        ]
        items.sort(key=lambda x: x[2], reverse=True)
        return items[:limit]

    @property
    def size(self) -> int:
        return len(self._patterns)


class MemoryManager:
    """Unified interface for all memory types."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.working = WorkingMemory(agent_id)
        self.episodic = EpisodicMemory(agent_id)
        self.semantic = SemanticMemory(agent_id)
        self.procedural = ProceduralMemory(agent_id)

    def stats(self) -> dict:
        return {
            "agent": self.agent_id,
            "working": self.working.size,
            "episodic": self.episodic.size,
            "semantic": self.semantic.size,
            "procedural": self.procedural.size,
        }
