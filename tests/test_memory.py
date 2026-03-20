import time
import pytest
from memory.manager import (
    MemoryManager, WorkingMemory, EpisodicMemory,
    SemanticMemory, SharedMemory, ProceduralMemory,
)


class TestWorkingMemory:
    def test_set_and_get(self):
        wm = WorkingMemory("agent1")
        wm.set("key1", {"data": "value"})
        assert wm.get("key1") == {"data": "value"}

    def test_missing_key(self):
        wm = WorkingMemory("agent1")
        assert wm.get("missing") is None

    def test_clear(self):
        wm = WorkingMemory("agent1")
        wm.set("k1", 1)
        wm.set("k2", 2)
        wm.clear()
        assert wm.size == 0


class TestEpisodicMemory:
    def test_store_and_recall(self):
        em = EpisodicMemory("agent1")
        em.store("i1", {"style": "roi"}, "positive", "c1")
        em.store("i2", {"style": "technical"}, "negative", "c1")
        results = em.recall(contact_id="c1")
        assert len(results) == 2

    def test_filter_by_contact(self):
        em = EpisodicMemory("agent1")
        em.store("i1", {}, "positive", "c1")
        em.store("i2", {}, "positive", "c2")
        assert len(em.recall(contact_id="c1")) == 1

    def test_successful_episodes(self):
        em = EpisodicMemory("agent1")
        em.store("i1", {}, "replied_positive", "c1")
        em.store("i2", {}, "no_response", "c1")
        em.store("i3", {}, "meeting_scheduled", "c2")
        assert len(em.successful_episodes()) == 2

    def test_max_episodes(self):
        em = EpisodicMemory("agent1", max_episodes=5)
        for i in range(10):
            em.store(f"i{i}", {}, "positive", "c1")
        assert len(em._episodes) == 5


class TestSemanticMemory:
    def test_learn_and_recall(self):
        sm = SemanticMemory("agent1")
        sm.learn("sarah:channel", "email", 0.9)
        assert sm.recall("sarah:channel") == "email"

    def test_higher_confidence_updates(self):
        sm = SemanticMemory("agent1")
        sm.learn("sarah:channel", "email", 0.7)
        sm.learn("sarah:channel", "linkedin", 0.9)  # higher confidence
        assert sm.recall("sarah:channel") == "linkedin"

    def test_lower_confidence_no_update(self):
        sm = SemanticMemory("agent1")
        sm.learn("sarah:channel", "email", 0.9)
        sm.learn("sarah:channel", "phone", 0.5)  # lower confidence
        assert sm.recall("sarah:channel") == "email"

    def test_search(self):
        sm = SemanticMemory("agent1")
        sm.learn("sarah:channel", "email")
        sm.learn("sarah:interest", "AI solutions")
        sm.learn("david:channel", "linkedin")
        results = sm.search({"sarah"})
        assert len(results) == 2


class TestSharedMemory:
    def test_publish_and_read(self):
        sm = SharedMemory()
        sm.publish("research:acme", {"revenue": "42M"}, "research")
        assert sm.read("research:acme", "outreach") == {"revenue": "42M"}

    def test_access_control(self):
        sm = SharedMemory()
        sm.publish("data", "secret", "research", available_to=["outreach"])
        assert sm.read("data", "outreach") == "secret"
        assert sm.read("data", "content") is None

    def test_list_keys(self):
        sm = SharedMemory()
        sm.publish("k1", "v1", "agent1", available_to=["agent2"])
        sm.publish("k2", "v2", "agent1")  # available to all
        keys = sm.list_keys("agent2")
        assert "k1" in keys
        assert "k2" in keys


class TestProceduralMemory:
    def test_store_and_get(self):
        pm = ProceduralMemory("agent1")
        pm.store_pattern("roi_pitch", {"template": "ROI first"}, 5)
        assert pm.get_pattern("roi_pitch") == {"template": "ROI first"}

    def test_success_count_accumulates(self):
        pm = ProceduralMemory("agent1")
        pm.store_pattern("p1", {"t": "test"}, 3)
        pm.store_pattern("p1", {"t": "updated"}, 5)
        best = pm.best_patterns()
        assert best[0][2] == 8  # 3 + 5

    def test_best_patterns_sorted(self):
        pm = ProceduralMemory("agent1")
        pm.store_pattern("low", {}, 2)
        pm.store_pattern("high", {}, 10)
        pm.store_pattern("mid", {}, 5)
        best = pm.best_patterns()
        assert best[0][0] == "high"


class TestMemoryManager:
    def test_stats(self):
        mm = MemoryManager("test")
        mm.working.set("k", "v")
        mm.episodic.store("i1", {}, "ok", "c1")
        mm.semantic.learn("fact", "value")
        mm.procedural.store_pattern("p1", {})
        stats = mm.stats()
        assert stats["working"] == 1
        assert stats["episodic"] == 1
        assert stats["semantic"] == 1
        assert stats["procedural"] == 1
