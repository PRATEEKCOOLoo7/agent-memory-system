# Agent Memory System

A structured memory layer for autonomous AI agents that need to remember context across conversations, learn from past interactions, and share knowledge between agents in a multi-agent system.

Agents without memory are stateless workers. Agents with memory are colleagues that get better over time.

## Memory Types

```
┌────────────────────────────────────────────────────────────────┐
│                     AGENT MEMORY SYSTEM                        │
│                                                                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │  Working     │  │  Episodic    │  │  Semantic             │ │
│  │  Memory      │  │  Memory      │  │  Memory               │ │
│  │             │  │              │  │                       │ │
│  │  Current    │  │  Past inter- │  │  Learned facts,       │ │
│  │  context,   │  │  actions,    │  │  preferences,         │ │
│  │  active     │  │  outcomes,   │  │  domain knowledge,    │ │
│  │  task state │  │  what worked │  │  entity relationships │ │
│  │             │  │              │  │                       │ │
│  │  TTL: task  │  │  TTL: weeks  │  │  TTL: permanent       │ │
│  └─────────────┘  └──────────────┘  └───────────────────────┘ │
│                                                                │
│  ┌──────────────────────┐  ┌────────────────────────────────┐ │
│  │  Shared Memory       │  │  Procedural Memory             │ │
│  │                      │  │                                │ │
│  │  Cross-agent context │  │  How to do things:             │ │
│  │  (Research agent's   │  │  successful prompt templates,  │ │
│  │  findings available  │  │  effective outreach patterns,  │ │
│  │  to Outreach agent)  │  │  learned workflows             │ │
│  └──────────────────────┘  └────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

## Why This Matters for Revenue AI

In sales and revenue workflows, memory is the difference between a cold robot and a smart colleague:

| Without Memory | With Memory |
|---|---|
| "Hi Sarah" (who is Sarah?) | "Sarah prefers email over LinkedIn, responds best on Tuesdays, cares about AI-native solutions" |
| Same generic pitch every time | "Last time we led with ROI metrics and she engaged — do that again" |
| Each agent starts from scratch | Research agent's findings flow directly into Outreach agent's context |
| No learning from outcomes | "This style of message had 3x reply rate for fintech VPs" |

## Project Structure

```
agent-memory-system/
├── README.md
├── requirements.txt
├── memory/
│   ├── __init__.py
│   ├── memory_manager.py        # Unified memory interface
│   ├── working_memory.py        # Current task context (ephemeral)
│   ├── episodic_memory.py       # Past interactions + outcomes
│   ├── semantic_memory.py       # Learned facts + knowledge graph
│   ├── shared_memory.py         # Cross-agent context sharing
│   ├── procedural_memory.py     # Learned patterns + templates
│   └── consolidation.py         # Working → Episodic → Semantic promotion
├── storage/
│   ├── __init__.py
│   ├── vector_store.py          # Embedding-based retrieval (Pinecone)
│   ├── graph_store.py           # Entity relationships (Neo4j)
│   └── kv_store.py              # Fast key-value (Redis)
├── retrieval/
│   ├── __init__.py
│   ├── context_retriever.py     # Pull relevant memories for current task
│   ├── relevance_scorer.py      # Rank memories by relevance + recency
│   └── memory_compiler.py       # Compile memories into LLM context window
├── learning/
│   ├── __init__.py
│   ├── outcome_learner.py       # Learn from interaction outcomes
│   ├── pattern_extractor.py     # Extract reusable patterns from success
│   └── forgetting.py            # TTL-based memory decay + pruning
├── tests/
│   ├── test_memory_manager.py
│   ├── test_episodic.py
│   ├── test_shared_memory.py
│   └── test_consolidation.py
└── examples/
    ├── sales_agent_with_memory.py
    └── multi_agent_shared_memory.py
```

## Usage

```python
from memory import MemoryManager, WorkingMemory, EpisodicMemory

# Initialize memory for an agent
memory = MemoryManager(
    agent_id="outreach_agent",
    vector_store="pinecone",
    kv_store="redis",
)

# Store interaction outcome
memory.episodic.store(
    interaction_id="interaction_456",
    contact_id="sarah_chen",
    context={"message_style": "roi_focused", "channel": "email"},
    outcome="replied_positive",
    metadata={"reply_time_hours": 2.3},
)

# Retrieve relevant memories for a new task
memories = memory.retrieve(
    query="Crafting outreach to a fintech VP interested in AI",
    contact_id="sarah_chen",  # Optional: get contact-specific memories
    max_results=5,
)

# Compile into LLM context
context_block = memory.compile_for_llm(
    memories=memories,
    max_tokens=500,
    prioritize="recency_and_outcome",
)

# Share context with another agent
memory.shared.publish(
    key="research:acme_corp",
    data=research_findings,
    available_to=[AgentRole.OUTREACH, AgentRole.CONTENT],
    ttl_hours=24,
)
```

## Memory Consolidation

Memories promote from volatile to permanent through a consolidation process inspired by human memory:

```
Working Memory (current task)
    │ task completes
    ▼
Episodic Memory (what happened, what worked)
    │ patterns emerge across episodes
    ▼
Semantic Memory (learned facts: "Sarah prefers email")
    │ repeated success
    ▼
Procedural Memory (templates: "ROI-first pitch works for fintech VPs")
```

Each promotion step involves:
1. **Pattern detection**: Are there repeated signals across episodes?
2. **Confidence scoring**: How reliable is this pattern?
3. **Conflict resolution**: Does this contradict existing semantic memory?
4. **Decay management**: Old, low-confidence memories are pruned

## Design Decisions

- **Embedding + graph hybrid**: Vector similarity finds *related* memories. The knowledge graph finds *connected* memories (Sarah → works at → Acme → competitor of → Pearl). Both are needed.
- **TTL-based forgetting**: Not all memories should persist forever. Working memory dies with the task. Episodic memory decays over weeks unless reinforced by repeated interactions.
- **Shared memory bus, not shared state**: Agents publish to shared memory; other agents subscribe. No agent can modify another agent's private memory. This prevents cross-contamination.
- **Compile for context window**: LLM context is finite. The memory compiler ranks, truncates, and formats memories to fit within the token budget while preserving the most useful context.

