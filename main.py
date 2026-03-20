"""Agent Memory System — Demo

Shows all 5 memory types in action with realistic sales agent scenarios.
"""

import logging
from memory.manager import MemoryManager, SharedMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s", datefmt="%H:%M:%S")


def main():
    print(f"\n{'='*60}")
    print("  Agent Memory System — Demo")
    print(f"{'='*60}")

    outreach = MemoryManager("outreach_agent")
    research = MemoryManager("research_agent")
    shared = SharedMemory()

    # 1. Working memory — current task context
    print(f"\n--- Working Memory ---")
    outreach.working.set("current_lead", {"name": "Sarah Chen", "company": "Acme Corp"})
    outreach.working.set("task", "draft_followup_email")
    lead = outreach.working.get("current_lead")
    print(f"  Current lead: {lead}")
    print(f"  Working memory size: {outreach.working.size}")

    # 2. Episodic memory — past interactions
    print(f"\n--- Episodic Memory ---")
    outreach.episodic.store("int_001", {"message_style": "roi_focused", "channel": "email"}, "replied_positive", "sarah_chen")
    outreach.episodic.store("int_002", {"message_style": "technical", "channel": "email"}, "no_response", "sarah_chen")
    outreach.episodic.store("int_003", {"message_style": "roi_focused", "channel": "linkedin"}, "meeting_scheduled", "sarah_chen")
    outreach.episodic.store("int_004", {"message_style": "casual", "channel": "email"}, "replied_positive", "david_kim")

    sarah_history = outreach.episodic.recall(contact_id="sarah_chen")
    print(f"  Sarah's interactions: {len(sarah_history)}")
    for h in sarah_history:
        print(f"    {h['context']['message_style']} via {h['context']['channel']} → {h['outcome']}")

    successful = outreach.episodic.successful_episodes()
    print(f"  Successful episodes: {len(successful)}")

    # 3. Semantic memory — learned facts
    print(f"\n--- Semantic Memory ---")
    outreach.semantic.learn("sarah_chen:preferred_channel", "email", 0.9, "interaction_history")
    outreach.semantic.learn("sarah_chen:best_day", "Tuesday", 0.7, "response_patterns")
    outreach.semantic.learn("sarah_chen:interest", "AI-native solutions", 0.85, "conversation_analysis")
    outreach.semantic.learn("acme_corp:tech_stack", ["Salesforce", "Snowflake", "dbt"], 0.8, "crm_data")

    pref = outreach.semantic.recall("sarah_chen:preferred_channel")
    print(f"  Sarah's preferred channel: {pref}")

    results = outreach.semantic.search({"sarah", "channel", "interest"})
    print(f"  Search 'sarah channel interest': {len(results)} results")
    for key, val, score in results:
        print(f"    [{score:.2f}] {key}: {val}")

    # 4. Shared memory — cross-agent
    print(f"\n--- Shared Memory ---")
    shared.publish(
        "research:acme_corp",
        {"revenue": "$42M", "growth": "15% YoY", "news": ["New CTO from Google", "AI initiative launched"]},
        publisher="research_agent",
        available_to=["outreach_agent", "analysis_agent"],
        ttl_hours=24,
    )

    acme_data = shared.read("research:acme_corp", "outreach_agent")
    print(f"  Outreach reads research on Acme: {acme_data}")

    blocked = shared.read("research:acme_corp", "content_agent")
    print(f"  Content agent (not authorized): {blocked}")

    # 5. Procedural memory — learned patterns
    print(f"\n--- Procedural Memory ---")
    outreach.procedural.store_pattern("roi_first_for_fintech_vps", {
        "template": "Lead with ROI metrics, mention specific competitor comparison",
        "channel": "email", "audience": "VP-level fintech",
        "avg_reply_rate": 0.34,
    }, success_count=12)
    outreach.procedural.store_pattern("technical_deep_dive", {
        "template": "Lead with architecture details, reference their tech stack",
        "channel": "email", "audience": "CTO/engineering",
        "avg_reply_rate": 0.28,
    }, success_count=8)

    best = outreach.procedural.best_patterns()
    print(f"  Best patterns:")
    for name, pattern, count in best:
        print(f"    {name}: {count} successes, {pattern['avg_reply_rate']:.0%} reply rate")

    # Stats
    print(f"\n--- Stats ---")
    print(f"  Outreach agent: {outreach.stats()}")
    print(f"  Research agent: {research.stats()}")

    print(f"\n{'='*60}")
    print("  Demo complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
