"""Strategy update workflow."""

import collections
from datetime import datetime, timedelta
from typing import List, Optional

from .io import BETS_DIR, JOURNAL_DIR, get_history, read_text, write_text
from .llm import complete
from .prompts import SYSTEM_ANALYST, UPDATE_STRATEGY_PROMPT, format_history_summary


def load_recent_journals(days: int = 7) -> str:
    """Load journal entries from the last N days."""
    entries = []
    today = datetime.now()

    for i in range(days):
        date = today - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        journal_path = JOURNAL_DIR / f"{date_str}.md"

        content = read_text(journal_path)
        if content:
            entries.append(f"### {date_str}\n{content}")

    if not entries:
        return "No recent journal entries."

    return "\n\n".join(entries)


def format_recent_bets(bets: List[dict]) -> str:
    """Format recent bets for the prompt."""
    if not bets:
        return "No completed bets yet."

    lines = []
    for bet in bets:
        result_emoji = "W" if bet["result"] == "win" else "L"
        lines.append(
            f"- [{result_emoji}] {bet['matchup']}: {bet['pick']} "
            f"({bet['confidence']}, {bet['units']}u) - {bet['primary_edge']}"
        )
        if bet.get("reflection"):
            lines.append(f"  Reflection: {bet['reflection']}")

    return "\n".join(lines)


def aggregate_reflections(bets: List[dict]) -> str:
    """Aggregate structured reflections into a pattern summary."""
    refs = [b["structured_reflection"] for b in bets if b.get("structured_reflection")]
    if not refs:
        return "No structured reflections available yet."

    total = len(refs)
    edge_valid_count = sum(1 for r in refs if r.get("edge_valid"))
    edge_invalid_count = total - edge_valid_count

    # Process assessments
    assessments = collections.Counter(r.get("process_assessment", "sound") for r in refs)

    # Most common missed factors
    all_missed = []
    for r in refs:
        all_missed.extend(r.get("missed_factors", []))
    missed_counter = collections.Counter(all_missed)
    top_missed = missed_counter.most_common(5)

    # Last 5 key lessons
    lessons = [r["key_lesson"] for r in refs[-5:] if r.get("key_lesson")]

    lines = [
        f"## Reflection Patterns ({total} bets analyzed)",
        f"- Edge validity: {edge_valid_count}/{total} ({edge_valid_count/total:.0%}) edges were valid",
        f"- Edge invalid: {edge_invalid_count}/{total}",
        "",
        "### Process Assessments",
    ]
    for assessment, count in assessments.most_common():
        lines.append(f"- {assessment}: {count} ({count/total:.0%})")

    if top_missed:
        lines.append("")
        lines.append("### Most Common Missed Factors")
        for factor, count in top_missed:
            lines.append(f"- {factor} ({count}x)")

    if lessons:
        lines.append("")
        lines.append("### Recent Key Lessons")
        for lesson in lessons:
            lines.append(f"- {lesson}")

    return "\n".join(lines)


async def generate_strategy(
    current: Optional[str],
    summary: dict,
    recent_bets: List[dict],
    recent_journals: str,
) -> Optional[str]:
    """Generate updated strategy document."""
    reflection_patterns = aggregate_reflections(recent_bets)

    prompt = UPDATE_STRATEGY_PROMPT.format(
        current_strategy=current or "No strategy defined yet.",
        history_summary=format_history_summary(summary),
        recent_bets=format_recent_bets(recent_bets),
        recent_journals=recent_journals,
        reflection_patterns=reflection_patterns,
        wins=summary.get("wins", 0),
        losses=summary.get("losses", 0),
        roi=round(summary.get("roi", 0) * 100, 1),
    )

    return await complete(prompt, system=SYSTEM_ANALYST)


async def run_strategy_workflow() -> None:
    """Run the strategy update workflow."""
    history = get_history()

    if history["summary"]["total_bets"] < 5:
        print(
            f"Need at least 5 completed bets to update strategy. "
            f"Currently have {history['summary']['total_bets']}."
        )
        return

    print("Loading context...")
    current = read_text(BETS_DIR / "strategy.md")
    recent_bets = history["bets"][-20:]
    recent_journals = load_recent_journals(days=7)

    print("Generating updated strategy...")
    new_strategy = await generate_strategy(
        current, history["summary"], recent_bets, recent_journals
    )

    if new_strategy:
        # Archive previous strategy before overwriting
        if current:
            versions_dir = BETS_DIR / "versions"
            versions_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            (versions_dir / f"strategy_{ts}.md").write_text(current)
            # Keep last 10
            for old in sorted(versions_dir.glob("strategy_*.md"), reverse=True)[10:]:
                old.unlink()
            print(f"  Archived previous strategy → versions/strategy_{ts}.md")

        write_text(BETS_DIR / "strategy.md", new_strategy)
        print("Updated bets/strategy.md")
        print("\n--- Preview ---")
        print(new_strategy[:500] + "..." if len(new_strategy) > 500 else new_strategy)
    else:
        print("Strategy generation failed. Check LLM errors above.")
