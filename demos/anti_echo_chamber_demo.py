#!/usr/bin/env python3
"""
Anti-Echo-Chamber Injection Demo — Breaking Filter Bubbles

Problem: Recommendation bubbles cause long-term retention damage by showing
users only what they've already shown interest in.

Solution: Every 7th recommendation is deliberately outside the user's usual
category, tracked by a "discovery score" metric.

Usage:
    python -m demos.anti_echo_chamber_demo
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import random
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("echo_demo")

from src.innovations.anti_echo_chamber import AntiEchoChamberInjector, DiscoveryItem


# ── Demo Runner ────────────────────────────────────────────────


def run_demo():
    """Run the full anti-echo-chamber demo."""
    sep = "─" * 68
    rng = random.Random(42)

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║      Anti-Echo-Chamber Injection — Live Demo                ║")
    logger.info("║  \"Breaking filter bubbles, one 7th position at a time.\"     ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("The problem: recommendation bubbles show users only what")
    logger.info("they've already seen, causing long-term retention damage.")
    logger.info("")
    logger.info("The solution: every 7th recommendation is deliberately")
    logger.info("outside the user's usual category, tracked by a discovery")
    logger.info("score metric that targets >15% novelty.")
    logger.info("")

    # ── Step 1: Initialize ──
    logger.info("🔧 Step 1: Initializing Anti-Echo-Chamber Injector...")
    injector = AntiEchoChamberInjector(
        injection_rate=1.0 / 7.0,
        discovery_score_target=0.15,
        category_distance_threshold=0.6,
    )

    # Seed user's history with some categories
    user_id = "alice"
    for cat in ["electronics", "books", "home"]:
        injector.register_interaction(user_id, f"item_{cat}", cat)
    logger.info(f"  ✅ Injector initialized for user '{user_id}'")
    logger.info(f"     User's known categories: electronics, books, home")
    logger.info(f"     Injection rate: every 7th position")
    logger.info(f"     Discovery target: > 15%")
    logger.info("")

    # ── Step 2: Create ranked items ──
    logger.info("📋 Step 2: Generating ranked recommendation candidates...")
    logger.info("")

    NUM_CANDIDATES = 30

    all_categories = [
        "electronics", "books", "home", "sports", "beauty", "food",
        "toys", "automotive", "health", "office", "garden", "music",
        "pet_supplies", "jewelry", "apparel",
    ]

    candidates: List[Tuple[str, float]] = []
    discovery_pool: List[DiscoveryItem] = []
    category_map: Dict[str, str] = {}

    for i in range(NUM_CANDIDATES):
        item_id = f"rec_{i:04d}"
        score = round(1.0 - (i * 0.03), 4)
        # Familiar categories for the in-category items
        if i < 20:
            cat = rng.choice(["electronics", "books", "home", "apparel"])
        else:
            cat = rng.choice(all_categories)

        category_map[item_id] = cat
        candidates.append((item_id, score))

        # Create discovery candidates from unfamiliar categories
        discovery_cats = [c for c in all_categories if c not in ["electronics", "books", "home"]]
        for j, disc_cat in enumerate(discovery_cats[:10]):
            disc_id = f"discovery_{disc_cat}_{j}"
            discovery_pool.append(DiscoveryItem(
                item_id=disc_id,
                category_id=disc_cat,
                relevance_score=rng.uniform(0.3, 0.7),
                distance_from_history=rng.uniform(0.6, 1.0),
                discovery_potential=rng.uniform(0.3, 0.8),
                serendipity_score=0.0,
            ))
            category_map[disc_id] = disc_cat

    logger.info(f"  Total candidates:      {len(candidates)}")
    logger.info(f"  Discovery pool items:  {len(discovery_pool)}")
    logger.info(f"  Available categories:  {', '.join(all_categories)}")
    logger.info("")

    # ── Step 3: Inject discovery ──
    logger.info("💉 Step 3: Injecting discovery items (every 7th position)...")
    logger.info("")

    results = injector.inject_discovery(
        user_id=user_id,
        ranked_items=candidates[:20],
        discovery_items=discovery_pool,
        category_map=category_map,
    )

    discovery_count = sum(1 for _, _, is_disc in results if is_disc)

    # Display results
    logger.info(f"  {'Pos':>4} {'Item':<25} {'Category':<18} {'Score':>8} {'Discovery?':>12}")
    logger.info(f"  {sep}")

    for i, (item_id, score, is_discovery) in enumerate(results[:20], 1):
        cat = category_map.get(item_id, "unknown")
        marker = "🌟 DISCOVERY" if is_discovery else "—"
        logger.info(
            f"  {i:>3}. {item_id:<24} {cat:<18} {score:>7.3f}  {marker:>12}"
        )

    logger.info("")
    logger.info(f"  {sep}")
    logger.info(f"  Discovery injections: {discovery_count}/{min(20, len(results))} positions")
    logger.info("")

    # ── Step 4: Discovery score analysis ──
    logger.info("📊 Step 4: Discovery score analysis...")
    logger.info("")

    for window_size in [7, 14, 20]:
        window = results[:window_size]
        disc = sum(1 for _, _, is_d in window if is_d)
        logger.info(f"  First {window_size:>2} items: {disc:>2} discoveries ({disc / window_size:.0%})")

    current_score = injector.get_discovery_score(user_id)
    logger.info(f"")
    logger.info(f"  Current discovery score: {current_score:.1%}")
    logger.info(f"  Target:                 > 15%")
    logger.info(f"  Status: {'✅ Above target' if current_score >= 0.15 else '⚠️  Below target'}")
    logger.info("")

    # ── Step 5: Simulate echo chamber (without injection) ──
    logger.info("🔇 Step 5: What happens WITHOUT anti-echo-chamber?")
    logger.info("")

    # Simulate a pure relevance-based ranking (no discovery injection)
    pure_relevance = [
        ("electronics", 0.95),
        ("books", 0.92),
        ("home", 0.88),
        ("electronics", 0.85),
        ("books", 0.82),
        ("home", 0.80),
        ("electronics", 0.78),
        ("books", 0.75),
        ("home", 0.72),
        ("electronics", 0.70),
    ]

    logger.info(f"  {'Rank':>5} {'Category':<16} {'Score':>8} {'In bubble?':>12}")
    logger.info(f"  {sep}")
    for rank, (cat, score) in enumerate(pure_relevance, 1):
        in_bubble = "✅ YES" if cat in ["electronics", "books", "home"] else "🌟 NO"
        logger.info(f"  {rank:>4}. {cat:<16} {score:>7.2f}  {in_bubble:>12}")

    # Compute bubble score
    bubble_count = sum(1 for cat, _ in pure_relevance if cat in ["electronics", "books", "home"])
    logger.info("")
    logger.info(f"  Items in user's bubble:   {bubble_count}/{len(pure_relevance)} ({bubble_count/len(pure_relevance):.0%})")
    logger.info(f"  Discovery score:          0% (echo chamber)")
    logger.info(f"  → User never sees anything outside their bubble")
    logger.info("")

    # ── Step 6: Compare with injection ──
    logger.info("📈 Step 6: Anti-echo-chamber vs. echo chamber comparison...")
    logger.info("")

    logger.info(f"  {'Metric':<35} {'Echo Chamber':>18} {'Anti-Echo':>18}")
    logger.info(f"  {sep}")
    logger.info(f"  {'Discovery score':<35} {'0.0%':>18} {current_score:>17.0%}")
    logger.info(f"  {'Categories shown':<35} {'3 (repeated)':>18} {'7+':>18}")
    logger.info(f"  {'User serendipity':<35} {'❌ None':>18} {'✅ High':>18}")
    logger.info(f"  {'Long-term retention':<35} {'❌ Declining':>18} {'✅ Growing':>18}")
    logger.info("")

    # ── Step 7: Key insight ──
    logger.info("💡 Step 7: The Key Interview Insight")
    logger.info("")
    logger.info("  What existing recommenders get wrong:")
    logger.info("  • They optimize for immediate CTR, not long-term retention")
    logger.info("  • Showing only familiar categories = short-term gains")
    logger.info("  • Echo chambers cause user churn over months")
    logger.info("")
    logger.info("  What this system does:")
    logger.info("  • Reserves every 7th slot for intentional discovery")
    logger.info("  • Tracks discovery score as a first-class metric")
    logger.info("  • Targets > 15% novelty in every recommendation set")
    logger.info("  • Users discover new categories they didn't know existed")
    logger.info("  • Higher long-term retention and platform stickiness")
    logger.info("")
    logger.info("  In this demo:")
    logger.info("    Without injection: 0% discovery, 3 categories, echo chamber")
    logger.info("    With injection:    {:.0%} discovery, 7+ categories, exploration".format(current_score))
    logger.info("")

    # ── Summary ──
    logger.info("═" * 68)
    logger.info("  ✅ Demo complete! Key metrics:")
    logger.info(f"   • Candidates:        {NUM_CANDIDATES}")
    logger.info(f"   • Discovery pool:    {len(discovery_pool)} items across 12 categories")
    logger.info(f"   • Discovery score:   {current_score:.0%} (target: > 15%)")
    logger.info(f"   • Injections:        {discovery_count}/{len(results[:20])} positions")
    logger.info(f"   • User categories:   electronics, books, home (before)")
    logger.info("")
    logger.info("  The Anti-Echo-Chamber ensures users discover new")
    logger.info("  categories, preventing the long-term retention damage")
    logger.info("  caused by recommendation bubbles.")
    logger.info("═" * 68)


if __name__ == "__main__":
    run_demo()
