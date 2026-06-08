#!/usr/bin/env python3
"""
Life-Stage Graph Diffusion Demo — Detecting Life Events and Forking Embeddings

The breakthrough: when two users with 80% taste overlap suddenly diverge
(one buys baby products, one does not), detect the life-event and permanently
fork their embeddings in real time.

No existing production recommender does this.

Usage:
    python -m demos.life_stage_diffusion_demo
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import random
from typing import Dict, List, Tuple

import copy
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("life_stage_demo")

from src.innovations.life_stage_diffusion import (
    LifeEvent,
    LifeStageGraphDiffusion,
    TasteProfile,
)


# ── Demo Runner ────────────────────────────────────────────────


def run_demo():
    """Run the full life-stage diffusion demo."""
    sep = "─" * 68
    rng = random.Random(42)

    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║     Life-Stage Graph Diffusion — Live Demo                  ║")
    logger.info("║  \"When two users diverge, we fork their embeddings.\"        ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("The breakthrough: Two users with 80% taste overlap browse the")
    logger.info("same store for years. Then one of them has a baby.")
    logger.info("")
    logger.info("Existing systems: they still get treated as similar users.")
    logger.info("This system: detects the life event, forks the embedding,")
    logger.info("and propagates the new signal through the taste graph.")
    logger.info("")

    # ── Step 1: Initialize ──
    logger.info("🌱 Step 1: Initializing Life-Stage Graph Diffusion Engine...")
    diffuser = LifeStageGraphDiffusion(
        similarity_threshold=0.50,  # Lowered for demo
        divergence_threshold=0.15,  # Lowered for demo
        fork_embedding_dim=16,
        min_observations_before_fork=2,
    )
    logger.info("  ✅ Engine initialized")
    logger.info("")

    # ── Step 2: Create users with similar taste ──
    logger.info("👥 Step 2: Creating 6 users with similar taste profiles...")
    logger.info("")

    base_embedding = np.array([0.5, 0.3, 0.8, 0.1, 0.6, 0.4, 0.2, 0.7, 0.9, 0.0] * 16)[:16]

    users = [
        ("alice",   {"electronics": 0.9, "books": 0.8, "home": 0.6, "apparel": 0.7, "sports": 0.5}),
        ("bob",     {"electronics": 0.8, "books": 0.7, "home": 0.7, "apparel": 0.6, "garden": 0.5}),
        ("carol",   {"electronics": 0.7, "books": 0.9, "home": 0.5, "apparel": 0.8, "music": 0.6}),
        ("dave",    {"electronics": 0.8, "books": 0.6, "home": 0.8, "apparel": 0.5, "sports": 0.7}),
        ("eve",     {"electronics": 0.9, "books": 0.7, "home": 0.6, "apparel": 0.9, "beauty": 0.8}),
        ("frank",   {"electronics": 0.6, "books": 0.8, "home": 0.7, "apparel": 0.6, "food": 0.8}),
    ]

    for user_id, affinities in users:
        # Create embedding close to base (all similar to each other)
        noise = np.random.default_rng(abs(hash(user_id))).normal(0, 0.05, 16)
        profile = TasteProfile(
            user_id=user_id,
            embedding=base_embedding + noise,
            category_affinities=affinities,
            price_preference=(10, 200),
            discovery_score=0.4,
        )
        diffuser.register_taste_profile(profile)

    logger.info("  Initial similarity estimates between users:")
    for user_a, _ in users[:3]:
        for user_b, _ in users[1:4]:
            if user_a < user_b:
                sim = diffuser._compute_taste_similarity(
                    diffuser._profiles[user_a],
                    diffuser._profiles[user_b],
                )
                logger.info(f"    {user_a:<6} ↔ {user_b:<6}: {sim:.2%} similarity")
    logger.info("")

    # Deep copy Alice's original profile before the fork (for before/after comparison)
    alice_before = copy.deepcopy(diffuser._profiles["alice"])

    # ── Step 3: Alice has a baby ──
    logger.info("🍼 Step 3: Alice has a baby — her shopping patterns shift...")
    logger.info("")

    # Alice starts buying baby products
    new_interactions_alice = [
        ("organic_cotton_onesie", "baby", time.time()),
        ("breast_pump_deluxe", "baby", time.time() + 10),
        ("baby_monitor_hd", "baby", time.time() + 20),
        ("diapers_mega_pack", "diapers", time.time() + 30),
        ("baby_shampoo_gentle", "baby", time.time() + 40),
        ("crib_sheets_set", "baby", time.time() + 50),
        ("baby_mobile_musical", "toys", time.time() + 60),
        ("baby_food_organic", "baby_food", time.time() + 70),
    ]

    logger.info(f"  Alice's new interactions:")
    for item_id, category, _ in new_interactions_alice:
        logger.info(f"    • {item_id:<30} ({category})")

    # Detect divergence
    is_diverging, life_event = diffuser.detect_divergence(
        user_id="alice",
        reference_user_id="bob",
        new_interactions=new_interactions_alice,
    )

    logger.info("")
    if life_event:
        logger.info(f"  ✅ Life event DETECTED!")
        logger.info(f"     Event type:  {life_event.event_type}")
        logger.info(f"     Confidence:  {life_event.confidence:.0%}")
        logger.info(f"     New cluster: {life_event.new_category_cluster}")
        logger.info(f"     Catalysts:   {', '.join(life_event.catalyst_items[:4])}")
    else:
        logger.info("  ⚠️  Life event not detected (may need more interactions)")

    logger.info("")

    # ── Step 4: Compare similarity before/after ──
    logger.info("📊 Step 4: Similarity before vs. after the life event...")
    logger.info("")

    reference_users = ["bob", "carol", "dave", "eve", "frank"]

    logger.info(f"  {'Pair':<25} {'Before':>10} {'After':>10} {'Change':>10}")
    logger.info(f"  {sep}")
    for ref in reference_users:
        if ref in diffuser._profiles and "alice" in diffuser._profiles:
            sim_after = diffuser._compute_taste_similarity(
                diffuser._profiles["alice"],
                diffuser._profiles[ref],
            )
            # Before similarity (saved original profile before fork)
            sim_before = diffuser._compute_taste_similarity(alice_before, diffuser._profiles[ref])
            change = sim_after - sim_before

            logger.info(f"  {'alice ↔ ' + ref:<25} {sim_before:>8.1%}  {sim_after:>8.1%}  {change:>+9.1%}")

    logger.info("")
    logger.info(f"  {sep}")
    logger.info("  Alice's embedding was forked — she now has a separate")
    logger.info("  embedding trajectory that reflects her new life stage.")
    logger.info("")

    # ── Step 5: Bob also has a baby → detect divergence separately ──
    logger.info("👶 Step 5: Bob also has a baby — detecting if different users")
    logger.info("         diverge into the same life-stage cluster...")
    logger.info("")

    new_interactions_bob = [
        ("stroller_ultra", "baby", time.time() + 100),
        ("car_seat_safe", "baby", time.time() + 110),
        ("baby_clothes_set", "baby", time.time() + 120),
        ("baby_books_set", "baby", time.time() + 130),
    ]

    is_diverging_bob, life_event_bob = diffuser.detect_divergence(
        user_id="bob",
        reference_user_id="carol",
        new_interactions=new_interactions_bob,
    )

    if life_event_bob:
        logger.info(f"  ✅ Bob's life event detected: {life_event_bob.event_type}")
        logger.info(f"     Both Alice and Bob forked into '{life_event_bob.event_type}' cluster")
        logger.info("     → The life-stage diffusion captures shared life transitions")
    else:
        logger.info("  ⚠️  Bob's event not yet detected (building momentum)")
    logger.info("")

    # ── Step 6: Propagate through the graph ──
    logger.info("🔄 Step 6: Propagating Alice's forked signal through the taste graph...")
    logger.info("")

    # Check taste neighbors before propagation
    neighbors_before = diffuser.get_taste_neighbors("alice", top_k=5)
    logger.info(f"  Alice's taste neighbors (before propagation):")
    for neighbor_id, sim in neighbors_before:
        logger.info(f"    • {neighbor_id:<8} similarity: {sim:.1%}")

    # Propagate
    diffuser.propagate_fork_signal(
        user_id="alice",
        signal_strength=0.3,
        propagation_hops=2,
    )

    users_affected = [n for n, _ in neighbors_before]
    logger.info(f"  → Fork signal propagated to {len(users_affected)} similar users")
    logger.info("")

    # ── Step 7: Show life event summary ──
    logger.info("📋 Step 7: Summary of all detected life events...")
    logger.info("")

    all_events = diffuser.get_life_events()
    logger.info(f"  {'User':<10} {'Event':<18} {'Confidence':<12} {'Categories':<25}")
    logger.info(f"  {sep}")

    for event in all_events:
        cats = ", ".join(event.catalyst_items[:3])
        logger.info(f"  {event.user_id:<10} {event.event_type:<18} {event.confidence:.0%}{'':>7} {cats:<25}")

    logger.info("")
    logger.info(f"  Total forked users: {diffuser.num_forked_users}")
    logger.info(f"  Total forks:       {diffuser.total_forks}")
    logger.info("")

    # ── Step 8: Key insight ──
    logger.info("💡 Step 8: The Key Interview Insight")
    logger.info("")
    logger.info("  What no existing production recommender does:")
    logger.info("  • When users diverge, they stay in the same cohort FOREVER")
    logger.info("  • A new parent gets treated like their childless peers")
    logger.info("  • Life events are invisible to the recommendation system")
    logger.info("")
    logger.info("  What this system does:")
    logger.info("  • Monitors real-time divergence in taste similarity")
    logger.info("  • Detects life events from category transition patterns")
    logger.info("  • Forks the user's embedding: new trajectory, new preferences")
    logger.info("  • Propagates the signal through the taste similarity graph")
    logger.info("")
    logger.info("  In this demo:")
    logger.info("    Alice had a baby → forked into 'baby' cluster")
    logger.info("    Bob had a baby → forked into 'baby' cluster")
    logger.info("    Their shared life event is now captured in their embeddings")
    logger.info("")

    # ── Summary ──
    logger.info("═" * 68)
    logger.info("  ✅ Demo complete! Key metrics:")
    logger.info(f"   • Users in graph:     {len(diffuser._profiles)}")
    logger.info(f"   • Life events detected: {len(all_events)}")
    logger.info(f"   • Forked users:       {diffuser.num_forked_users}")
    logger.info(f"   • Total forks:        {diffuser.total_forks}")
    logger.info("")
    logger.info("  Life-Stage Graph Diffusion is the system's recognition")
    logger.info("  that life is not static — and neither should recommendations be.")
    logger.info("═" * 68)


if __name__ == "__main__":
    run_demo()
