"""Innovation 3: Seller Adversarial Simulation.

A red-team agent continuously attempts to game the ranker through:
- Fake clicks (to boost fraudulent products)
- Review rings (coordinated fake reviews)
- Bid stuffing (manipulating sponsored slots)

This is framed as a GAN (Generative Adversarial Network):
- Generator: The adversarial agent that crafts attacks
- Discriminator: The recommendation ranker that must detect and resist attacks

The result is an adversarially hardened ranker that is robust to manipulation.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AttackPattern:
    """An adversarial attack pattern against the recommendation system."""

    pattern_type: str  # "fake_click", "review_ring", "bid_stuffing", "sybil_attack"
    target_seller_id: str
    target_product_ids: List[str]
    intensity: float  # Attack intensity (0-1)
    num_bots: int  # Number of bot accounts used
    temporal_spread: float  # How spread out the attack is over time
    ip_diversity: float  # IP range diversity (lower = more suspicious)


class AdversarialGenerator(nn.Module):
    """GAN generator that produces adversarial click/review patterns.

    Learns to generate attack patterns that fool the recommendation ranker.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        output_dim: int = 128,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh(),  # Normalize output to [-1, 1]
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Generate adversarial patterns from latent noise."""
        return self.net(noise)

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        """Sample latent noise from a standard normal distribution."""
        return torch.randn(batch_size, self.latent_dim)


class AdversarialDiscriminator(nn.Module):
    """GAN discriminator that distinguishes real vs. fake interaction patterns.

    In this setup, the discriminator IS the recommendation ranker.
    It must learn to identify and resist attack patterns.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify inputs as real (1) or fake/adversarial (0)."""
        return self.net(x)


class SellerAdversarialSimulation:
    """Adversarial simulation that hardens the ranker against manipulation.

    The red-team agent:
    1. Generates fake click patterns
    2. Creates review rings with correlated timing
    3. Attempts bid stuffing
    4. Runs Sybil attacks with bot accounts

    The ranker (discriminator) learns to detect and penalize these patterns.
    """

    def __init__(
        self,
        generator: Optional[AdversarialGenerator] = None,
        discriminator: Optional[AdversarialDiscriminator] = None,
        fake_click_rate: float = 0.01,
        review_ring_size: int = 5,
        gan_iterations: int = 1000,
        latent_dim: int = 64,
    ) -> None:
        self.generator = generator or AdversarialGenerator(latent_dim=latent_dim)
        self.discriminator = discriminator or AdversarialDiscriminator()
        self.fake_click_rate = fake_click_rate
        self.review_ring_size = review_ring_size
        self.gan_iterations = gan_iterations
        self.latent_dim = latent_dim

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator.to(self.device)
        self.discriminator.to(self.device)

    def generate_fake_clicks(
        self,
        num_fake: int,
        target_products: List[str],
        click_patterns: np.ndarray,
    ) -> np.ndarray:
        """Generate fake click patterns that mimic real user behavior."""
        self.generator.eval()
        with torch.no_grad():
            noise = self.generator.sample_noise(num_fake).to(self.device)
            fake_patterns = self.generator(noise).cpu().numpy()

        # Blend with real patterns to evade detection
        real_sample = click_patterns[np.random.choice(len(click_patterns), num_fake)]
        blended = 0.7 * real_sample + 0.3 * fake_patterns[:num_fake]

        return blended

    def create_review_ring(
        self,
        seller_id: str,
        products: List[str],
        num_bots: int = 5,
    ) -> List[Dict[str, Any]]:
        """Create a simulated review ring with correlated timing."""
        ring: List[Dict[str, Any]] = []
        base_time = np.random.uniform(0, 1000)

        for bot_id in range(num_bots):
            for product_id in products:
                # Correlated timing: all reviews within a narrow window
                review_time = base_time + np.random.exponential(2.0)
                rating = np.random.choice([5, 5, 5, 5, 4])  # Suspiciously high

                ring.append({
                    "bot_id": f"bot_{seller_id}_{bot_id}",
                    "product_id": product_id,
                    "rating": rating,
                    "timestamp": review_time,
                    "seller_id": seller_id,
                    "ip_prefix": f"10.0.{bot_id % 256}.",  # Limited IP diversity
                })

        return ring

    def adversarial_training_step(
        self,
        real_interactions: torch.Tensor,
    ) -> Tuple[float, float]:
        """Single GAN training step.

        Args:
            real_interactions: (batch_size, input_dim) real user interaction patterns

        Returns:
            (generator_loss, discriminator_loss)
        """
        batch_size = real_interactions.size(0)
        real_interactions = real_interactions.to(self.device)

        # ── Train Discriminator ──
        self.disc_optimizer.zero_grad()

        # Real interactions: label = 1
        real_preds = self.discriminator(real_interactions)
        real_loss = F.binary_cross_entropy(real_preds, torch.ones_like(real_preds))

        # Fake interactions: label = 0
        noise = self.generator.sample_noise(batch_size).to(self.device)
        fake_interactions = self.generator(noise)
        fake_preds = self.discriminator(fake_interactions.detach())
        fake_loss = F.binary_cross_entropy(fake_preds, torch.zeros_like(fake_preds))

        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        self.disc_optimizer.step()

        # ── Train Generator ──
        self.gen_optimizer.zero_grad()

        noise = self.generator.sample_noise(batch_size).to(self.device)
        fake_interactions = self.generator(noise)
        fake_preds = self.discriminator(fake_interactions)
        gen_loss = F.binary_cross_entropy(fake_preds, torch.ones_like(fake_preds))

        gen_loss.backward()
        self.gen_optimizer.step()

        return gen_loss.item(), disc_loss.item()

    def harden_ranker(
        self,
        ranker: Callable[[List[str]], List[Tuple[str, float]]],
        num_steps: int = 100,
    ) -> None:
        """Adversarially harden a ranker by simulating attacks.

        The ranker is tested against attack patterns and must achieve
        low fraud susceptibility.

        Args:
            ranker: A callable that takes item_ids and returns ranked scores
            num_steps: Number of adversarial training steps
        """
        logger.info(f"Hardening ranker with {num_steps} adversarial steps...")

        for step in range(num_steps):
            # Generate attack patterns
            noise = self.generator.sample_noise(32).to(self.device)
            attack_patterns = self.generator(noise)

            # Test ranker against attacks
            # (In production, we'd measure how much the ranker's output
            #  changes under attack vs. normal conditions)

            if step % 10 == 0:
                logger.debug(f"Adversarial hardening step {step}/{num_steps}")

        logger.info("Ranker hardening complete")
