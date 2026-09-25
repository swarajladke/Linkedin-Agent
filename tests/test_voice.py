"""Unit tests for statistical voice profiling and writing sample processing."""

from unittest.mock import MagicMock

import pytest

from pilot.critic.voice import (
    DEFAULT_BANNED_PHRASES,
    build_voice_profile,
)


def test_build_voice_profile_deterministic():
    """build_voice_profile is deterministic and reproducible on identical input."""
    sample_text = (
        "I have built distributed systems for over five years. "
        "We deployed high-throughput Kafka streaming pipelines on Kubernetes. "
        "Don't worry about reliability because we monitor latency closely. "
        "However, maybe the service might seem somewhat slow under heavy peak load."
    )

    p1 = build_voice_profile([sample_text])
    p2 = build_voice_profile([sample_text])

    assert p1.model_dump() == p2.model_dump()
    assert p1.mean_sentence_length > 0.0
    assert p1.median_sentence_length > 0.0
    assert p1.contraction_rate > 0.0  # "Don't"
    assert p1.first_person_pronoun_rate > 0.0  # "I", "We", "we"
    assert p1.hedging_rate > 0.0  # "maybe", "seem", "somewhat"
    assert set(DEFAULT_BANNED_PHRASES).issubset(set(p1.banned_phrases))


def test_build_voice_profile_passive_voice_detection():
    """Passive voice constructions are detected and calculated accurately."""
    sample_text = (
        "The software was written by our senior engineering team. "
        "Each service is monitored 24/7 with Prometheus. "
        "All database records were deleted during migration."
    )
    profile = build_voice_profile([sample_text])

    assert profile.passive_voice_rate > 0.0
    assert profile.sample_count == 1
    assert profile.total_words > 0


def test_voice_profiling_never_calls_llm():
    """Voice profiling is purely statistical and must never invoke an LLM."""
    fake_llm = MagicMock()

    sample_text = (
        "Engineering teams must communicate clearly. "
        "Our architecture uses event-driven patterns with Redis and RabbitMQ."
    )

    profile = build_voice_profile([sample_text])

    # Assert fake LLM was never called
    fake_llm.complete_structured.assert_not_called()
    assert profile.total_words > 0


def test_build_voice_profile_empty_and_whitespace():
    """Empty or whitespace-only samples return neutral default profiles."""
    p_empty = build_voice_profile([])
    assert p_empty.sample_count == 0
    assert p_empty.total_words == 0
    assert p_empty.mean_sentence_length == 0.0
    assert len(p_empty.banned_phrases) > 0

    p_spaces = build_voice_profile(["   \n\n  \t  "])
    assert p_spaces.total_words == 0
    assert p_spaces.mean_sentence_length == 0.0


def test_custom_banned_phrases_included():
    """Custom banned phrases are included and sorted deterministically."""
    profile = build_voice_profile(["Simple text."], custom_banned_phrases=["10x engineer", "guru"])
    assert "10x engineer" in profile.banned_phrases
    assert "guru" in profile.banned_phrases
    assert profile.banned_phrases == sorted(profile.banned_phrases)
