"""Statistical voice profiling and writing sample ingestion for candidate voice consistency."""

import collections
import hashlib
import re
import statistics
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from pilot.critic.schemas import VoiceProfile
from pilot.db.models import WritingSample
from pilot.ingestion.reader import ResumeReader

# Non-negotiable clichés and buzzwords never used in professional grounded writing
DEFAULT_BANNED_PHRASES: list[str] = [
    "bandwidth",
    "boil the ocean",
    "circle back",
    "deep dive",
    "game changer",
    "hit the ground running",
    "low hanging fruit",
    "move the needle",
    "ninja",
    "paradigm shift",
    "rockstar",
    "synergy",
    "take it to the next level",
    "think outside the box",
    "wheelhouse",
]

FIRST_PERSON_PRONOUNS: set[str] = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
}

HEDGING_WORDS: set[str] = {
    "apparent",
    "apparently",
    "arguably",
    "conceivably",
    "fairly",
    "likely",
    "maybe",
    "partially",
    "perhaps",
    "possibly",
    "probably",
    "roughly",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "somewhat",
    "suggested",
    "suggesting",
    "suggests",
    "tend",
    "tended",
    "tends",
    "unlikely",
}

PASSIVE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+(?:\w+\s+)?"
    r"([a-z]+(?:ed|en|wn)|done|seen|made|written|given|taken|found|built|held|taught|run|paid)\b",
    re.IGNORECASE,
)


def _extract_text(sample: WritingSample | str) -> str:
    if isinstance(sample, WritingSample):
        return sample.raw_text
    return str(sample)


def build_voice_profile(
    samples: Sequence[WritingSample | str],
    custom_banned_phrases: Sequence[str] | None = None,
) -> VoiceProfile:
    """
    Compute deterministic statistical metrics across one or more writing samples.

    Guarantees:
    - Pure, deterministic calculation (zero randomness, zero LLM calls).
    - Reproducible output for the same input.
    - Robust against empty strings or minimal inputs.
    """
    if not samples:
        banned = sorted(set(DEFAULT_BANNED_PHRASES + list(custom_banned_phrases or [])))
        return VoiceProfile(banned_phrases=banned, sample_count=0, total_words=0)

    combined_text = "\n\n".join(
        _extract_text(s).strip() for s in samples if _extract_text(s).strip()
    )
    if not combined_text:
        banned = sorted(set(DEFAULT_BANNED_PHRASES + list(custom_banned_phrases or [])))
        return VoiceProfile(banned_phrases=banned, sample_count=len(samples), total_words=0)

    # 1. Words & Tokens
    all_words = re.findall(r"\b[A-Za-z0-9'-]+\b", combined_text)
    total_words = len(all_words)

    # 2. Sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+", combined_text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    sentence_lengths = [len(re.findall(r"\b[A-Za-z0-9'-]+\b", s)) for s in sentences]
    sentence_lengths = [length for length in sentence_lengths if length > 0]

    if sentence_lengths:
        mean_sentence_length = round(float(sum(sentence_lengths) / len(sentence_lengths)), 4)
        median_sentence_length = round(float(statistics.median(sentence_lengths)), 4)
        sentence_length_variance = (
            round(float(statistics.variance(sentence_lengths)), 4)
            if len(sentence_lengths) > 1
            else 0.0
        )
    else:
        mean_sentence_length = 0.0
        median_sentence_length = 0.0
        sentence_length_variance = 0.0

    # 3. Paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", combined_text) if p.strip()]
    para_lengths = [len(re.findall(r"\b[A-Za-z0-9'-]+\b", p)) for p in paragraphs]
    mean_paragraph_length = (
        round(float(sum(para_lengths) / len(para_lengths)), 4) if para_lengths else 0.0
    )

    # 4. Contractions
    contractions = re.findall(
        r"\b[A-Za-z]+'(?:d|s|re|ve|m|ll|t|clock)\b",
        combined_text,
        re.IGNORECASE,
    )
    contraction_rate = round(float(len(contractions) / total_words), 4) if total_words > 0 else 0.0

    # 5. First-person pronoun rate
    fp_count = sum(1 for w in all_words if w.lower() in FIRST_PERSON_PRONOUNS)
    first_person_pronoun_rate = round(float(fp_count / total_words), 4) if total_words > 0 else 0.0

    # 6. Passive voice rate
    passive_matches = PASSIVE_PATTERN.findall(combined_text)
    sentence_count = max(len(sentences), 1)
    passive_voice_rate = round(float(len(passive_matches) / sentence_count), 4)

    # 7. Hedging rate
    hedging_count = sum(1 for w in all_words if w.lower() in HEDGING_WORDS)
    hedging_rate = round(float(hedging_count / total_words), 4) if total_words > 0 else 0.0

    # 8. Punctuation frequencies (per 100 words)
    factor = 100.0 / total_words if total_words > 0 else 0.0
    exclamation_freq = round(float(combined_text.count("!") * factor), 4)
    em_dash_count = combined_text.count("—") + combined_text.count("--") + combined_text.count("–")
    em_dash_freq = round(float(em_dash_count * factor), 4)
    semicolon_freq = round(float(combined_text.count(";") * factor), 4)

    # 9. Type-token ratio
    unique_words = {w.lower() for w in all_words}
    type_token_ratio = round(float(len(unique_words) / total_words), 4) if total_words > 0 else 0.0

    # 10. Common sentence openers
    openers: list[str] = []
    for s in sentences:
        words = re.findall(r"\b[A-Za-z0-9'-]+\b", s)
        if words:
            openers.append(words[0].capitalize())

    opener_counter = collections.Counter(openers)
    common_sentence_openers = [word for word, _ in opener_counter.most_common(5)]

    # 11. Banned phrases (sorted deterministically)
    banned_list = sorted(set(DEFAULT_BANNED_PHRASES + list(custom_banned_phrases or [])))

    return VoiceProfile(
        mean_sentence_length=mean_sentence_length,
        median_sentence_length=median_sentence_length,
        sentence_length_variance=sentence_length_variance,
        mean_paragraph_length=mean_paragraph_length,
        contraction_rate=contraction_rate,
        first_person_pronoun_rate=first_person_pronoun_rate,
        passive_voice_rate=passive_voice_rate,
        hedging_rate=hedging_rate,
        exclamation_freq=exclamation_freq,
        em_dash_freq=em_dash_freq,
        semicolon_freq=semicolon_freq,
        type_token_ratio=type_token_ratio,
        common_sentence_openers=common_sentence_openers,
        banned_phrases=banned_list,
        sample_count=len(samples),
        total_words=total_words,
    )


def ingest_writing_sample(
    session: Session,
    user_id: UUID,
    path: str | Path,
) -> WritingSample:
    """
    Ingest a candidate's writing sample using ResumeReader and persist with statistical profile.

    Guarantees:
    - Verbatim span extraction via ResumeReader (PDF, TXT, MD).
    - Idempotency: Duplicate content hash for the same user returns existing record.
    - Computes and stores statistical VoiceProfile on writing_samples.voice_profile.
    """
    reader = ResumeReader()
    doc = reader.read(path)

    raw_bytes = doc.raw_text.encode("utf-8")
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Check for existing sample (idempotency)
    stmt = select(WritingSample).where(
        WritingSample.user_id == user_id,
        WritingSample.content_hash == content_hash,
    )
    existing = session.execute(stmt).scalars().first()
    if existing:
        return existing

    words = re.findall(r"\b[A-Za-z0-9'-]+\b", doc.raw_text)
    word_count = len(words)
    spans = [s.model_dump(mode="json") for s in doc.spans]

    # Compute voice profile for this sample
    profile = build_voice_profile([doc.raw_text])

    sample = WritingSample(
        user_id=user_id,
        source_url=doc.source_url,
        content_type=doc.content_type,
        content_hash=content_hash,
        raw_text=doc.raw_text,
        spans=spans,
        word_count=word_count,
        voice_profile=profile.model_dump(mode="json"),
    )
    session.add(sample)
    session.commit()
    session.refresh(sample)
    return sample


def get_user_voice_profile(session: Session, user_id: UUID) -> VoiceProfile:
    """
    Resolve aggregate voice profile across all candidate writing samples.

    Falls back to neutral baseline if no writing samples exist.
    """
    stmt = (
        select(WritingSample)
        .where(WritingSample.user_id == user_id)
        .order_by(WritingSample.created_at.asc())
    )
    samples = session.execute(stmt).scalars().all()
    if not samples:
        return build_voice_profile([])
    return build_voice_profile([s.raw_text for s in samples])
