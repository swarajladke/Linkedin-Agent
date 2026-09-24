"""Transforms structured GitHub candidate data into uniform groundable SourceSpan objects."""

import re

from pilot.ingestion.github import GitHubUserData
from pilot.ingestion.reader import SourceSpan


def github_to_spans(data: GitHubUserData) -> list[SourceSpan]:
    """Flatten GitHub profile, repositories, READMEs, and commit counts into atomic SourceSpans."""
    spans: list[SourceSpan] = []

    # 1. Profile Spans
    profile = data.profile
    p_url = profile.html_url

    if profile.name and profile.name.strip():
        spans.append(
            SourceSpan(
                text=profile.name.strip(),
                locator="profile;field=name",
                source_url=p_url,
            )
        )

    if profile.bio and profile.bio.strip():
        spans.append(
            SourceSpan(
                text=profile.bio.strip(),
                locator="profile;field=bio",
                source_url=p_url,
            )
        )

    if profile.company and profile.company.strip():
        spans.append(
            SourceSpan(
                text=f"Company: {profile.company.strip()}",
                locator="profile;field=company",
                source_url=p_url,
            )
        )

    if profile.location and profile.location.strip():
        spans.append(
            SourceSpan(
                text=f"Location: {profile.location.strip()}",
                locator="profile;field=location",
                source_url=p_url,
            )
        )

    # 2. Repository Spans
    for repo in data.repos:
        r_url = repo.html_url

        # Description
        if repo.description and repo.description.strip():
            spans.append(
                SourceSpan(
                    text=repo.description.strip(),
                    locator=f"repo={repo.name};field=description",
                    source_url=r_url,
                )
            )

        # Languages
        if repo.languages:
            lang_items = [f"{lang} ({bytes_} bytes)" for lang, bytes_ in repo.languages.items()]
            lang_text = f"Languages used in {repo.name}: {', '.join(lang_items)}"
            spans.append(
                SourceSpan(
                    text=lang_text,
                    locator=f"repo={repo.name};field=languages",
                    source_url=r_url,
                )
            )

        # Topics
        if repo.topics:
            topics_text = f"Topics for {repo.name}: {', '.join(repo.topics)}"
            spans.append(
                SourceSpan(
                    text=topics_text,
                    locator=f"repo={repo.name};field=topics",
                    source_url=r_url,
                )
            )

        # User Commits
        if repo.user_commit_count is not None:
            commit_text = (
                f"User {data.username} contributed {repo.user_commit_count} commits "
                f"to repository {repo.name}."
            )
            spans.append(
                SourceSpan(
                    text=commit_text,
                    locator=f"repo={repo.name};field=commits",
                    source_url=r_url,
                )
            )

        # README Chunks (chunked on blank lines, capped ~4000 characters)
        if repo.readme and repo.readme.strip():
            readme_url = repo.readme_url or f"{r_url}#readme"
            readme_spans = _chunk_readme(
                repo_name=repo.name,
                readme_text=repo.readme,
                readme_url=readme_url,
                max_chunk_chars=4000,
            )
            spans.extend(readme_spans)

    return spans


def _chunk_readme(
    repo_name: str,
    readme_text: str,
    readme_url: str,
    max_chunk_chars: int = 4000,
) -> list[SourceSpan]:
    """Split README into paragraph chunks capped at max_chunk_chars with character offsets."""
    chunks: list[SourceSpan] = []
    paragraphs = re.split(r"(\n\s*\n)", readme_text)

    current_chunk_parts: list[str] = []
    current_chunk_len = 0
    chunk_start_char = 0
    running_offset = 0
    chunk_idx = 1

    for part in paragraphs:
        part_len = len(part)
        if not part:
            continue

        if current_chunk_len + part_len > max_chunk_chars and current_chunk_parts:
            # Emit current chunk
            chunk_text = "".join(current_chunk_parts)
            stripped_text = chunk_text.strip()
            if stripped_text:
                rel_start = chunk_text.find(stripped_text)
                start_offset = chunk_start_char + rel_start
                end_offset = start_offset + len(stripped_text)
                locator = f"repo={repo_name};file=README;chunk={chunk_idx};chars={start_offset}-{end_offset}"
                chunks.append(
                    SourceSpan(text=stripped_text, locator=locator, source_url=readme_url)
                )
                chunk_idx += 1

            current_chunk_parts = []
            current_chunk_len = 0
            chunk_start_char = running_offset

        current_chunk_parts.append(part)
        current_chunk_len += part_len
        running_offset += part_len

    if current_chunk_parts:
        chunk_text = "".join(current_chunk_parts)
        stripped_text = chunk_text.strip()
        if stripped_text:
            rel_start = chunk_text.find(stripped_text)
            start_offset = chunk_start_char + rel_start
            end_offset = start_offset + len(stripped_text)
            locator = (
                f"repo={repo_name};file=README;chunk={chunk_idx};chars={start_offset}-{end_offset}"
            )
            chunks.append(SourceSpan(text=stripped_text, locator=locator, source_url=readme_url))

    return chunks
