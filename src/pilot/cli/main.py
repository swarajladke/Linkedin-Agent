"""Command Line Interface (CLI) for Pilot autonomous career agent."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import typer
from alembic import command
from alembic.config import Config
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from pilot.config import get_settings
from pilot.db.models import EvidenceClaim, Goal, GoalStatus, Strategy, User
from pilot.db.session import get_db_session, get_engine
from pilot.extraction import (
    GroundedExtractor,
    OpenAIStructuredClient,
    github_to_spans,
    upsert_evidence_claims,
)
from pilot.goals import GoalCompiler, InfeasibleGoalError, persist_compiled_goal
from pilot.ingestion import (
    EncryptedPDFError,
    GitHubClient,
    GitHubClientError,
    GitHubRateLimitError,
    GitHubUserNotFoundError,
    ImageOnlyPDFError,
    ResumeReader,
    ResumeReaderError,
    UnsupportedFileFormatError,
)

app = typer.Typer(no_args_is_help=True, help="Pilot: Goal-directed autonomous career agent CLI.")
goal_app = typer.Typer(no_args_is_help=True, help="Manage candidate goals and strategy versions.")
app.add_typer(goal_app, name="goal")

console = Console()


class CLIUserError(Exception):
    """Raised when an active candidate cannot be resolved."""

    pass


def run_migrations() -> None:
    """Run Alembic migrations programmatically up to head."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    ini_path = repo_root / "alembic.ini"
    if not ini_path.is_file():
        ini_path = Path("alembic.ini").resolve()

    cfg = Config(str(ini_path))
    migrations_dir = repo_root / "migrations"
    if migrations_dir.is_dir():
        cfg.set_main_option("script_location", str(migrations_dir))

    command.upgrade(cfg, "head")


def resolve_active_user(session: Session) -> User:
    """Resolve the active candidate user, returning the most recently registered candidate.

    Raises CLIUserError if no candidate has been initialized via 'pilot init'.
    """
    user = session.scalars(select(User).order_by(User.created_at.desc())).first()
    if user is None:
        raise CLIUserError("No initialized candidate found. Please run 'pilot init' first.")
    return user


@app.command()
def init(
    email: Annotated[str, typer.Option("--email", "-e", help="Candidate email address")],
    name: Annotated[str, typer.Option("--name", "-n", help="Candidate full name")],
    github: Annotated[
        str | None, typer.Option("--github", "-g", help="Candidate GitHub username")
    ] = None,
) -> None:
    """Initialize database connection, apply database migrations, and register/update candidate user."""
    # 1. Verify DB connectivity
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        console.print(
            f"[red]Database connection failed: {exc}\n"
            f"Is the PostgreSQL container running? Try running: docker compose up -d[/red]"
        )
        raise typer.Exit(1) from None

    # 2. Run Alembic migrations programmatically
    try:
        run_migrations()
    except Exception as exc:
        console.print(f"[red]Database migration failed: {exc}[/red]")
        raise typer.Exit(1) from None

    # 3. Upsert user by unique email
    with get_db_session() as session:
        user = session.scalar(select(User).where(User.email == email))
        if user is None:
            user = User(
                email=email,
                full_name=name,
                github_username=github,
            )
            session.add(user)
            session.flush()
            action = "Created"
        else:
            user.full_name = name
            if github is not None:
                user.github_username = github
            session.flush()
            action = "Updated"
        user_id = user.id

    console.print(
        f"[green]✓ User {action.lower()} successfully.[/green] User ID: [bold]{user_id}[/bold]"
    )


@app.command()
def ingest(
    resume: Annotated[
        Path, typer.Option("--resume", "-r", help="Path to resume file (PDF, TXT, MD)")
    ],
    github: Annotated[
        str | None, typer.Option("--github", "-g", help="Optional GitHub username override")
    ] = None,
    force_refresh: Annotated[
        bool, typer.Option("--force-refresh", help="Bypass local ingestion cache")
    ] = False,
) -> None:
    """Ingest candidate resume and GitHub portfolio, extract grounded evidence claims, and persist to database."""
    settings = get_settings()
    if not settings.openai_api_key or not settings.openai_api_key.get_secret_value().strip():
        console.print(
            "[red]OPENAI_API_KEY is not configured. Set OPENAI_API_KEY in .env before running ingest.[/red]"
        )
        raise typer.Exit(1)

    if not resume.exists():
        console.print(f"[red]Resume file not found at: {resume}[/red]")
        raise typer.Exit(1)

    # Resolve active candidate
    try:
        with get_db_session() as session:
            user = resolve_active_user(session)
            user_id: UUID = user.id
            target_github = github or user.github_username
    except CLIUserError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(1) from None

    # Read resume spans
    reader = ResumeReader()
    try:
        resume_doc = reader.read(resume)
    except (
        ImageOnlyPDFError,
        EncryptedPDFError,
        UnsupportedFileFormatError,
        ResumeReaderError,
    ) as exc:
        console.print(f"[red]Failed to read resume: {exc}[/red]")
        raise typer.Exit(1) from None

    # Fetch GitHub spans if username is resolved
    github_spans = []
    if target_github:
        try:
            token = settings.github_token.get_secret_value() if settings.github_token else None
            client = GitHubClient(token=token)
            user_data = client.fetch_user_data(target_github, force_refresh=force_refresh)
            github_spans = github_to_spans(user_data)
        except (GitHubUserNotFoundError, GitHubRateLimitError, GitHubClientError) as exc:
            console.print(f"[red]Failed to fetch GitHub data for '{target_github}': {exc}[/red]")
            raise typer.Exit(1) from None

    # Extract claims via GroundedExtractor
    llm_client = OpenAIStructuredClient()
    extractor = GroundedExtractor(llm=llm_client)

    resume_result = extractor.extract(
        spans=resume_doc.spans,
        entity_type="user",
        entity_id=user_id,
        source="resume",
    )

    github_result = None
    if github_spans:
        github_result = extractor.extract(
            spans=github_spans,
            entity_type="user",
            entity_id=user_id,
            source="github",
        )

    all_claims = list(resume_result.claims)
    if github_result:
        all_claims.extend(github_result.claims)

    # Persist claims with hash deduplication
    with get_db_session() as session:
        existing_hashes = set(
            session.scalars(
                select(EvidenceClaim.content_hash).where(EvidenceClaim.entity_id == user_id)
            ).all()
        )
        new_claims_count = sum(1 for c in all_claims if c.content_hash not in existing_hashes)
        upsert_evidence_claims(session, all_claims)

    # Render Extraction Summary Table
    summary_table = Table(title="Extraction & Grounding Summary", show_lines=True)
    summary_table.add_column("Source", style="cyan", no_wrap=True)
    summary_table.add_column("Proposed", justify="right")
    summary_table.add_column("Accepted", justify="right", style="green")
    summary_table.add_column("Dropped", justify="right", style="red")
    summary_table.add_column("Pass Rate", justify="right", style="bold")

    summary_table.add_row(
        "resume",
        str(resume_result.proposed_count),
        str(len(resume_result.claims)),
        str(len(resume_result.dropped)),
        f"{resume_result.grounding_pass_rate * 100:.1f}%",
    )
    if github_result:
        summary_table.add_row(
            "github",
            str(github_result.proposed_count),
            str(len(github_result.claims)),
            str(len(github_result.dropped)),
            f"{github_result.grounding_pass_rate * 100:.1f}%",
        )

    console.print(summary_table)

    # Render Dropped Claims Audit Table
    all_dropped = [
        ("resume", claim, excerpt, reason) for claim, excerpt, reason in resume_result.dropped
    ]
    if github_result:
        all_dropped.extend(
            [("github", claim, excerpt, reason) for claim, excerpt, reason in github_result.dropped]
        )

    if all_dropped:
        dropped_table = Table(title="Ungrounded Claims Dropped (Audit Trail)", show_lines=True)
        dropped_table.add_column("Source", style="cyan", width=8)
        dropped_table.add_column("Proposed Claim", style="white", width=40)
        dropped_table.add_column("Quoted Excerpt", style="magenta", width=30)
        dropped_table.add_column("Rejection Reason", style="red", width=30)

        for src, claim, excerpt, reason in all_dropped[:15]:
            dropped_table.add_row(src, claim[:60], excerpt[:50], reason)

        console.print(dropped_table)
        if len(all_dropped) > 15:
            console.print(
                f"[dim]... and {len(all_dropped) - 15} additional ungrounded claims dropped[/dim]"
            )

    if new_claims_count == 0 and all_claims:
        console.print(
            "[yellow]Idempotent ingest: 0 new rows written (all claims up to date in database).[/yellow]"
        )
    else:
        console.print(
            f"[green]Successfully ingested and persisted {new_claims_count} new evidence claims.[/green]"
        )


@goal_app.command(name="set")
def goal_set(
    objective: Annotated[
        str,
        typer.Argument(help="Career objective text (e.g. 'Land an Applied AI role by Dec 1')"),
    ],
    deadline: Annotated[
        str,
        typer.Option("--deadline", "-d", help="Target completion deadline (YYYY-MM-DD)"),
    ],
    constraint: Annotated[
        list[str] | None,
        typer.Option(
            "--constraint",
            "-c",
            help="Repeatable constraint key=value pairs (e.g. --constraint max_applications_per_day=5)",
        ),
    ] = None,
) -> None:
    """Compile a career objective into structured target specs, numeric criteria, and back-solved milestones."""
    settings = get_settings()
    if not settings.openai_api_key or not settings.openai_api_key.get_secret_value().strip():
        console.print(
            "[red]OPENAI_API_KEY is not configured. Set OPENAI_API_KEY in .env before running goal set.[/red]"
        )
        raise typer.Exit(1)

    try:
        parsed_date = datetime.strptime(deadline, "%Y-%m-%d").date()
        deadline_dt = datetime(
            parsed_date.year, parsed_date.month, parsed_date.day, 23, 59, 59, tzinfo=UTC
        )
    except ValueError:
        console.print(f"[red]Invalid deadline format '{deadline}'. Expected YYYY-MM-DD.[/red]")
        raise typer.Exit(1) from None

    constraints_dict: dict[str, Any] = {}
    if constraint:
        for item in constraint:
            if "=" not in item:
                console.print(
                    f"[red]Invalid constraint '{item}'. Must be in format key=value.[/red]"
                )
                raise typer.Exit(1)
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                num = float(v)
                constraints_dict[k] = int(num) if num.is_integer() else num
            except ValueError:
                constraints_dict[k] = v

    try:
        with get_db_session() as session:
            user = resolve_active_user(session)
            user_id = user.id
    except CLIUserError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(1) from None

    llm_client = OpenAIStructuredClient()
    compiler = GoalCompiler(llm=llm_client)

    try:
        goal_create = compiler.compile(
            user_id=user_id,
            objective_text=objective,
            constraints=constraints_dict,
            deadline=deadline_dt,
        )
    except InfeasibleGoalError as exc:
        console.print(f"[red]Goal cannot be scheduled: {exc}[/red]")
        raise typer.Exit(1) from None

    with get_db_session() as session:
        db_goal, db_strategy = persist_compiled_goal(session, goal_create)

    # Render Compiled Goal Output
    console.print(
        Panel(
            f"[bold]Objective:[/bold] {db_goal.objective_text}\n"
            f"[bold]Deadline:[/bold] {db_goal.deadline.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"[bold]Strategy Version:[/bold] v{db_strategy.version}",
            title="[bold green]Goal Compiled & Persisted[/bold green]",
        )
    )

    spec_table = Table(title="Target Specification Breakdown", show_lines=True)
    spec_table.add_column("Bucket", style="cyan", width=18)
    spec_table.add_column("Requirements", style="white")

    spec_table.add_row(
        "Must-Have", "\n".join(f"• {req}" for req in goal_create.target_spec.must_have) or "None"
    )
    spec_table.add_row(
        "Nice-to-Have",
        "\n".join(f"• {req}" for req in goal_create.target_spec.nice_to_have) or "None",
    )
    spec_table.add_row(
        "Unstated-but-Real",
        "\n".join(f"• {req}" for req in goal_create.target_spec.unstated_but_real) or "None",
    )
    console.print(spec_table)

    criteria_table = Table(title="Numeric Success Criteria", show_lines=True)
    criteria_table.add_column("Criterion", style="magenta")
    criteria_table.add_column("Target Value", style="bold green", justify="right")
    for k, v in goal_create.success_criteria.items():
        criteria_table.add_row(k, str(v))
    console.print(criteria_table)

    sub_goal_table = Table(title="Back-Solved Sub-Goal Timeline", show_lines=True)
    sub_goal_table.add_column("ID", style="cyan", width=6)
    sub_goal_table.add_column("Title", style="white")
    sub_goal_table.add_column("Metric Target", style="green")
    sub_goal_table.add_column("Deadline", style="magenta")
    sub_goal_table.add_column("Days from Now", justify="right")

    now = datetime.now(UTC)
    for sg in goal_create.sub_goals:
        metric_str = ", ".join(f"{k}: {v}" for k, v in sg.metric_target.items())
        days_from_now = max(0.0, (sg.deadline - now).total_seconds() / 86400.0)
        sub_goal_table.add_row(
            sg.id,
            sg.title,
            metric_str,
            sg.deadline.strftime("%Y-%m-%d %H:%M UTC"),
            f"{days_from_now:.1f}d",
        )
    console.print(sub_goal_table)


@app.command()
def show(
    claims: Annotated[
        bool, typer.Option("--claims", help="Display evidence claims section only")
    ] = False,
    goal: Annotated[bool, typer.Option("--goal", help="Display active goal section only")] = False,
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Maximum evidence claims to display")
    ] = 20,
) -> None:
    """Display current active goal, back-solved milestones, and verified evidence claims."""
    show_both = not claims and not goal
    show_claims = claims or show_both
    show_goal = goal or show_both

    with get_db_session() as session:
        user = session.scalars(select(User).order_by(User.created_at.desc())).first()
        if user is None:
            console.print(
                "[yellow]Database is empty. Run 'pilot init' to register a candidate user.[/yellow]"
            )
            return

        if show_goal:
            active_goal = session.scalar(
                select(Goal).where(Goal.user_id == user.id, Goal.status == GoalStatus.ACTIVE)
            )
            if not active_goal:
                console.print(
                    '[yellow]No active goal found. Run: pilot goal set "<objective>" --deadline <YYYY-MM-DD>[/yellow]'
                )
            else:
                strategy = session.scalar(
                    select(Strategy)
                    .where(Strategy.goal_id == active_goal.id)
                    .order_by(Strategy.version.desc())
                )
                strat_ver = strategy.version if strategy else 1
                now = datetime.now(UTC)
                days_left = max(0.0, (active_goal.deadline - now).total_seconds() / 86400.0)

                console.print(
                    Panel(
                        f"[bold]Objective:[/bold] {active_goal.objective_text}\n"
                        f"[bold]Deadline:[/bold] {active_goal.deadline.strftime('%Y-%m-%d %H:%M UTC')} ({days_left:.1f} days remaining)\n"
                        f"[bold]Strategy:[/bold] version {strat_ver} (active)\n"
                        f"[bold]Success Criteria:[/bold] {active_goal.success_criteria}",
                        title="[bold cyan]Active Goal Overview[/bold cyan]",
                    )
                )

                sub_table = Table(title="Sub-Goal Milestones", show_lines=True)
                sub_table.add_column("ID", style="cyan", width=6)
                sub_table.add_column("Title", style="white")
                sub_table.add_column("Metric Target", style="green")
                sub_table.add_column("Deadline", style="magenta")
                sub_table.add_column("Status", style="yellow")

                for sg_dict in active_goal.sub_goals:
                    metric_str = ", ".join(
                        f"{k}: {v}" for k, v in sg_dict.get("metric_target", {}).items()
                    )
                    sub_table.add_row(
                        sg_dict.get("id", ""),
                        sg_dict.get("title", ""),
                        metric_str,
                        sg_dict.get("deadline", "")[:16],
                        sg_dict.get("status", "pending"),
                    )
                console.print(sub_table)

        if show_claims:
            claims_rows = session.scalars(
                select(EvidenceClaim)
                .where(EvidenceClaim.entity_id == user.id)
                .order_by(EvidenceClaim.confidence.desc())
            ).all()

            if not claims_rows:
                console.print(
                    "[yellow]No evidence claims found. Run: pilot ingest --resume <path> [--github <username>][/yellow]"
                )
            else:
                resume_count = sum(1 for c in claims_rows if c.source == "resume")
                github_count = sum(1 for c in claims_rows if c.source == "github")
                other_count = len(claims_rows) - resume_count - github_count
                latest_verified = max(c.verified_at for c in claims_rows)

                console.print(
                    Panel(
                        f"[bold]Total Claims:[/bold] {len(claims_rows)} | "
                        f"[bold]Resume:[/bold] {resume_count} | "
                        f"[bold]GitHub:[/bold] {github_count}"
                        + (f" | [bold]Other:[/bold] {other_count}" if other_count > 0 else "")
                        + f"\n[bold]Latest Verification:[/bold] {latest_verified.strftime('%Y-%m-%d %H:%M UTC')}",
                        title="[bold green]Verified Evidence Claims[/bold green]",
                    )
                )

                claims_table = Table(
                    title=f"Evidence Claims (Top {min(limit, len(claims_rows))})", show_lines=True
                )
                claims_table.add_column("Source", style="cyan", width=8)
                claims_table.add_column("Conf", justify="right", style="green", width=6)
                claims_table.add_column("Claim", style="white")
                claims_table.add_column("Locator", style="magenta")

                for claim_row in claims_rows[:limit]:
                    locator = (
                        claim_row.source_url.split("#")[-1]
                        if "#" in claim_row.source_url
                        else claim_row.source_url
                    )
                    claims_table.add_row(
                        claim_row.source,
                        f"{claim_row.confidence:.2f}",
                        claim_row.claim,
                        locator,
                    )
                console.print(claims_table)
