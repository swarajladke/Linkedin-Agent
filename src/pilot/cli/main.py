"""Command Line Interface (CLI) for Pilot autonomous career agent."""

import json
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
from pilot.db.models import (
    Action,
    CriticReview,
    Cycle,
    EvidenceClaim,
    Goal,
    GoalStatus,
    Role,
    RoleAssessment,
    RoleStatus,
    Strategy,
    User,
    WritingSample,
)
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
from pilot.intelligence import RoleAssessor, upsert_assessments
from pilot.sourcing import (
    AshbySource,
    GreenhouseSource,
    LeverSource,
    SourceQuery,
    upsert_roles,
)
from pilot.sourcing.config import load_boards_config

app = typer.Typer(no_args_is_help=True, help="Pilot: Goal-directed autonomous career agent CLI.")
goal_app = typer.Typer(no_args_is_help=True, help="Manage candidate goals and strategy versions.")
cycle_app = typer.Typer(no_args_is_help=True, help="Run and inspect decision cycles.")
voice_app = typer.Typer(
    no_args_is_help=True, help="Manage writing samples and statistical voice profile."
)
app.add_typer(goal_app, name="goal")
app.add_typer(cycle_app, name="cycle")
app.add_typer(voice_app, name="voice")

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
        ("resume", item.claim, item.source_excerpt, item.reason) for item in resume_result.dropped
    ]
    if github_result:
        all_dropped.extend(
            [
                ("github", item.claim, item.source_excerpt, item.reason)
                for item in github_result.dropped
            ]
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
def source(
    board: Annotated[
        list[str] | None,
        typer.Option("--board", "-b", help="Specific board token/slug(s) to source"),
    ] = None,
    all_boards: Annotated[
        bool,
        typer.Option("--all", "-a", help="Source all configured job boards"),
    ] = False,
    force_refresh: Annotated[
        bool,
        typer.Option("--force-refresh", help="Bypass local cache"),
    ] = False,
) -> None:
    """Fetch job postings from permitted job board APIs and idempotently upsert into database."""
    configured_boards = load_boards_config()
    if not configured_boards:
        console.print("[yellow]No boards configured in config/boards.yaml.[/yellow]")
        raise typer.Exit(1)

    targets: list[tuple[str, str, str]] = []  # (source, token, name)
    selected_tokens = set(board) if board else set()

    for src_name, entries in configured_boards.items():
        for entry in entries:
            tok = entry["token"]
            name = entry["name"]
            if (
                all_boards
                or not selected_tokens
                or tok in selected_tokens
                or tok.lower() in selected_tokens
                or name.lower() in selected_tokens
            ):
                targets.append((src_name, tok, name))

    if not targets and selected_tokens:
        console.print(
            f"[red]No configured boards matched selection: {', '.join(selected_tokens)}[/red]"
        )
        raise typer.Exit(1)

    table = Table(title="Job Sourcing Results", show_lines=True)
    table.add_column("Company / Board", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Fetched", justify="right")
    table.add_column("New", justify="right", style="green")
    table.add_column("Refreshed", justify="right", style="blue")
    table.add_column("Closed", justify="right", style="yellow")

    total_fetched = 0
    total_new = 0
    total_refreshed = 0
    total_closed = 0

    adapters = {
        "greenhouse": GreenhouseSource(),
        "ashby": AshbySource(),
        "lever": LeverSource(),
    }

    with get_db_session() as session:
        for src_name, tok, comp_name in targets:
            adapter = adapters.get(src_name)
            if not adapter:
                continue
            try:
                postings = adapter.fetch(
                    SourceQuery(token=tok, company_name=comp_name),
                    force_refresh=force_refresh,
                )
                res = upsert_roles(session, postings)
                session.commit()

                total_fetched += len(postings)
                total_new += res.new_count
                total_refreshed += res.refreshed_count
                total_closed += res.closed_count

                table.add_row(
                    f"{comp_name} ({tok})",
                    src_name,
                    str(len(postings)),
                    str(res.new_count),
                    str(res.refreshed_count),
                    str(res.closed_count),
                )
            except Exception as exc:
                console.print(f"[red]Error sourcing {src_name}:{tok} - {exc}[/red]")
                table.add_row(f"{comp_name} ({tok})", src_name, "ERROR", "-", "-", "-")

    console.print(table)
    console.print(
        f"[bold green]✓ Sourcing complete:[/bold green] "
        f"{total_fetched} fetched, {total_new} new, {total_refreshed} refreshed, {total_closed} closed."
    )


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

        # Top Opportunities Table
        if active_goal:
            top_assessments = session.scalars(
                select(RoleAssessment)
                .where(RoleAssessment.goal_id == active_goal.id)
                .order_by(RoleAssessment.fit_score.desc())
                .limit(limit)
            ).all()

            if top_assessments:
                opp_table = Table(
                    title=f"Top Assessed Opportunities (Goal: {active_goal.objective_text})",
                    show_lines=True,
                )
                opp_table.add_column("Role ID", style="dim", width=8)
                opp_table.add_column("Role Title", style="bold white")
                opp_table.add_column("Company", style="cyan")
                opp_table.add_column("Fit Score", justify="right", style="bold green")
                opp_table.add_column("Recommended Action", style="yellow")
                opp_table.add_column("Blocking Gaps", style="red")

                for a in top_assessments:
                    r = a.role
                    comp_name = r.company.name if r and r.company else "Unknown"
                    blocking_count = (
                        sum(1 for g in a.skill_gaps if isinstance(g, dict) and g.get("blocking"))
                        if a.skill_gaps
                        else 0
                    )
                    opp_table.add_row(
                        str(a.role_id)[:8],
                        r.title if r else "Unknown",
                        comp_name,
                        f"{a.fit_score:.2f}",
                        a.recommended_action,
                        f"{blocking_count} blocking" if blocking_count > 0 else "None",
                    )
                console.print(opp_table)


@app.command()
def assess(
    limit: Annotated[
        int, typer.Option("--limit", "-l", help="Maximum unassessed roles to evaluate")
    ] = 20,
    min_fit: Annotated[
        float, typer.Option("--min-fit", help="Minimum fit score threshold to display")
    ] = 0.0,
    version: Annotated[str, typer.Option("--version", "-v", help="Assessor policy version")] = "v1",
) -> None:
    """Assess unassessed open roles against active career goal using grounded candidate evidence claims."""
    with get_db_session() as session:
        try:
            user = resolve_active_user(session)
        except CLIUserError as err:
            console.print(f"[red]{err}[/red]")
            raise typer.Exit(1) from None

        active_goal = session.scalars(
            select(Goal).where(Goal.user_id == user.id, Goal.status == GoalStatus.ACTIVE)
        ).first()

        if not active_goal:
            console.print(
                '[yellow]No active goal found. Run: pilot goal set "<objective>" --deadline <YYYY-MM-DD>[/yellow]'
            )
            raise typer.Exit(1)

        # Load candidate evidence claims
        claims = list(
            session.scalars(select(EvidenceClaim).where(EvidenceClaim.entity_id == user.id)).all()
        )

        if not claims:
            console.print(
                "[yellow]No evidence claims found for user. Candidate fit scores will be grounded to 0. Run 'pilot ingest' first.[/yellow]"
            )

        # Find unassessed open roles for this (goal_id, assessor_version)
        assessed_role_ids = set(
            session.scalars(
                select(RoleAssessment.role_id).where(
                    RoleAssessment.goal_id == active_goal.id,
                    RoleAssessment.assessor_version == version,
                )
            ).all()
        )

        stmt = select(Role).where(Role.status == RoleStatus.OPEN).order_by(Role.created_at.desc())
        all_open_roles = session.scalars(stmt).all()
        unassessed_roles = [r for r in all_open_roles if r.id not in assessed_role_ids][:limit]

        if not unassessed_roles:
            console.print(
                f"[green]All open roles already assessed under assessor version '{version}' for active goal.[/green]"
            )
            return

        llm_client = OpenAIStructuredClient()
        assessor = RoleAssessor(llm=llm_client, assessor_version=version)

        assessment_creates = []
        for role in unassessed_roles:
            res_create = assessor.assess(role=role, goal=active_goal, claims=claims)
            if res_create.fit_score >= min_fit:
                assessment_creates.append(res_create)

        persisted = upsert_assessments(session, assessment_creates)
        session.commit()

        # Render assessment table
        table = Table(
            title=f"Role Assessments (Assessor: {version}, Goal: {active_goal.objective_text[:40]}...)",
            show_lines=True,
        )
        table.add_column("Role ID", style="dim", width=8)
        table.add_column("Title", style="bold white")
        table.add_column("Company", style="cyan")
        table.add_column("Fit Score", justify="right", style="bold green")
        table.add_column("Blocking Gaps", style="red")
        table.add_column("Recommended Action", style="yellow")

        for a in persisted:
            r = a.role
            comp_name = r.company.name if r and r.company else "Unknown"
            blocking_count = (
                sum(1 for g in a.skill_gaps if isinstance(g, dict) and g.get("blocking"))
                if a.skill_gaps
                else 0
            )
            table.add_row(
                str(a.role_id)[:8],
                r.title if r else "Unknown",
                comp_name,
                f"{a.fit_score:.2f}",
                f"{blocking_count} blocking" if blocking_count > 0 else "None",
                a.recommended_action,
            )

        console.print(table)
        console.print(
            f"[bold green]✓ Assessed {len(persisted)} roles successfully.[/bold green] Use 'pilot explain <role_id>' to inspect evidence citations."
        )


@app.command()
def explain(
    role_id: Annotated[str, typer.Argument(help="Role ID (or prefix) to explain")],
) -> None:
    """Print role assessment with grounded evidence claim citations and full provenance."""
    with get_db_session() as session:
        # Match by prefix or exact UUID
        role = None
        try:
            target_uuid = UUID(role_id)
            role = session.scalar(select(Role).where(Role.id == target_uuid))
        except ValueError:
            stmt = select(Role).where(text("id::text LIKE :prefix")).params(prefix=f"{role_id}%")
            role = session.scalars(stmt).first()

        if not role:
            console.print(f"[red]Role not found with ID/prefix: {role_id}[/red]")
            raise typer.Exit(1)

        # Get latest assessment for this role
        assessment = session.scalars(
            select(RoleAssessment)
            .where(RoleAssessment.role_id == role.id)
            .order_by(RoleAssessment.assessed_at.desc())
        ).first()

        if not assessment:
            console.print(
                f"[yellow]Role '{role.title}' has not been assessed yet. Run: pilot assess[/yellow]"
            )
            raise typer.Exit(1)

        comp_name = role.company.name if role.company else "Unknown"

        # Overview Header
        header_text = (
            f"[bold]Role:[/bold] {role.title}\n"
            f"[bold]Company:[/bold] {comp_name}\n"
            f"[bold]Location:[/bold] {role.location or 'Unspecified'} ({role.location_type})\n"
            f"[bold]Posting URL:[/bold] {role.posting_url or 'N/A'}\n"
            f"[bold]Fit Score:[/bold] [bold green]{assessment.fit_score:.2f}[/bold green] | "
            f"[bold]Action:[/bold] [bold yellow]{assessment.recommended_action}[/bold yellow] | "
            f"[bold]Assessor Version:[/bold] {assessment.assessor_version}"
        )
        console.print(
            Panel(header_text, title="[bold cyan]Role Intelligence Assessment[/bold cyan]")
        )

        # Fit Rationale
        console.print(
            Panel(
                assessment.fit_rationale,
                title="[bold green]Fit Rationale & Alignment[/bold green]",
            )
        )

        # Skill Gaps Table
        if assessment.skill_gaps:
            gaps_table = Table(title="Identified Skill & Experience Gaps", show_lines=True)
            gaps_table.add_column("Gap Description", style="white")
            gaps_table.add_column("Severity", style="cyan", width=10)
            gaps_table.add_column("Blocking?", style="bold red", width=12)

            for g in assessment.skill_gaps:
                gap_desc = g.get("gap", "") if isinstance(g, dict) else getattr(g, "gap", "")
                sev = g.get("severity", "") if isinstance(g, dict) else getattr(g, "severity", "")
                blocking = (
                    g.get("blocking", False)
                    if isinstance(g, dict)
                    else getattr(g, "blocking", False)
                )
                gaps_table.add_row(
                    gap_desc,
                    sev,
                    "[bold red]YES (Blocking)[/bold red]" if blocking else "[green]No[/green]",
                )
            console.print(gaps_table)

        # Supporting Grounded Evidence Claims
        claim_ids = [UUID(cid) for cid in assessment.supporting_claim_ids]
        if claim_ids:
            claims_rows = session.scalars(
                select(EvidenceClaim).where(EvidenceClaim.id.in_(claim_ids))
            ).all()

            claims_table = Table(
                title=f"Grounded Evidence Claims Supporting Fit ({len(claims_rows)} citations)",
                show_lines=True,
            )
            claims_table.add_column("Claim ID", style="dim", width=8)
            claims_table.add_column("Evidence Claim", style="white")
            claims_table.add_column("Source", style="cyan", width=8)
            claims_table.add_column("Source Locator / URL", style="magenta")
            claims_table.add_column("Verbatim Source Excerpt", style="green")

            for c in claims_rows:
                claims_table.add_row(
                    str(c.id)[:8],
                    c.claim,
                    c.source,
                    c.source_url,
                    f'"{c.source_excerpt}"',
                )
            console.print(claims_table)
        else:
            console.print(
                "[yellow]No grounded candidate claims support this role (score capped).[/yellow]"
            )


# ============================================================================
# Cycle Commands
# ============================================================================


def _resolve_active_goal(session: Session) -> Goal:
    """Resolve the active goal for the current user."""
    goal = session.scalars(
        select(Goal).where(Goal.status == GoalStatus.ACTIVE).order_by(Goal.created_at.desc())
    ).first()
    if not goal:
        raise CLIUserError("No active goal found. Please set a goal first with 'pilot goal set'.")
    return goal


@cycle_app.command("run")
def cycle_run(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", "-d", help="Evaluate pipeline without committing actions")
    ] = False,
) -> None:
    """Run a Pilot decision cycle: observe → diagnose → generate → score → select → execute."""
    from pilot.planner.cycle import run_cycle

    now = datetime.now(UTC)
    with get_db_session() as session:
        try:
            goal = _resolve_active_goal(session)
        except CLIUserError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        console.print(
            Panel(
                "[bold cyan]Running Decision Cycle[/bold cyan]"
                + (" [yellow](DRY RUN)[/yellow]" if dry_run else ""),
                expand=False,
            )
        )

        try:
            result = run_cycle(session, goal, now=now, dry_run=dry_run)
        except Exception as exc:
            console.print(f"[red]Cycle failed: {exc}[/red]")
            raise typer.Exit(1) from None

        diag = result.diagnosis
        console.print(
            f"\n[bold]Cycle #{result.cycle_number}[/bold] "
            f"| Diagnosis: [yellow]{diag.category.value}[/yellow]"
            + (f" (starved: {diag.starved_stage})" if diag.starved_stage else "")
        )

        t = Table(title="Funnel Observation")
        t.add_column("Stage")
        t.add_column("Count", justify="right")
        obs = result.observation
        t.add_row("Roles Sourced", str(obs.funnel_counts.roles_sourced))
        t.add_row("Roles Assessed", str(obs.funnel_counts.roles_assessed))
        t.add_row("Applications", str(obs.funnel_counts.applications))
        t.add_row("Responses", str(obs.funnel_counts.responses))
        t.add_row("Interviews", str(obs.funnel_counts.interviews))
        t.add_row("Offers", str(obs.funnel_counts.offers))
        console.print(t)

        console.print(
            f"\nProposed: [bold]{result.actions_proposed}[/bold] actions | "
            f"Selected: [bold]{result.actions_selected}[/bold] actions"
        )

        passed_count = sum(1 for er in result.execution_results if not er.dropped)
        regen_count = sum(1 for er in result.execution_results if er.critic_attempts > 1)
        drop_count = sum(1 for er in result.execution_results if er.dropped)

        if result.execution_results:
            console.print(
                f"Critic Gate: [bold green]{passed_count} passed[/bold green] | "
                f"[bold yellow]{regen_count} regenerated[/bold yellow] | "
                f"[bold red]{drop_count} dropped[/bold red]"
            )
            et = Table(title="Executed Actions")
            et.add_column("Role")
            et.add_column("Company")
            et.add_column("Status")
            et.add_column("Attempts", justify="right")
            et.add_column("Escalation ID")
            for er in result.execution_results:
                role_title = er.role_title or (
                    er.draft_package.role_title if er.draft_package else "—"
                )
                comp_name = er.company_name or (
                    er.draft_package.company_name if er.draft_package else "—"
                )
                status_str = "[green]PASSED[/green]" if not er.dropped else "[red]DROPPED[/red]"
                esc_id = str(er.escalation_id)[:8] if er.escalation_id else "—"
                et.add_row(role_title, comp_name, status_str, str(er.critic_attempts), esc_id)
            console.print(et)
        elif dry_run and result.selected_actions:
            et = Table(title="Would Execute (Dry Run)")
            et.add_column("Role")
            et.add_column("Company")
            et.add_column("Fit")
            et.add_column("Score")
            for sa in result.selected_actions:
                et.add_row(
                    sa.candidate.role_title,
                    sa.candidate.company_name,
                    f"{sa.candidate.fit_score:.2f}",
                    f"{sa.score:.3f}",
                )
            console.print(et)
        else:
            console.print("[yellow]No actions selected this cycle.[/yellow]")

        if not dry_run and result.cycle_id:
            console.print(
                f"\n[green]✓ Cycle #{result.cycle_number} committed. ID: {result.cycle_id}[/green]"
            )


@cycle_app.command("list")
def cycle_list(
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max cycles to show")] = 10,
) -> None:
    """List past decision cycles for the active goal."""
    with get_db_session() as session:
        try:
            goal = _resolve_active_goal(session)
        except CLIUserError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        cycles = session.scalars(
            select(Cycle)
            .where(Cycle.goal_id == goal.id)
            .order_by(Cycle.cycle_number.desc())
            .limit(limit)
        ).all()

        if not cycles:
            console.print("[yellow]No cycles found. Run 'pilot cycle run' to start.[/yellow]")
            return

        t = Table(title=f"Cycles for Goal: {goal.objective_text[:60]}")
        t.add_column("#", justify="right")
        t.add_column("Started")
        t.add_column("Diagnosis")
        t.add_column("Proposed", justify="right")
        t.add_column("Selected", justify="right")

        for c in cycles:
            diag_data = c.diagnosis or {}
            category = diag_data.get("category", "—")
            starved = diag_data.get("starved_stage")
            diag_str = f"{category}" + (f" ({starved})" if starved else "")
            t.add_row(
                str(c.cycle_number),
                c.started_at.strftime("%Y-%m-%d %H:%M") if c.started_at else "—",
                diag_str,
                str(c.actions_proposed),
                str(c.actions_selected),
            )
        console.print(t)


@cycle_app.command("show")
def cycle_show(
    cycle_number: Annotated[int, typer.Argument(help="Cycle number to inspect")],
) -> None:
    """Show detailed observation, diagnosis, and actions for a specific cycle."""
    with get_db_session() as session:
        try:
            goal = _resolve_active_goal(session)
        except CLIUserError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        cycle = session.scalars(
            select(Cycle).where(Cycle.goal_id == goal.id, Cycle.cycle_number == cycle_number)
        ).first()

        if not cycle:
            console.print(f"[red]Cycle #{cycle_number} not found.[/red]")
            raise typer.Exit(1) from None

        obs = cycle.observation or {}
        diag = cycle.diagnosis or {}
        funnel = obs.get("funnel_counts", {})

        console.print(
            Panel(
                f"[bold cyan]Cycle #{cycle.cycle_number}[/bold cyan]  Started: {cycle.started_at}",
                expand=False,
            )
        )

        ot = Table(title="Funnel Observation")
        ot.add_column("Stage")
        ot.add_column("Count", justify="right")
        for stage, val in funnel.items():
            ot.add_row(stage, str(val))
        console.print(ot)

        console.print(
            f"\nDiagnosis: [yellow]{diag.get('category', '—')}[/yellow]"
            + (f" (starved: {diag.get('starved_stage')})" if diag.get("starved_stage") else "")
        )
        for h in diag.get("hypotheses", []):
            console.print(
                f"  • [{h.get('cause')}] {h.get('explanation')} "
                f"(metric: {h.get('supporting_metric_name')} = {h.get('supporting_metric_value')})"
            )

        actions = session.scalars(select(Action).where(Action.cycle_id == cycle.id)).all()

        if actions:
            at = Table(title="Actions in Cycle")
            at.add_column("ID")
            at.add_column("Target")
            at.add_column("Pred. Prob.")
            at.add_column("Executed At")
            at.add_column("Status")
            for a in actions:
                status_str = "[green]passed[/green]"
                if a.actual_outcome:
                    try:
                        data = json.loads(a.actual_outcome)
                        if data.get("status") == "dropped":
                            status_str = "[red]dropped[/red]"
                    except Exception:
                        pass
                at.add_row(
                    str(a.id)[:8],
                    str(a.target_id)[:8],
                    f"{a.predicted_probability:.2f}" if a.predicted_probability else "—",
                    str(a.executed_at)[:19] if a.executed_at else "—",
                    status_str,
                )
            console.print(at)
        else:
            console.print("[yellow]No actions in this cycle.[/yellow]")


@app.command("outcome")
def outcome_record(
    action_id: Annotated[str, typer.Argument(help="Action UUID to record outcome for")],
    success: Annotated[bool, typer.Option("--success/--fail", help="Outcome result")] = True,
    details: Annotated[
        str | None, typer.Option("--details", help="Optional outcome details (JSON string)")
    ] = None,
    diagnosis_note: Annotated[
        str | None, typer.Option("--diagnosis", help="Retrospective diagnosis note")
    ] = None,
) -> None:
    """Record the actual outcome of an executed action and compute Brier score."""

    with get_db_session() as session:
        try:
            action_uuid = UUID(action_id)
        except ValueError:
            console.print(f"[red]Invalid UUID: {action_id}[/red]")
            raise typer.Exit(1) from None

        action = session.get(Action, action_uuid)
        if not action:
            console.print(f"[red]Action {action_id} not found.[/red]")
            raise typer.Exit(1) from None

        if action.predicted_probability is None:
            console.print(
                f"[red]Action {action_id} has no predicted_probability. Cannot compute Brier score.[/red]"
            )
            raise typer.Exit(1) from None

        p = float(action.predicted_probability)
        o = 1.0 if success else 0.0
        brier = (p - o) ** 2

        details_dict = None
        if details:
            try:
                details_dict = json.loads(details)
            except (json.JSONDecodeError, ValueError):
                details_dict = {"raw": details}

        from pilot.db.models import ActionOutcome

        outcome = ActionOutcome(
            action_id=action.id,
            binary_success=success,
            actual_outcome_details=details_dict,
            brier_score=brier,
            diagnosis=diagnosis_note,
            recorded_at=datetime.now(UTC),
        )
        session.add(outcome)
        session.commit()

        console.print(
            f"[green]✓ Outcome recorded for Action {action_id[:8]}...[/green]\n"
            f"  Success: {'✓' if success else '✗'}  |  "
            f"Predicted: {p:.2f}  |  Brier Score: {brier:.4f}"
        )


@app.command("replay")
def replay(
    cycle_number: Annotated[int, typer.Argument(help="Cycle number to replay counterfactually")],
    min_fit: Annotated[
        float | None, typer.Option("--min-fit", help="Minimum fit score filter")
    ] = None,
    max_actions: Annotated[
        int | None, typer.Option("--max-actions", help="Cap on actions to select")
    ] = None,
) -> None:
    """Replay a past cycle counterfactually under an alternate selection policy."""
    from pilot.planner.replay import replay_cycle

    with get_db_session() as session:
        try:
            goal = _resolve_active_goal(session)
        except CLIUserError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        cycle = session.scalars(
            select(Cycle).where(Cycle.goal_id == goal.id, Cycle.cycle_number == cycle_number)
        ).first()
        if not cycle:
            console.print(f"[red]Cycle #{cycle_number} not found.[/red]")
            raise typer.Exit(1) from None

        result = replay_cycle(session, cycle.id, min_fit=min_fit, max_actions=max_actions)

        console.print(Panel(f"[bold cyan]Replay: Cycle #{cycle_number}[/bold cyan]", expand=False))
        console.print(result.rationale_diff)

        rt = Table(title="Selection Diff")
        rt.add_column("Action ID")
        rt.add_column("Change")
        for aid in result.added_action_ids:
            rt.add_row(str(aid)[:8], "[green]+added[/green]")
        for aid in result.removed_action_ids:
            rt.add_row(str(aid)[:8], "[red]-removed[/red]")
        if not result.added_action_ids and not result.removed_action_ids:
            rt.add_row("—", "[dim]No change from original selection[/dim]")
        console.print(rt)


@app.command("review")
def review(
    action_id: Annotated[UUID, typer.Argument(help="UUID of action to inspect critic reviews for")],
) -> None:
    """Inspect critic review attempts, check results, and failure details for an action."""
    run_migrations()

    with get_db_session() as session:
        reviews = (
            session.execute(
                select(CriticReview)
                .where(CriticReview.action_id == action_id)
                .order_by(CriticReview.attempt.asc())
            )
            .scalars()
            .all()
        )

        if not reviews:
            console.print(f"[yellow]No critic reviews found for Action {action_id}.[/yellow]")
            return

        console.print(
            Panel(
                f"[bold cyan]Critic Verification Gate: Action {action_id}[/bold cyan]\n"
                f"Total Evaluation Attempts: {len(reviews)}",
                expand=False,
            )
        )

        for rev in reviews:
            verdict_color = (
                "green"
                if rev.verdict == "pass"
                else ("yellow" if rev.verdict == "regenerate" else "red")
            )
            title = (
                f"Attempt #{rev.attempt} — Verdict: "
                f"[{verdict_color}]{rev.verdict.upper()}[/{verdict_color}]"
            )
            rev_table = Table(title=title)
            rev_table.add_column("Check", style="bold")
            rev_table.add_column("Status")
            rev_table.add_column("Details")

            rev_table.add_row(
                "Grounding",
                "[green]✓ PASS[/green]" if rev.grounding_passed else "[red]✗ FAIL[/red]",
                "Verbatim claim provenance check",
            )
            rev_table.add_row(
                "Voice Consistency",
                "[green]✓ PASS[/green]" if rev.voice_passed else "[red]✗ FAIL[/red]",
                "Statistical voice & banned cliché check",
            )
            rev_table.add_row(
                "Factual Integrity",
                "[green]✓ PASS[/green]" if rev.factual_passed else "[red]✗ FAIL[/red]",
                "Entity, metric & duration inflation check",
            )
            console.print(rev_table)

            failures = rev.failures or []
            if failures:
                fail_table = Table(
                    title=f"Attempt #{rev.attempt} Failures ({len(failures)})",
                    style="red",
                )
                fail_table.add_column("Check", style="cyan")
                fail_table.add_column("Severity", style="bold red")
                fail_table.add_column("Offending Text", style="yellow")
                fail_table.add_column("Reason")

                for f in failures:
                    fail_table.add_row(
                        f.get("check", "—"),
                        f.get("severity", "—").upper(),
                        f.get("offending_text", "—"),
                        f.get("detail", "—"),
                    )
                console.print(fail_table)
            else:
                console.print("[dim green]  No defects detected on this attempt.[/dim green]\n")


@voice_app.command("add")
def voice_add(
    path: Annotated[Path, typer.Argument(help="Path to writing sample file (.txt, .md, .pdf)")],
) -> None:
    """Ingest a candidate writing sample and compute statistical voice metrics."""
    from pilot.critic.voice import ingest_writing_sample

    run_migrations()

    if not path.is_file():
        console.print(f"[red]Writing sample file not found at {path}[/red]")
        raise typer.Exit(1)

    with get_db_session() as session:
        try:
            user = resolve_active_user(session)
        except CLIUserError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        try:
            sample = ingest_writing_sample(session, user.id, path)
        except Exception as e:
            console.print(f"[red]Failed to ingest writing sample: {e}[/red]")
            raise typer.Exit(1) from None

        vp = sample.voice_profile or {}

        console.print(
            Panel(
                f"[bold green]✓ Ingested Writing Sample[/bold green]\n"
                f"  Sample ID:     {sample.id}\n"
                f"  Candidate:     {user.full_name}\n"
                f"  Content Type:  {sample.content_type}\n"
                f"  Words:         {sample.word_count}\n"
                f"  Mean Sent Len: {vp.get('mean_sentence_length', 0.0)} words\n"
                f"  Contractions:  {vp.get('contraction_rate', 0.0):.4f}\n"
                f"  Passive Voice: {vp.get('passive_voice_rate', 0.0):.4f}\n"
                f"  Pronoun Rate:  {vp.get('first_person_pronoun_rate', 0.0):.4f}",
                title="Voice Ingestion",
                expand=False,
            )
        )


@voice_app.command("show")
def voice_show() -> None:
    """Display the active candidate's writing samples and aggregate statistical voice profile."""
    from pilot.critic.voice import get_user_voice_profile

    run_migrations()

    with get_db_session() as session:
        try:
            user = resolve_active_user(session)
        except CLIUserError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from None

        samples = (
            session.execute(
                select(WritingSample)
                .where(WritingSample.user_id == user.id)
                .order_by(WritingSample.created_at.asc())
            )
            .scalars()
            .all()
        )

        if not samples:
            console.print(
                "[yellow]No writing samples found for candidate.[/yellow]\n"
                "Run 'pilot voice add <path>' with blog posts, cover letters, or essays."
            )
            return

        table = Table(title=f"Writing Samples for {user.full_name}")
        table.add_column("Sample ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Words", justify="right")
        table.add_column("Source URL")
        table.add_column("Created", style="dim")

        for s in samples:
            table.add_row(
                str(s.id)[:8],
                s.content_type,
                str(s.word_count),
                s.source_url or "—",
                s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else "—",
            )
        console.print(table)

        profile = get_user_voice_profile(session, user.id)

        stats_table = Table(title="Aggregate Statistical Voice Profile")
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", style="cyan")

        stats_table.add_row("Samples Analyzed", str(profile.sample_count))
        stats_table.add_row("Total Words", str(profile.total_words))
        stats_table.add_row("Mean Sentence Length", f"{profile.mean_sentence_length:.2f} words")
        stats_table.add_row("Median Sentence Length", f"{profile.median_sentence_length:.2f} words")
        stats_table.add_row("Sentence Length Variance", f"{profile.sentence_length_variance:.2f}")
        stats_table.add_row("Mean Paragraph Length", f"{profile.mean_paragraph_length:.2f} words")
        stats_table.add_row("Contraction Rate", f"{profile.contraction_rate:.4f} per word")
        stats_table.add_row(
            "1st-Person Pronoun Rate", f"{profile.first_person_pronoun_rate:.4f} per word"
        )
        stats_table.add_row("Passive Voice Rate", f"{profile.passive_voice_rate:.4f} per sent")
        stats_table.add_row("Hedging Rate", f"{profile.hedging_rate:.4f} per word")
        stats_table.add_row("Exclamation Frequency", f"{profile.exclamation_freq:.2f} / 100 words")
        stats_table.add_row("Em-Dash Frequency", f"{profile.em_dash_freq:.2f} / 100 words")
        stats_table.add_row("Semicolon Frequency", f"{profile.semicolon_freq:.2f} / 100 words")
        stats_table.add_row("Lexical Diversity (TTR)", f"{profile.type_token_ratio:.4f}")
        stats_table.add_row(
            "Common Openers",
            ", ".join(profile.common_sentence_openers) if profile.common_sentence_openers else "—",
        )
        stats_table.add_row(
            "Banned Clichés",
            ", ".join(profile.banned_phrases[:8])
            + ("..." if len(profile.banned_phrases) > 8 else ""),
        )

        console.print(stats_table)
