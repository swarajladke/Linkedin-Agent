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
from pilot.db.models import (
    EvidenceClaim,
    Goal,
    GoalStatus,
    Role,
    RoleAssessment,
    RoleStatus,
    Strategy,
    User,
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
