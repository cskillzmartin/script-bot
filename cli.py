"""Command-line interface for the content generation pipeline."""

import asyncio
import json
import sys
from typing import Optional

import structlog
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agents.orchestrator import OrchestratorAgent
from config import ensure_directories, settings
from models import UserInput

# Set up logging
logger = structlog.get_logger(__name__)
console = Console()

# Create the CLI app
app = typer.Typer(
    name="content-pipeline",
    help="Autonomous multi-agent content generation pipeline for YouTube scripts"
)


@app.callback()
def callback():
    """Content Generation Pipeline CLI."""
    # Ensure output directories exist
    ensure_directories()


@app.command()
def generate(
    subject: str = typer.Option(..., prompt="Enter the main topic/subject"),
    scope: str = typer.Option(..., prompt="Enter the specific scope or angle"),
    audience: str = typer.Option(
        ...,
        prompt="Enter target audience (e.g., 'healthcare professionals', 'marketing managers', 'high school students')"
    ),
    length: int = typer.Option(
        10,
        prompt="Enter target script length in minutes",
        min=1,
        max=30
    ),
    instructions: Optional[str] = typer.Option(
        None,
        help="Additional specific instructions for the script"
    ),
    verbose: bool = typer.Option(False, help="Enable verbose logging")
):
    """
    Generate a complete YouTube script using the autonomous pipeline.

    This command runs the full multi-agent workflow to research, validate,
    structure, and write a polished script.
    """
    # Set up logging level
    if verbose:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    console.print(f"[bold blue]🚀 Starting Content Generation Pipeline[/bold blue]")
    console.print(f"Subject: [cyan]{subject}[/cyan]")
    console.print(f"Scope: [cyan]{scope}[/cyan]")
    console.print(f"Audience: [cyan]{audience}[/cyan]")
    console.print(f"Target Length: [cyan]{length} minutes[/cyan]")
    console.print()

    # Create user input
    user_input = UserInput(
        subject=subject,
        scope=scope,
        target_audience=audience,
        target_length_minutes=length,
        additional_instructions=instructions
    )

    # Initialize orchestrator
    with console.status("[bold green]Initializing agents...") as status:
        orchestrator = OrchestratorAgent()

    # Run the workflow using asyncio
    try:
        # Run the async workflow in a synchronous context
        workflow_state = asyncio.run(_run_async_workflow(orchestrator, user_input, console))

        # Display results
        console.print()
        console.print("[bold green]✅ Workflow completed successfully![/bold green]")
        console.print()

        # Display script information
        if workflow_state.get("final_script"):
            script = workflow_state["final_script"]
            display_script_results(script, console)
        else:
            console.print("[red]❌ No script was generated[/red]")

        # Display workflow metadata
        metadata = workflow_state.get("metadata", {})
        display_workflow_summary(metadata, console)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Workflow interrupted by user[/yellow]")
        return
    except Exception as e:
        console.print(f"\n[red]❌ Workflow failed: {str(e)}[/red]")
        logger.error("Workflow execution failed", error=str(e))
        return


async def _run_async_workflow(orchestrator, user_input, console):
    """Run the async workflow with progress monitoring."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Create progress tasks
        research_task = progress.add_task("🔍 Research & Content Harvesting", total=None)
        validation_task = progress.add_task("✅ Fact-checking & Validation", total=None)
        structuring_task = progress.add_task("📋 Content Structuring", total=None)
        script_task = progress.add_task("✍️ Script Generation", total=None)

        # Monitor progress in background
        async def monitor_progress():
            while True:
                status = orchestrator.get_workflow_status()
                current_stage = status.get("current_stage", "")

                if current_stage == "research":
                    progress.update(research_task, completed=True)
                elif current_stage == "validation":
                    progress.update(validation_task, completed=True)
                elif current_stage == "structuring":
                    progress.update(structuring_task, completed=True)
                elif current_stage == "script_generation":
                    progress.update(script_task, completed=True)

                if status.get("status") == "completed":
                    break

                await asyncio.sleep(1)

        # Run workflow and monitoring concurrently
        workflow_task = asyncio.create_task(orchestrator.execute_workflow(user_input))
        monitor_task = asyncio.create_task(monitor_progress())

        # Wait for workflow completion
        workflow_state = await workflow_task

        # Cancel monitoring
        monitor_task.cancel()

        # Mark all tasks as complete
        progress.update(research_task, completed=True)
        progress.update(validation_task, completed=True)
        progress.update(structuring_task, completed=True)
        progress.update(script_task, completed=True)

        return workflow_state


@app.command()
def status():
    """Check the status of recent workflow runs."""
    from database.mongodb_client import MongoDBStore

    console.print("[bold blue]📊 Workflow Status[/bold blue]")
    console.print()

    try:
        # Get recent workflow runs
        with console.status("Fetching workflow data..."):
            db = MongoDBStore()
            recent_runs = []

            # Get last 10 workflow runs
            workflow_runs = db.database["workflow_runs"].find(
                {},
                {"_id": 0, "session_id": 1, "status": 1, "created_at": 1, "updated_at": 1}
            ).sort("created_at", -1).limit(10)

            for run in workflow_runs:
                recent_runs.append({
                    "session_id": run["session_id"],
                    "status": run["status"],
                    "created_at": run["created_at"],
                    "duration": (
                        run["updated_at"] - run["created_at"]
                    ).total_seconds() if run.get("updated_at") else None
                })

        if not recent_runs:
            console.print("[yellow]No workflow runs found[/yellow]")
            return

        # Display in table format
        table = Table(title="Recent Workflow Runs")
        table.add_column("Session ID", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Created", style="green")
        table.add_column("Duration (s)", style="yellow")

        for run in recent_runs:
            status_style = {
                "completed": "green",
                "in_progress": "yellow",
                "error": "red",
                "cancelled": "red"
            }.get(run["status"], "white")

            # Handle created_at as either datetime or string
            created_at = run["created_at"]
            if isinstance(created_at, str):
                created_at_str = created_at[:16].replace("T", " ")  # Format ISO string
            else:
                created_at_str = created_at.strftime("%Y-%m-%d %H:%M")
            
            table.add_row(
                run["session_id"][:8] + "...",
                f"[{status_style}]{run['status']}[/{status_style}]",
                created_at_str,
                f"{run['duration']:.1f}" if run["duration"] else "N/A"
            )

        console.print(table)

        # Show analytics
        analytics = asyncio.run(db.get_workflow_analytics(5))
        if analytics:
            console.print()
            console.print("[bold]📈 Analytics[/bold]")

            for stat in analytics:
                status_name = stat["_id"]
                count = stat["count"]
                avg_duration = stat.get("avg_duration_seconds", 0)
                console.print(f"  {status_name}: {count} runs (avg {avg_duration:.1f}s)")

    except Exception as e:
        console.print(f"[red]Failed to get status: {str(e)}[/red]")


@app.command()
def scripts(limit: int = typer.Option(5, help="Number of recent scripts to show")):
    """List recently generated scripts."""
    from database.mongodb_client import MongoDBStore

    console.print(f"[bold blue]📜 Recent Scripts (last {limit})[/bold blue]")
    console.print()

    try:
        with console.status("Fetching scripts..."):
            db = MongoDBStore()
            recent_scripts = asyncio.run(db.get_recent_scripts(limit))

        if not recent_scripts:
            console.print("[yellow]No scripts found[/yellow]")
            return

        # Display in table format
        table = Table(title="Generated Scripts")
        table.add_column("Title", style="cyan")
        table.add_column("Subject", style="white")
        table.add_column("Words", style="green")
        table.add_column("Created", style="yellow")

        for script in recent_scripts:
            # Handle created_at as either datetime or string
            created_at = script.get("created_at", "N/A")
            if isinstance(created_at, str):
                created_at_str = created_at[:16].replace("T", " ") if created_at != "N/A" else "N/A"
            else:
                created_at_str = created_at.strftime("%Y-%m-%d %H:%M") if created_at else "N/A"
            
            table.add_row(
                script["title"][:50] + ("..." if len(script["title"]) > 50 else ""),
                script["subject"],
                str(script["total_word_count"]),
                created_at_str
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Failed to list scripts: {str(e)}[/red]")


@app.command()
def cleanup(days: int = typer.Option(30, help="Delete runs older than N days")):
    """Clean up old workflow data."""
    from database.mongodb_client import MongoDBStore

    console.print(f"[bold yellow]🧹 Cleaning up data older than {days} days[/bold yellow]")

    try:
        with console.status("Cleaning up..."):
            db = MongoDBStore()
            deleted_count = asyncio.run(db.cleanup_old_runs(days))

        console.print(f"[green]✅ Cleanup completed! Deleted {deleted_count} records[/green]")

    except Exception as e:
        console.print(f"[red]❌ Cleanup failed: {str(e)}[/red]")


def display_script_results(script, console):
    """Display script generation results."""
    console.print("[bold green]📝 Generated Script[/bold green]")
    console.print(f"Title: [cyan]{script.title}[/cyan]")
    console.print(f"Subject: [white]{script.subject}[/white]")
    console.print(f"Target Audience: [white]{script.target_audience}[/white]")
    console.print(f"Word Count: [yellow]{script.total_word_count:,}[/yellow]")
    console.print(f"Estimated Time: [yellow]{script.estimated_read_time_seconds // 60} minutes[/yellow]")
    console.print()

    # Show first few sections
    console.print("[bold]📋 Script Sections:[/bold]")
    for i, section in enumerate(script.sections[:3], 1):
        console.print(f"{i}. [cyan]{section.section_title}[/cyan] ({section.word_count} words)")

    if len(script.sections) > 3:
        console.print(f"... and {len(script.sections) - 3} more sections")

    console.print()
    console.print("[dim]💡 Tip: Find the complete script in the outputs/scripts/ directory[/dim]")


def display_workflow_summary(metadata, console):
    """Display workflow execution summary."""
    console.print("[bold]📊 Workflow Summary[/bold]")
    console.print(f"Search Results: [cyan]{metadata.search_results_count}[/cyan]")
    console.print(f"Validated Content: [cyan]{metadata.validated_content_count}[/cyan]")
    console.print(f"Script Generated: [cyan]{'Yes' if metadata.script_generated else 'No'}[/cyan]")

    if metadata.error_message:
        console.print(f"Error: [red]{metadata.error_message}[/red]")


def main():
    """Main entry point for the CLI."""
    try:
        app()
    except Exception as e:
        console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
        logger.error("CLI execution failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
