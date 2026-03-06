"""CLI interface for vesuvius-mesh-qa."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from vesuvius_mesh_qa.io.discovery import discover_segments
from vesuvius_mesh_qa.io.loader import load_mesh
from vesuvius_mesh_qa.metrics.summary import compute_all_metrics, aggregate_score, letter_grade
from vesuvius_mesh_qa.report.json_report import build_json_report
from vesuvius_mesh_qa.report.csv_report import build_csv_row
from vesuvius_mesh_qa.report.visualize import export_visualization

console = Console()


@click.group()
def cli():
    """vesuvius-mesh-qa: Automated mesh quality scoring."""
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text")
@click.option("--weights", type=str, default=None, help="JSON string of metric weight overrides")
@click.option("--visualize", type=click.Path(), default=None, help="Export colored PLY highlighting problem regions")
@click.option("--volume", type=str, default=None,
              help="OME-Zarr volume URL for CT-informed sheet switching detection")
def score(path: str, fmt: str, weights: str | None, visualize: str | None, volume: str | None):
    """Score a single mesh file or segment directory."""
    path = Path(path)

    weight_overrides = json.loads(weights) if weights else None

    if path.is_dir():
        obj_files = list(path.glob("*.obj"))
        if not obj_files:
            console.print(f"[red]No .obj files found in {path}[/red]")
            sys.exit(1)
        mesh_path = obj_files[0]
    else:
        mesh_path = path

    console.print(f"Loading [bold]{mesh_path.name}[/bold]...")
    mesh = load_mesh(mesh_path)

    n_vertices = len(mesh.vertices)
    n_faces = len(mesh.triangles)
    console.print(f"  Vertices: {n_vertices:,}  Faces: {n_faces:,}")

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as progress:
        n_metrics = 7 if volume else 6
        task = progress.add_task("Computing metrics...", total=n_metrics)

        def _on_progress(name: str, idx: int, total: int):
            progress.update(task, completed=idx, description=f"  {name}")

        results = compute_all_metrics(
            mesh, weight_overrides=weight_overrides, on_progress=_on_progress,
            volume_url=volume,
        )
        progress.update(task, completed=n_metrics)
    agg = aggregate_score(results)
    grade = letter_grade(agg)

    if visualize:
        viz_path = Path(visualize)
        export_visualization(mesh, results, viz_path)
        console.print(f"  Visualization saved to [bold]{viz_path}[/bold]")

    if fmt == "json":
        report = build_json_report(mesh_path, mesh, results, agg, grade)
        click.echo(json.dumps(report, indent=2))
    else:
        _print_text_report(mesh_path, n_vertices, n_faces, results, agg, grade)


def _print_text_report(
    mesh_path: Path,
    n_vertices: int,
    n_faces: int,
    results: list,
    aggregate: float,
    grade: str,
):
    """Print a rich text report to the console."""
    table = Table(title=f"Mesh Quality Report: {mesh_path.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Weighted", justify="right")
    table.add_column("Details", style="dim")

    for r in results:
        detail_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in r.details.items()
                               if k != "problem_regions")
        score_color = "green" if r.score >= 0.75 else "yellow" if r.score >= 0.5 else "red"
        table.add_row(
            r.name,
            f"[{score_color}]{r.score:.3f}[/{score_color}]",
            f"{r.weight:.2f}",
            f"{r.weighted_score:.3f}",
            detail_str[:80],
        )

    console.print(table)

    grade_color = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "red"}.get(
        grade, "white"
    )
    console.print(
        f"\n  Aggregate Score: [bold]{aggregate:.3f}[/bold]  "
        f"Grade: [bold {grade_color}]{grade}[/bold {grade_color}]\n"
    )


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output CSV path")
@click.option("--weights", type=str, default=None, help="JSON string of metric weight overrides")
def batch(directory: str, output: str | None, weights: str | None):
    """Score all segment meshes in a directory tree, output ranked CSV."""
    weight_overrides = json.loads(weights) if weights else None

    segments = discover_segments(directory)
    if not segments:
        console.print(f"[red]No segments found in {directory}[/red]")
        sys.exit(1)

    console.print(f"Found [bold]{len(segments)}[/bold] segments")

    rows = []
    for seg in segments:
        console.print(f"  Scoring [cyan]{seg.segment_id}[/cyan]...")
        try:
            mesh = load_mesh(seg.obj_path)
            results = compute_all_metrics(mesh, weight_overrides=weight_overrides)
            agg = aggregate_score(results)
            grade = letter_grade(agg)
            row = build_csv_row(seg, mesh, results, agg, grade)
            rows.append(row)
        except Exception as e:
            console.print(f"    [red]Error: {e}[/red]")
            rows.append({
                "segment_id": seg.segment_id,
                "error": str(e),
                "aggregate_score": 0.0,
                "grade": "F",
            })
        finally:
            gc.collect()

    df = pd.DataFrame(rows).sort_values("aggregate_score", ascending=True)

    if output:
        df.to_csv(output, index=False)
        console.print(f"\nResults saved to [bold]{output}[/bold]")

    # Always print a summary table
    table = Table(title="Batch Results (worst → best)")
    table.add_column("Segment", style="cyan")
    table.add_column("Faces", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Grade", justify="center")
    table.add_column("Worst Metric", style="dim")

    metric_cols = ["triangle_quality", "topology", "normal_consistency",
                   "sheet_switching", "self_intersections", "noise"]
    for _, row in df.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            table.add_row(row["segment_id"], "—", "[red]ERR[/red]", "F", str(row.get("error", ""))[:40])
            continue
        n_faces = f"{int(row.get('n_faces', 0)):,}" if pd.notna(row.get("n_faces")) else "—"
        agg = row["aggregate_score"]
        grade = row["grade"]
        grade_color = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "red"}.get(grade, "white")
        score_color = "green" if agg >= 0.95 else "yellow" if agg >= 0.85 else "red"
        # Find worst metric
        worst_name, worst_val = "", 1.0
        for mc in metric_cols:
            val = row.get(mc)
            if val is not None and pd.notna(val) and val < worst_val:
                worst_val = val
                worst_name = mc
        worst_str = f"{worst_name}={worst_val:.3f}" if worst_name else ""
        table.add_row(
            row["segment_id"],
            n_faces,
            f"[{score_color}]{agg:.3f}[/{score_color}]",
            f"[{grade_color}]{grade}[/{grade_color}]",
            worst_str,
        )

    console.print(table)
    console.print(f"\n[bold]{len(rows)}[/bold] segments scored.")
