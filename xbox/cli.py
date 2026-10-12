"""
X-Box Pipeline CLI — orchestrates the full analysis pipeline.

Usage:
    python -m xbox.cli analyze --archive /path/to/tweets.xlsx --senator "Mike Lee"
    python -m xbox.cli analyze --hf-dataset --all-senators
    python -m xbox.cli fetch-handles
    python -m xbox.cli list-models
"""
import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import (
    CLASSIFIER_MODELS,
    EMBEDDING_MODEL,
    OUTPUT_DIR,
    TOXICITY_MODEL,
)

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose):
    """X-Box Pipeline — Classifier-based tweet analysis for political accounts."""
    setup_logging(verbose)


@cli.command()
def fetch_handles():
    """Fetch current US senator Twitter/X handles."""
    from .data import fetch_senator_handles

    df = fetch_senator_handles(cache=True)

    table = Table(title=f"US Senators with Twitter Handles ({len(df)} found)")
    table.add_column("Name", style="cyan")
    table.add_column("Party", style="green")
    table.add_column("State")
    table.add_column("Handle", style="yellow")

    for _, row in df.iterrows():
        table.add_row(
            f"{row['first_name']} {row['last_name']}",
            row["party"],
            row["state"],
            f"@{row['twitter_handle']}",
        )

    console.print(table)


@cli.command()
def list_models():
    """List all models used in the pipeline."""
    table = Table(title="Pipeline Models")
    table.add_column("Component", style="cyan")
    table.add_column("Model ID", style="yellow")
    table.add_column("~Params", style="green")

    table.add_row("Embeddings", EMBEDDING_MODEL, "600M")
    for name, model_id in CLASSIFIER_MODELS.items():
        table.add_row(f"Classifier ({name})", model_id, "~125M")
    table.add_row("Toxicity", TOXICITY_MODEL, "~355M")

    console.print(table)


@cli.command()
@click.option("--archive", "-a", type=click.Path(exists=True), help="Path to tweet archive (xlsx/csv/json)")
@click.option("--senator", "-s", type=str, default="", help="Senator name for labeling")
@click.option("--handle", "-h", type=str, default="", help="Twitter handle for labeling")
@click.option("--hf-dataset", is_flag=True, help="Also load HuggingFace senator-tweets dataset")
@click.option("--output-dir", "-o", type=click.Path(), default=None, help="Output directory")
@click.option("--skip-embeddings", is_flag=True, help="Skip embedding generation (faster)")
@click.option("--batch-size", "-b", type=int, default=32, help="Classification batch size")
@click.option("--party", type=str, default="", help="Party affiliation")
@click.option("--state", type=str, default="", help="State")
def analyze(archive, senator, handle, hf_dataset, output_dir, skip_embeddings, batch_size, party, state):
    """
    Run the full analysis pipeline on a tweet dataset.

    Example:
        python -m xbox.cli analyze -a /mnt/c/x_box/BasedMikeLee_full_archive.xlsx -s "Mike Lee" -h "BasedMikeLee"
    """
    from .behavioral import BehavioralAnalyzer
    from .classifiers import MultiHeadClassifier
    from .data import load_local_archive, load_hf_senator_tweets
    from .embeddings import TweetEmbedder
    from .fusion import ScoreFusion
    from .report import generate_json_report, generate_markdown_report

    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load data ─────────────────────────────
    console.print("\n[bold cyan]Step 1: Loading data...[/]")

    frames = []
    if archive:
        df = load_local_archive(archive, senator_name=senator)
        frames.append(df)
    if hf_dataset:
        hf_df = load_hf_senator_tweets()
        frames.append(hf_df)

    if not frames:
        console.print("[red]No data source specified. Use --archive or --hf-dataset[/]")
        sys.exit(1)

    import pandas as pd
    data = pd.concat(frames, ignore_index=True)
    console.print(f"  Loaded [green]{len(data):,}[/] tweets")

    # ── Step 2: Behavioral analysis ───────────────────
    console.print("\n[bold cyan]Step 2: Behavioral analysis...[/]")
    analyzer = BehavioralAnalyzer()
    behavioral = analyzer.analyze(data, senator_name=senator, twitter_handle=handle)
    console.print(f"  Compulsion score: [yellow]{behavioral.compulsion_score}/100[/]")

    # ── Step 3: Text classification ───────────────────
    console.print("\n[bold cyan]Step 3: Multi-head classification...[/]")
    classifier = MultiHeadClassifier()

    if "text" not in data.columns:
        console.print("[red]No 'text' column found in data[/]")
        sys.exit(1)

    classified = classifier.classify_tweets(data, batch_size=batch_size)
    console.print(f"  Classified [green]{len(classified):,}[/] tweets across {len(CLASSIFIER_MODELS) + 1} heads")

    # ── Step 4: Embeddings (optional) ─────────────────
    if not skip_embeddings:
        console.print("\n[bold cyan]Step 4: Generating embeddings...[/]")
        embedder = TweetEmbedder()
        emb_path = str(out_dir / f"{handle or senator or 'tweets'}_embeddings.npy")
        embeddings = embedder.embed_dataframe(classified, save_path=emb_path)
        console.print(f"  Generated embeddings: shape {embeddings.shape}")
    else:
        console.print("\n[dim]Step 4: Skipping embeddings (--skip-embeddings)[/]")

    # ── Step 5: Score fusion ──────────────────────────
    console.print("\n[bold cyan]Step 5: Score fusion...[/]")
    fusion = ScoreFusion()
    classified = fusion.compute_tweet_virulence(classified)
    profile = fusion.aggregate_senator_profile(
        classified, behavioral,
        senator_name=senator,
        twitter_handle=handle,
        party=party,
        state=state,
    )
    console.print(f"  Virulence score: [yellow]{profile.virulence_score}/100[/]")
    console.print(f"  Overall risk:    [bold red]{profile.overall_risk_score}/100[/]")

    # ── Step 6: Report generation ─────────────────────
    console.print("\n[bold cyan]Step 6: Generating reports...[/]")
    slug = handle or senator.replace(" ", "_") or "analysis"

    json_path = str(out_dir / f"{slug}_report.json")
    md_path = str(out_dir / f"{slug}_report.md")

    generate_json_report(profile, output_path=json_path)
    generate_markdown_report(profile, output_path=md_path)

    # Save classified tweets — coerce problematic mixed-type columns to string
    for col in classified.columns:
        if classified[col].dtype == object:
            classified[col] = classified[col].astype(str)
    classified_path = str(out_dir / f"{slug}_classified_tweets.parquet")
    classified.to_parquet(classified_path, index=False)

    console.print(f"\n[bold green]Done![/]")
    console.print(f"  JSON report:  {json_path}")
    console.print(f"  MD report:    {md_path}")
    console.print(f"  Tweets data:  {classified_path}")


@cli.command()
@click.option("--output-dir", "-o", type=click.Path(), default=None)
def batch_analyze(output_dir):
    """
    Batch analyze all senators with available data.
    Fetches handles, loads available datasets, runs pipeline for each.
    """
    from .data import fetch_senator_handles, load_hf_senator_tweets
    from .behavioral import BehavioralAnalyzer
    from .classifiers import MultiHeadClassifier
    from .fusion import ScoreFusion
    from .report import generate_json_report, generate_markdown_report

    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR / "batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]Loading senator handles...[/]")
    handles = fetch_senator_handles()

    console.print("[bold cyan]Loading HuggingFace senator tweets...[/]")
    try:
        hf_data = load_hf_senator_tweets()
    except Exception as e:
        console.print(f"[yellow]Could not load HF data: {e}[/]")
        hf_data = None

    # Load models once
    console.print("[bold cyan]Loading classification models...[/]")
    classifier = MultiHeadClassifier()
    classifier.load_all()

    fusion = ScoreFusion()
    analyzer = BehavioralAnalyzer()

    results = []
    import pandas as pd

    for _, row in handles.iterrows():
        name = f"{row['first_name']} {row['last_name']}"
        handle = row["twitter_handle"]
        console.print(f"\n[cyan]Analyzing {name} (@{handle})...[/]")

        # Filter HF data for this senator
        senator_tweets = pd.DataFrame()
        if hf_data is not None and "username" in hf_data.columns:
            senator_tweets = hf_data[
                hf_data["username"].str.lower() == handle.lower()
            ].copy()

        if senator_tweets.empty:
            console.print(f"  [dim]No tweets found, skipping[/]")
            continue

        console.print(f"  Found {len(senator_tweets)} tweets")

        # Run pipeline
        behavioral = analyzer.analyze(senator_tweets, senator_name=name, twitter_handle=handle)
        classified = classifier.classify_tweets(senator_tweets)
        classified = fusion.compute_tweet_virulence(classified)
        profile = fusion.aggregate_senator_profile(
            classified, behavioral,
            senator_name=name,
            twitter_handle=handle,
            party=row.get("party", ""),
            state=row.get("state", ""),
        )

        # Save reports
        slug = handle
        generate_json_report(profile, output_path=str(out_dir / f"{slug}_report.json"))
        generate_markdown_report(profile, output_path=str(out_dir / f"{slug}_report.md"))

        results.append({
            "senator": name,
            "handle": handle,
            "party": row.get("party", ""),
            "state": row.get("state", ""),
            "compulsion_score": profile.compulsion_score,
            "virulence_score": profile.virulence_score,
            "overall_risk": profile.overall_risk_score,
            "n_tweets": profile.n_tweets_analyzed,
        })

    # Summary table
    if results:
        summary = pd.DataFrame(results)
        summary = summary.sort_values("overall_risk", ascending=False)
        summary.to_csv(str(out_dir / "batch_summary.csv"), index=False)

        table = Table(title="Senator Analysis Summary")
        table.add_column("Senator")
        table.add_column("Party")
        table.add_column("State")
        table.add_column("Tweets")
        table.add_column("Compulsion", justify="right")
        table.add_column("Virulence", justify="right")
        table.add_column("Overall Risk", justify="right")

        for _, r in summary.head(20).iterrows():
            table.add_row(
                r["senator"],
                r["party"],
                r["state"],
                str(r["n_tweets"]),
                f"{r['compulsion_score']:.1f}",
                f"{r['virulence_score']:.1f}",
                f"{r['overall_risk']:.1f}",
            )
        console.print(table)
        console.print(f"\nFull results saved to {out_dir}/")


if __name__ == "__main__":
    cli()
