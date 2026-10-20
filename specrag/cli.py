"""specrag — command line.

    specrag ingest                         # PDF -> chunks -> embeddings (once)
    specrag extract "torque for brake caliper bolts"
    specrag extract "brake fluid capacity" --format csv --out out.csv
    specrag batch eval/queries.txt         # -> outputs/specs.json + outputs/specs.csv
    specrag search "sway bar link nut"     # retrieval only, no LLM
    specrag eval                           # golden-set retrieval + extraction
    specrag export-index                   # build index/ for deployment
    specrag serve                          # API + UI on http://localhost:8000
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from specrag import config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Vehicle spec extraction over a service manual.")
console = Console()


@app.command()
def ingest(
    pdf: Path = typer.Option(config.PDF_PATH, "--pdf", help="Path to the service manual PDF."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Only extract, clean, chunk and parse tables."),
):
    """Extract, clean, chunk, parse spec tables, embed into Qdrant."""
    from specrag.evaluate import make_generated
    from specrag.pipeline import ingest as run_ingest

    if not pdf.exists():
        console.print(f"[red]PDF not found:[/red] {pdf}\nPut the manual in data/ or pass --pdf.")
        raise typer.Exit(1)

    with console.status("Reading PDF, cleaning, chunking, embedding...") as status:
        stats = run_ingest(pdf, embed=not no_embed, progress=lambda n: status.update(f"Embedded {n} chunks..."))

    rows = json.loads(config.TABLES_FILE.read_text(encoding="utf-8"))
    n_generated = make_generated(rows)

    table = Table(title="Ingest", show_header=False)
    for key, value in stats.items():
        table.add_row(key, str(value))
    table.add_row("generated golden queries", str(n_generated))
    console.print(table)


def _print_result(result) -> None:
    if result.note and not result.specs:
        console.print(f"[yellow]{result.note}[/yellow]")
    table = Table(title=f"{result.query}  [dim]({result.mode}, {sum(result.timings_ms.values())} ms)[/dim]")
    for col in ("Component", "Type", "Value", "Unit", "Alt", "Condition", "Section", "Pages"):
        table.add_column(col)
    for s in result.specs:
        alt = f"{s.alt_value} {s.alt_unit or ''}".strip() if s.alt_value else ""
        table.add_row(s.component, s.spec_type, s.value, s.unit, alt, s.condition or "",
                      f"{s.section}", s.pages)
    console.print(table)
    if result.rejected:
        console.print(f"[dim]{len(result.rejected)} item(s) rejected by the grounding check:[/dim]")
        for r in result.rejected:
            console.print(f"  [dim]- {r.get('component')} = {r.get('value')} {r.get('unit')} ({r.get('reason')})[/dim]")


def _run_or_exit(fn, *args, **kwargs):
    """Config problems (no key, no index) are user errors: say what to do,
    don't print a traceback."""
    try:
        return fn(*args, **kwargs)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


def _write(results, fmt: str, out: Path | None, quiet: bool = False) -> None:
    from specrag.output import to_csv, to_json

    if fmt == "table":
        for r in results:
            _print_result(r)
        return
    text = to_csv(results) if fmt == "csv" else to_json(results, full=(fmt == "full"))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        if not quiet:
            console.print(f"Wrote {out}")
    else:
        print(text)


@app.command()
def extract(
    query: str = typer.Argument(..., help='e.g. "Torque for brake caliper bolts"'),
    mode: str = typer.Option(config.DEFAULT_MODE, help="llm | rules"),
    top_k: int = typer.Option(config.TOP_K, "--top-k", "-k"),
    system: str = typer.Option(None, help="Suspension | Driveline | Brakes"),
    fmt: str = typer.Option("table", "--format", "-f", help="table | json | csv | full"),
    out: Path = typer.Option(None, "--out", "-o"),
):
    """Retrieve + extract structured specs for one query."""
    from specrag.extractor import extract as run

    _write([_run_or_exit(run, query, top_k=top_k, system=system, mode=mode)], fmt, out)


@app.command()
def batch(
    queries_file: Path = typer.Argument(..., help="Text file, one query per line."),
    mode: str = typer.Option(config.DEFAULT_MODE, help="llm | rules"),
    top_k: int = typer.Option(config.TOP_K, "--top-k", "-k"),
    fmt: str = typer.Option("both", "--format", "-f", help="both | json | csv | full"),
    out: Path = typer.Option(Path("outputs/specs"), "--out", "-o",
                             help="Output path. With --format both, .json and .csv are added."),
):
    """Run many queries and write JSON and/or CSV.

    Results are written after every query, so a quota error part-way through
    keeps everything answered so far instead of losing it.
    """
    from specrag.extractor import extract as run

    queries = [q.strip() for q in queries_file.read_text(encoding="utf-8").splitlines()
               if q.strip() and not q.startswith("#")]
    results = []

    def save():
        if fmt == "both":
            _write(results, "json", out.with_suffix(".json"), quiet=True)
            _write(results, "csv", out.with_suffix(".csv"), quiet=True)
        else:
            _write(results, fmt, out, quiet=True)

    for i, q in enumerate(queries, start=1):
        try:
            with console.status(f"[{i}/{len(queries)}] {q}"):
                results.append(run(q, top_k=top_k, mode=mode))
        except RuntimeError as exc:
            if results:
                save()
            console.print(f"[red]{exc}[/red]")
            console.print(f"Saved {len(results)} of {len(queries)} answers. Re-run later for the rest "
                          f"(or remove the answered lines from {queries_file.name}).")
            raise typer.Exit(1)
        console.print(f"[green]✓[/green] {q} -> {len(results[-1].specs)} spec(s)")
        save()

    targets = [out.with_suffix(".json"), out.with_suffix(".csv")] if fmt == "both" else [out]
    console.print("Wrote " + ", ".join(str(t) for t in targets))


@app.command()
def search(
    query: str = typer.Argument(...),
    top_k: int = typer.Option(config.TOP_K, "--top-k", "-k"),
    method: str = typer.Option("hybrid", help="hybrid | dense | bm25"),
):
    """Retrieval only — what the LLM would see."""
    from specrag.retrieve import search as run

    for i, hit in enumerate(run(query, top_k=top_k, method=method), start=1):
        c = hit.chunk
        console.rule(f"[{i}] {c.section_id} · {c.doc_title} · {c.heading or c.kind} · p{c.pages} "
                     f"[dim](dense #{hit.dense_rank}, bm25 #{hit.bm25_rank})[/dim]")
        console.print(c.text[:700])


@app.command(name="eval")
def evaluate(
    k: int = typer.Option(config.TOP_K, "-k"),
    mode: str = typer.Option(config.DEFAULT_MODE, help="Extraction mode: llm | rules"),
    sample: int = typer.Option(30, help="Generated queries added to the extraction eval (fixed seed). 19 + 30 = 49 LLM calls."),
    retrieval_only: bool = typer.Option(False, "--retrieval-only"),
    out: Path = typer.Option(config.PROJECT_ROOT / "eval" / "results.json", "--out"),
):
    """Score retrieval (dense vs BM25 vs hybrid) and extraction on the golden set."""
    from specrag.evaluate import eval_extraction, eval_retrieval, load_golden

    items = load_golden()
    report: dict = {"retrieval": eval_retrieval(items, k=k)}

    table = Table(title=f"Retrieval — {report['retrieval']['n']} queries, k={k}")
    table.add_column("method")
    table.add_column("row recall@k (all)")
    table.add_column("handwritten")
    table.add_column("generated")
    table.add_column("hit rate@k")
    for method, r in report["retrieval"]["methods"].items():
        o = r["row_recall_by_origin"]
        table.add_row(method, f"{r['row_recall']:.1%}", f"{o.get('handwritten', 0):.1%}",
                      f"{o.get('generated', 0):.1%}", f"{r['hit_rate']:.1%}")
    console.print(table)

    if not retrieval_only:
        modes = ["rules", mode] if mode != "rules" else ["rules"]
        report["extraction"] = {}
        table = Table(title="Extraction — precision / recall over (value, section)")
        for col in ("mode", "queries", "precision", "recall", "F1", "handwritten P / R",
                    "generated P / R", "refusal acc."):
            table.add_column(col)
        for m in modes:
            with console.status(f"Extraction eval ({m})..."):
                r = eval_extraction(items, mode=m, k=k, sample=sample)
            report["extraction"][m] = r
            if r.get("errors"):
                console.print(f"[yellow]{len(r['errors'])} question(s) failed in {m} mode and were skipped "
                              f"(see 'errors' in the report). Re-run to include them.[/yellow]")
            refusal = f"{r['refusal_accuracy']:.0%}" if r["refusal_accuracy"] is not None else "-"
            by = r.get("by_origin", {})
            pr = lambda o: f"{by[o]['precision']:.1%} / {by[o]['recall']:.1%}" if o in by else "-"
            table.add_row(m, str(r["n"]), f"{r['precision']:.1%}", f"{r['recall']:.1%}", f"{r['f1']:.1%}",
                          pr("handwritten"), pr("generated"), refusal)
        console.print(table)

    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"Full report: {out}")


@app.command(name="golden")
def golden():
    """Regenerate eval/generated.json from the parsed tables."""
    from specrag.evaluate import make_generated

    n = make_generated(json.loads(config.TABLES_FILE.read_text(encoding="utf-8")))
    console.print(f"Wrote {n} generated golden queries.")


@app.command(name="export-index")
def export_index():
    """Write index/vectors.npz + index/spec_rows.json for deployment."""
    import shutil

    from specrag.index import export_vectors

    info = export_vectors(config.INDEX_EXPORT)
    shutil.copy(config.TABLES_FILE, config.INDEX_EXPORT.parent / "spec_rows.json")
    console.print(f"Exported {info['count']} vectors ({info['mb']} MB) to {config.INDEX_EXPORT.parent}/")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the API and the web UI."""
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
