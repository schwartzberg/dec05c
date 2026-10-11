"""Schwartzberg, Paul. Execute HW5 notebooks and create matching HTML reports.

Run from the cscie89 environment:
    python run_all.py
Or choose one problem:
    python run_all.py --problems 2
Training cells always train again. This does not substitute cached training metrics.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent

class ProgressClient(NotebookClient):
    """Print notebook progress while retaining its normal saved outputs."""
    def process_message(self, msg, cell, cell_index):
        if msg.get("msg_type") == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)

def run_problem(number, kernel, timeout):
    notebook_path = ROOT / f"E89_Schwartzberg_Paul_ProblemSet05_{number:02d}.ipynb"
    before = notebook_path.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    # Keep a recoverable copy outside the repository before replacing outputs.
    backup_dir = Path(tempfile.gettempdir()) / "hw05_notebook_backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    shutil.copy2(notebook_path, backup_dir / f"{notebook_path.stem}_{stamp}.ipynb")
    notebook = nbformat.reads(before.decode("utf-8"), as_version=4)
    nbformat.validate(notebook)
    started = datetime.now(timezone.utc).isoformat()
    tick = time.perf_counter()
    print(f"\nRunning Problem {number}: {notebook_path.name}", flush=True)
    client = ProgressClient(notebook, kernel_name=kernel, timeout=timeout,
                            resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    except Exception:
        # Save the failure separately; retain the user's original notebook.
        failed_path = backup_dir / f"{notebook_path.stem}_{stamp}_failed.ipynb"
        nbformat.write(notebook, failed_path)
        print("Execution failed. Diagnostic notebook:", failed_path, flush=True)
        raise
    if hashlib.sha256(notebook_path.read_bytes()).hexdigest() != digest:
        executed_path = backup_dir / f"{notebook_path.stem}_{stamp}_executed.ipynb"
        nbformat.write(notebook, executed_path)
        raise RuntimeError(f"The notebook changed during execution. Kept your edits; executed copy: {executed_path}")
    ended = datetime.now(timezone.utc).isoformat()
    notebook.metadata["hw5_execution"] = {
        "started_utc": started, "finished_utc": ended, "kernel": kernel,
        "elapsed_seconds": time.perf_counter() - tick, "all_code_cells_executed": True,
    }
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.strip():
            assert cell.execution_count is not None
            assert not any(output.output_type == "error" for output in cell.outputs)
    nbformat.write(notebook, notebook_path)
    # Export the executed notebook: its plots are embedded in the HTML file.
    html, _ = HTMLExporter().from_notebook_node(notebook)
    html_path = notebook_path.with_suffix(".html")
    html_path.write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n", encoding="utf-8")
    print(f"Problem {number} passed. Notebook and HTML saved.", flush=True)
    return {"problem": number, **dict(notebook.metadata["hw5_execution"])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=int, nargs="+", choices=range(1, 5), default=[1, 2, 3, 4])
    parser.add_argument("--kernel", default="cscie89")
    parser.add_argument("--timeout", type=int, default=7200, help="Maximum seconds for any one code cell.")
    args = parser.parse_args()
    log_path = ROOT / "outputs/execution_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    problems = sorted(set(args.problems))
    # Catch source errors before starting any long training run.
    for problem in problems:
        path = ROOT / f"E89_Schwartzberg_Paul_ProblemSet05_{problem:02d}.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code" and cell.source.strip():
                ast.parse(cell.source)
    previous = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"runs": []}
    runs_by_problem = {run["problem"]: run for run in previous["runs"]}
    for problem in problems:
        runs_by_problem[problem] = run_problem(problem, args.kernel, args.timeout)
        runs = [runs_by_problem[key] for key in sorted(runs_by_problem)]
        log_path.write_text(json.dumps({"author": "Schwartzberg, Paul", "runs": runs}, indent=2) + "\n", encoding="utf-8")
    if problems == [1, 2, 3, 4]:
        subprocess.run([sys.executable, str(ROOT / "check_submission.py")], cwd=ROOT, check=True)
    print("\nAll requested notebooks executed successfully.", flush=True)

if __name__ == "__main__":
    main()
