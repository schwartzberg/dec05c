"""Schwartzberg, Paul. Execute the combined HW5 notebook and export its HTML.

Run from the cscie89 environment:
    python run_all.py
Or run an archived problem separately:
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

def run_notebook(notebook_path, kernel, timeout, label):
    notebook_path = Path(notebook_path)
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
    print(f"\nRunning {label}: {notebook_path.name}", flush=True)
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
    print(f"{label} passed. Notebook and HTML saved.", flush=True)
    return {"notebook": notebook_path.name, **dict(notebook.metadata["hw5_execution"])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=int, nargs="+", choices=range(1, 5), default=None, help="Optional: run selected archived individual problems instead of the combined notebook.")
    parser.add_argument("--kernel", default="cscie89")
    parser.add_argument("--timeout", type=int, default=7200, help="Maximum seconds for any one code cell.")
    args = parser.parse_args()
    combined = args.problems is None
    if combined:
        jobs = [(ROOT / "E89_Schwartzberg_Paul_ProblemSet05_01.ipynb", "combined HW5 notebook")]
    else:
        jobs = [(ROOT / "individual_problems" / f"E89_Schwartzberg_Paul_ProblemSet05_{number:02d}.ipynb",
                 f"archived Problem {number}") for number in sorted(set(args.problems))]
    # Catch source errors before starting any long training run.
    for path, label in jobs:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code" and cell.source.strip():
                ast.parse(cell.source)
    log_path = ROOT / "outputs" / ("execution_log.json" if combined else "individual_execution_log.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for path, label in jobs:
        runs.append(run_notebook(path, args.kernel, args.timeout, label))
        log_path.write_text(json.dumps({"author": "Schwartzberg, Paul", "layout": "combined" if combined else "individual",
                                        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    if combined:
        subprocess.run([sys.executable, str(ROOT / "check_submission.py")], cwd=ROOT, check=True)
    print("\nAll requested cells executed successfully.", flush=True)

if __name__ == "__main__":
    main()
