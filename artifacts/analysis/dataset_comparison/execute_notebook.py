from __future__ import annotations

import contextlib
import io
import json
import traceback
from pathlib import Path


def execute(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    execution_count = 0
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        stdout = io.StringIO()
        outputs = []
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                exec("".join(cell.get("source", [])), namespace, namespace)
            text = stdout.getvalue()
            if text:
                outputs.append({"name": "stdout", "output_type": "stream", "text": text.splitlines(True)})
        except Exception as exc:
            outputs.append(
                {
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "output_type": "error",
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
            cell["execution_count"] = execution_count
            cell["outputs"] = outputs
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            raise
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    execute(Path(__file__).with_name("dataset_comparison.ipynb"))
    print("notebook execution: passed")
