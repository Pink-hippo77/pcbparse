"""Debug the CLI integration test failure."""
import sys, traceback
try:
    from pcbparse.cli import main
    import io, contextlib, json, tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    lock = root / "lock.json"
    lock.write_text(json.dumps({"schema_version": "1.0", "models": []}))
    dataset = Path("E:/EDA挑战题目六/Dataset300/Dataset300/0001/0001-KiCad.png")
    out_dir = root / "out"
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = main(["infer", "--input", str(dataset), "--output", str(out_dir),
                         "--profile", "debug", "--provider", "dataset_debug",
                         "--models-lock", str(lock)])
        except SystemExit as e:
            code = e.code
    print(f"Exit code: {code}", file=sys.stderr)
    em_path = out_dir / "error_manifest.json"
    if em_path.exists():
        with open(em_path, "rb") as f:
            data = json.loads(f.read())
        for row in data:
            if "error" in row:
                err = row["error"]
                print(f"  code: {err.get('code')}", file=sys.stderr)
                print(f"  type: {err.get('type')}", file=sys.stderr)
                msg = str(err.get("message", ""))
                print(f"  message: {msg[:500]}", file=sys.stderr)
                print(file=sys.stderr)
    else:
        print("No error manifest", file=sys.stderr)
except Exception:
    traceback.print_exc(file=sys.stderr)
    print("Output files:", list(out_dir.rglob("*")) if out_dir.exists() else "none")
