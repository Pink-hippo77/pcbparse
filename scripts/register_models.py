"""Register trained artifacts into models.lock.json.

Candidate status is the default; blessing requires explicit --bless --force,
formal metrics evidence, and a passing torch_run_status.json gate record.  Artifacts are copied
into models/candidates/<task>/ so lock paths stay workspace-relative.
"""
import argparse
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

ADAPTERS = {
    "component": "pcbparse.perception.torch_component_adapter:factory",
    "pin": "pcbparse.perception.torch_pin_adapter:factory",
    "ocr": "pcbparse.perception.torch_ocr_adapter:factory",
    "wire": "pcbparse.perception.torch_wire_adapter:factory",
    "junction": "pcbparse.perception.torch_junction_adapter:factory",
}
ARTIFACT_FILES = {
    "component": "model.pt",
    "pin": "model.pt",
    "ocr": "model.pt",
    "wire": "model.pt",
    "junction": "model.pt",
}
ARTIFACT_FORMATS = {
    "component": "checkpoint",
    "pin": "checkpoint",
    "ocr": "bundle",
    "wire": "torchscript",
    "junction": "checkpoint",
}
# GPU inference is the default runtime contract; ONNX models on this machine
# fall back to CPU when cuBLAS/cuDNN DLLs are missing, so torch-native
# checkpoint/torchscript formats are preferred for pin/junction/wire.
INFERENCE_DEFAULTS = {
    "component": {"device": "auto"},
    "pin": {"device": "auto"},
    "ocr": {"device": "auto"},
    "wire": {"device": "auto"},
    "junction": {"device": "auto"},
}
SUPPORTED_PRECISIONS = {"bf16", "float16", "float32", "fp16", "fp32", "int4", "int8", "mixed", "uint8"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with io.open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--source-dir", required=True, help="training output directory")
    parser.add_argument("--lock", default=None, help="models.lock.json path (default: models/models.lock.json)")
    parser.add_argument("--version", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--precision", default="fp32", choices=sorted(SUPPORTED_PRECISIONS))
    parser.add_argument("--inference", default=None, help="optional JSON adapter config (e.g. {\"score_threshold\":0.15})")
    parser.add_argument("--bless", action="store_true", help="mark entry blessed (requires --force and formal evidence)")
    parser.add_argument("--force", action="store_true", help="acknowledge policy override for blessing")
    args = parser.parse_args(argv)

    workspace = Path(__file__).resolve().parents[1]
    lock_path = Path(args.lock).expanduser() if args.lock else workspace / "models" / "models.lock.json"
    if not lock_path.is_absolute():
        lock_path = workspace / lock_path
    source = Path(args.source_dir).expanduser().resolve(strict=True)
    artifact_name = ARTIFACT_FILES[args.task]
    artifact = source / artifact_name
    if not artifact.is_file():
        print(f"missing artifact {artifact}", file=sys.stderr)
        return 1

    metrics_path = source / "metrics.json"
    formal = False
    gates_passed = False
    registerable = False
    if metrics_path.is_file():
        metrics = read_json(metrics_path)
        formal = bool(metrics.get("formal_evidence", False))
    status_path = source / "torch_run_status.json"
    if status_path.is_file():
        status = read_json(status_path)
        formal = formal and bool(status.get("formal_evidence", False))
        gates_passed = bool(status.get("gates_passed", False))
        registerable = bool(status.get("registerable", False))
    if args.bless and not formal:
        print("refusing to bless: formal evidence is not true in metrics.json and torch_run_status.json", file=sys.stderr)
        return 2
    if args.bless and not gates_passed:
        print("refusing to bless: training gates_passed is not true", file=sys.stderr)
        return 2
    if args.bless and not registerable:
        print("refusing to bless: training registerable is not true", file=sys.stderr)
        return 2
    if args.bless and not args.force:
        print("refusing to bless: pass --force to acknowledge the blessing", file=sys.stderr)
        return 2

    digest = sha256_file(artifact)
    candidate_dir = workspace / "models" / "candidates" / args.task
    candidate_dir.mkdir(parents=True, exist_ok=True)
    destination = candidate_dir / artifact.name
    if not destination.is_file() or sha256_file(destination) != digest:
        shutil.copy2(artifact, destination)
    relative = Path("candidates") / args.task / artifact.name

    inference = dict(INFERENCE_DEFAULTS.get(args.task, {}))
    if args.inference:
        override = json.loads(args.inference)
        if not isinstance(override, dict):
            raise SystemExit("--inference must be a JSON object")
        inference.update(override)

    entry = {
        "task": args.task,
        "path": relative.as_posix(),
        "sha256": digest,
        "status": "blessed" if args.bless else "candidate",
        "blessed": bool(args.bless),
        "version": args.version,
        "license": args.license,
        "precision": args.precision,
        "adapter": ADAPTERS[args.task],
        "artifact_format": ARTIFACT_FORMATS[args.task],
        "inference": inference,
        "note": (
            "Formal evidence verified; blessed for competition profile."
            if args.bless
            else "Diagnostic/unblessed artifact; do not submit until formal gates pass."
        ),
    }
    # 顶层同时写入推理参数：component/ocr/wire/junction 适配器读顶层键，
    # pin 适配器两者都读；保证 device/threshold 等对全部适配器生效。
    entry.update(inference)

    lock = {"schema_version": "1.0", "models": []}
    if lock_path.is_file():
        lock = read_json(lock_path)
        if lock.get("schema_version") != "1.0":
            print("unsupported lock schema", file=sys.stderr)
            return 1
    lock.setdefault("models", [])
    lock["models"] = [item for item in lock["models"] if item.get("task") != args.task]
    lock["models"].append(entry)
    lock["models"].sort(key=lambda item: sorted(ADAPTERS).index(item.get("task", "")))

    backup = lock_path.with_suffix(lock_path.suffix + ".bak")
    if lock_path.is_file():
        shutil.copy2(lock_path, backup)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(lock_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(lock, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"registered {args.task} -> {lock_path} ({entry['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
