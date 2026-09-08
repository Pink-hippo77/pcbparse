from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_release_case_suite.py"
SPEC = importlib.util.spec_from_file_location("release_case_suite", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class ReleaseCaseSuiteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _row(
        self,
        sample_id: str,
        *,
        split: str,
        group: str | None = None,
        group_size: int = 1,
    ) -> dict[str, object]:
        payload = f"image-{sample_id}".encode()
        relative = Path(sample_id) / f"{sample_id}.png"
        image = self.dataset / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(payload)
        (image.parent / f"{sample_id}.json").write_text("sidecar", encoding="utf-8")
        return {
            "schema_version": 1,
            "sample_id": sample_id,
            "split": split,
            "duplicate_group": group or f"group-{sample_id}",
            "duplicate_group_size": group_size,
            "source_class": "fixture",
            "source_family": None,
            "base_name": sample_id,
            "canvas": {"width": 10, "height": 10},
            "files": {
                "image": {
                    "path": relative.as_posix(),
                    "sha256": _sha(payload),
                    "bytes": len(payload),
                    "width": 10,
                    "height": 10,
                    "detected_format": "PNG",
                },
                "json": {"path": f"{sample_id}/{sample_id}.json"},
            },
        }

    def _build(self, rows: list[dict[str, object]], *, target: int = 3):
        manifest = self.root / "source.jsonl"
        _write_jsonl(manifest, rows)
        output_manifest = self.root / "release.jsonl"
        images_root = self.root / "images-only"
        report = self.root / "report.json"
        result = MODULE.build_release_case_suite(
            workspace_root=self.root,
            source_split_manifest=manifest,
            dataset_root=self.dataset,
            output_manifest=output_manifest,
            output_images_root=images_root,
            report=report,
            primary_split="test",
            supplemental_split="val",
            output_split="release",
            target_count=target,
        )
        return result, output_manifest, images_root, report

    def test_builds_exact_complete_image_only_suite(self) -> None:
        rows = [
            self._row("0002", split="test", group="pair", group_size=2),
            self._row("0001", split="test", group="pair", group_size=2),
            self._row("0004", split="val"),
            self._row("0003", split="val"),
        ]
        result, manifest, images_root, report = self._build(rows, target=3)

        released = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(["0001", "0002", "0003"], [row["sample_id"] for row in released])
        self.assertEqual({"release"}, {row["split"] for row in released})
        self.assertEqual({"test", "val"}, {row["release_source_split"] for row in released})
        self.assertTrue(all(set(row["files"]) == {"image"} for row in released))
        files = sorted(path for path in images_root.rglob("*") if path.is_file())
        self.assertEqual(3, len(files))
        self.assertTrue(all(path.suffix == ".png" for path in files))
        self.assertTrue(report.is_file())
        self.assertEqual(3, result["selection"]["selected_count"])
        self.assertEqual({"test": 2, "val": 1}, result["selection"]["selected_source_split_counts"])
        self.assertTrue(result["selection"]["complete_duplicate_groups"])
        self.assertFalse(result["accuracy_claim"])

    def test_refuses_duplicate_group_leakage(self) -> None:
        rows = [
            self._row("0001", split="test", group="leak", group_size=2),
            self._row("0002", split="val", group="leak", group_size=2),
        ]
        with self.assertRaisesRegex(MODULE.ReleaseCaseSuiteError, "leak"):
            self._build(rows, target=1)

    def test_refuses_group_size_mismatch(self) -> None:
        rows = [self._row("0001", split="test", group_size=2)]
        with self.assertRaisesRegex(MODULE.ReleaseCaseSuiteError, "declares"):
            self._build(rows, target=1)

    def test_refuses_insufficient_supplemental_singletons(self) -> None:
        rows = [
            self._row("0001", split="test"),
            self._row("0002", split="val", group="pair", group_size=2),
            self._row("0003", split="val", group="pair", group_size=2),
        ]
        with self.assertRaisesRegex(MODULE.ReleaseCaseSuiteError, "singleton"):
            self._build(rows, target=2)

    def test_refuses_corrupted_source_image(self) -> None:
        rows = [self._row("0001", split="test")]
        rows[0]["files"]["image"]["sha256"] = "0" * 64  # type: ignore[index]
        with self.assertRaisesRegex(MODULE.ReleaseCaseSuiteError, "SHA-256 mismatch"):
            self._build(rows, target=1)

    def test_refuses_any_existing_output_without_overwrite(self) -> None:
        rows = [self._row("0001", split="test")]
        source = self.root / "source.jsonl"
        _write_jsonl(source, rows)
        output = self.root / "release.jsonl"
        output.write_text("sentinel", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "must be new"):
            MODULE.build_release_case_suite(
                workspace_root=self.root,
                source_split_manifest=source,
                dataset_root=self.dataset,
                output_manifest=output,
                output_images_root=self.root / "images",
                report=self.root / "report.json",
                target_count=1,
            )
        self.assertEqual("sentinel", output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
