from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class DataToolsImportPathTests(unittest.TestCase):
    def test_repository_audits_import_when_src_is_the_only_pythonpath(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        source = repository / "src"
        expected_package = (source / "data_tools" / "__init__.py").resolve()
        expected_review = (repository / "data_tools" / "junction_review_pack.py").resolve()
        expected_truth = (repository / "data_tools" / "junction_truth_audit.py").resolve()
        expected_yolo_export = (repository / "data_tools" / "export_yolo_dataset.py").resolve()
        expected_yolo_validate = (repository / "data_tools" / "validate_yolo_dataset.py").resolve()
        code = (
            "from pathlib import Path; "
            "import data_tools; "
            "import data_tools.junction_review_pack as review; "
            "import data_tools.junction_truth_audit as truth; "
            "import data_tools.export_yolo_dataset as yolo_export; "
            "import data_tools.validate_yolo_dataset as yolo_validate; "
            f"assert Path(data_tools.__file__).resolve() == Path({str(expected_package)!r}); "
            f"assert Path(review.__file__).resolve() == Path({str(expected_review)!r}); "
            f"assert Path(truth.__file__).resolve() == Path({str(expected_truth)!r}); "
            f"assert Path(yolo_export.__file__).resolve() == Path({str(expected_yolo_export)!r}); "
            f"assert Path(yolo_validate.__file__).resolve() == Path({str(expected_yolo_validate)!r})"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(source)
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=temporary,
                env=environment,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        self.assertEqual(
            0,
            completed.returncode,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
