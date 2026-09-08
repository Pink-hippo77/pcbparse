from __future__ import annotations

"""Real, compact PyTorch OCR detector/recognizer and artifact lifecycle.

The module stays importable in the lightweight production environment: torch,
onnxruntime, and their native libraries are loaded only by operations that need
them.  The detector is a Tiny U-Net trained with ignore-aware BCE + Dice.  The
recognizer is a dynamic-width CNN/BiLSTM trained with strictly validated CTC.
"""

import hashlib
import importlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from PIL import Image

from pcbparse.training.torch_ocr import CTCCharset


BUNDLE_SCHEMA = "pcbparse.ocr.torch-bundle.v1"
EXPORT_SCHEMA = "pcbparse.ocr.torch-export-consistency.v1"
_DETECTOR_CLASS: type[Any] | None = None
_RECOGNIZER_CLASS: type[Any] | None = None


class TorchOCRRuntimeError(RuntimeError):
    """A PyTorch OCR operation cannot be completed safely."""


class OCRBundleError(TorchOCRRuntimeError):
    """The unified detector/recognizer bundle is corrupt or incompatible."""


class OCRExportError(TorchOCRRuntimeError):
    """An OCR submodel export failed or disagrees with eager inference."""


class TorchOCREvaluationDataError(TorchOCRRuntimeError):
    """Frozen OCR evaluation data is corrupt or violates its train-only charset."""


@dataclass(frozen=True, slots=True)
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    area: int

    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


@dataclass(frozen=True, slots=True)
class OCRBundleInfo:
    path: str
    sha256: str
    bytes: int
    version: str
    epoch: int
    global_step: int
    lineage: dict[str, Any]


@dataclass(slots=True)
class LoadedOCRBundle:
    detector: Any
    recognizer: Any
    charset: CTCCharset
    preprocess: dict[str, Any]
    lineage: dict[str, Any]
    metrics: dict[str, Any]
    epoch: int
    global_step: int
    checkpoint_sha256: str
    version: str
    detector_optimizer_state: Any | None
    recognizer_optimizer_state: Any | None
    path: Path


@dataclass(frozen=True, slots=True)
class OCRExportReport:
    schema: str
    path: str
    kind: Literal["detector", "recognizer"]
    format: Literal["torchscript", "onnx"]
    artifact_sha256: str
    artifact_bytes: int
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, ...]
    max_abs_error: float
    mean_abs_error: float
    tolerance: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TorchOCREvaluationDataset:
    """Recognition evaluation dataset that preserves train-charset OOV rows.

    The training :class:`TorchOCRRecognitionDataset` deliberately fails closed
    when any selected text cannot be encoded.  That is the correct training
    contract, but val/test evaluation must retain those rows as unavoidable
    errors.  This independent reader verifies the same frozen artifacts and
    returns an empty target plus ``eligible=False`` for OOV rows; it never
    changes ``charset.txt`` and never substitutes an unknown token.
    """

    contract = "pcbparse.ocr.train-charset-evaluation.v1"

    def __init__(
        self,
        materialization_root: str | Path,
        *,
        split: Literal["train", "val", "test"],
        source_root: str | Path | None = None,
        image_mode: Literal["materialized", "lazy", "auto"] = "auto",
        image_height: int = 48,
        verify_image_hashes: bool = True,
        max_records: int | None = None,
    ) -> None:
        _require_torch()
        if split not in ("train", "val", "test"):
            raise TorchOCREvaluationDataError("invalid OCR evaluation split")
        if image_mode not in ("materialized", "lazy", "auto"):
            raise TorchOCREvaluationDataError("invalid OCR evaluation image_mode")
        if type(image_height) is not int or image_height <= 0:
            raise TorchOCREvaluationDataError("image_height must be positive")
        if type(verify_image_hashes) is not bool:
            raise TorchOCREvaluationDataError("verify_image_hashes must be boolean")
        if max_records is not None and (
            type(max_records) is not int or max_records <= 0
        ):
            raise TorchOCREvaluationDataError("max_records must be positive")
        self.root = Path(materialization_root).expanduser().resolve(strict=True)
        self.source_root = (
            Path(source_root).expanduser().resolve(strict=True)
            if source_root is not None else None
        )
        self.split = split
        self.image_mode = image_mode
        self.image_height = image_height
        self.verify_image_hashes = verify_image_hashes
        self._verified_images: dict[str, tuple[str, int, int]] = {}

        manifest = _evaluation_json_object(self.root / "manifest.json")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("kind") != "pcbparse_ocr_paddle_materialization"
            or manifest.get("complete") is not True
        ):
            raise TorchOCREvaluationDataError(
                "not a complete frozen OCR materialization manifest"
            )
        for relative in ("recognition_index.jsonl", "charset.txt", "report.json"):
            _verify_evaluation_artifact(self.root, manifest, relative)
        report = _evaluation_json_object(self.root / "report.json")
        policy = report.get("policy")
        if not isinstance(policy, dict) or _canonical_sha256(policy) != manifest.get("policy_sha256"):
            raise TorchOCREvaluationDataError("OCR evaluation policy digest mismatch")
        max_length = policy.get("max_text_length")
        if type(max_length) is not int or max_length <= 0:
            raise TorchOCREvaluationDataError("OCR evaluation max_text_length is invalid")
        if policy.get("overlength") != "exclude_not_truncate" or policy.get("raw_text") != "preserved_without_strip_normalization_or_truncation":
            raise TorchOCREvaluationDataError("OCR evaluation raw-text policy is unsafe")
        self.charset = CTCCharset.from_file(self.root / "charset.txt")
        report_charset = report.get("charset")
        if not isinstance(report_charset, dict):
            raise TorchOCREvaluationDataError("OCR report lacks charset evidence")
        training_characters = report_charset.get("training_characters")
        if training_characters != list(self.charset.characters):
            raise TorchOCREvaluationDataError(
                "charset.txt differs from report train-only characters"
            )
        if report_charset.get("training_character_count") != len(self.charset.characters):
            raise TorchOCREvaluationDataError("training charset count mismatch")

        rows = _evaluation_jsonl(self.root / "recognition_index.jsonl", "record_id")
        counts = manifest.get("counts")
        if not isinstance(counts, dict) or counts.get("recognition_index") != len(rows):
            raise TorchOCREvaluationDataError("recognition index count mismatch")
        # Reuse the already-audited row validator without invoking the
        # training Dataset's eager charset encoding.
        from pcbparse.training import torch_ocr as frozen_contract

        selected: list[dict[str, Any]] = []
        charset_set = set(self.charset.characters)
        observed_training_characters: set[str] = set()
        included_count = 0
        excluded_count = 0
        for row in rows:
            try:
                frozen_contract._validate_index_record(  # type: ignore[attr-defined]
                    row, max_text_length=max_length
                )
            except Exception as exc:
                raise TorchOCREvaluationDataError(
                    f"invalid frozen recognition row {row.get('record_id')}: {exc}"
                ) from exc
            if row["status"] == "included":
                included_count += 1
            else:
                excluded_count += 1
                reason = str(row.get("exclusion_reason", "")).lower()
                if "oov" in reason or "charset" in reason:
                    raise TorchOCREvaluationDataError(
                        f"recognition row {row['record_id']} was excluded by charset/OOV policy"
                    )
            if row["split"] == "train" and row["status"] == "included":
                observed_training_characters.update(row["content"])
            if row["split"] == split and row["status"] == "included":
                copied = dict(row)
                copied["oov_characters"] = sorted(
                    set(row["content"]) - charset_set, key=ord
                )
                selected.append(copied)
        if (
            counts.get("recognition_included") != included_count
            or counts.get("recognition_excluded") != excluded_count
        ):
            raise TorchOCREvaluationDataError(
                "recognition included/excluded counts differ from manifest"
            )
        if tuple(sorted(observed_training_characters, key=ord)) != self.charset.characters:
            raise TorchOCREvaluationDataError(
                "charset.txt is not exactly the character set derived from included train rows"
            )
        selected.sort(key=lambda row: row["record_id"])
        if max_records is not None:
            selected = selected[:max_records]
        if not selected:
            raise TorchOCREvaluationDataError(
                f"no included OCR evaluation rows for split {split!r}"
            )
        self.records = tuple(selected)
        self.manifest = manifest
        self.report = report
        self.policy = dict(policy)

        actual_oov_records = [
            row["record_id"] for row in self.records if row["oov_characters"]
        ]
        actual_oov_characters = sorted(
            {
                character
                for row in self.records
                for character in row["oov_characters"]
            },
            key=ord,
        )
        # Full-split loads must match the materializer's declared OOV audit.
        if max_records is None and split in ("val", "test"):
            declared = report_charset.get("oov_by_split", {}).get(split)
            if not isinstance(declared, dict):
                raise TorchOCREvaluationDataError(
                    f"OCR report lacks {split} OOV evidence"
                )
            if declared.get("record_ids") != actual_oov_records:
                raise TorchOCREvaluationDataError(
                    f"{split} OOV record audit differs from recognition index"
                )
            if declared.get("characters") != actual_oov_characters:
                raise TorchOCREvaluationDataError(
                    f"{split} OOV character audit differs from recognition index"
                )
        if image_mode == "lazy" and self.source_root is None:
            raise TorchOCREvaluationDataError("source_root is required for lazy evaluation")
        if image_mode == "materialized" and manifest.get("materialize_images") is not True:
            raise TorchOCREvaluationDataError(
                "materialized evaluation requested from a lazy-only freeze"
            )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def eligible_count(self) -> int:
        return sum(not row["oov_characters"] for row in self.records)

    @property
    def oov_sample_count(self) -> int:
        return len(self.records) - self.eligible_count

    @property
    def oov_characters(self) -> tuple[str, ...]:
        return tuple(sorted(
            {
                character
                for row in self.records
                for character in row["oov_characters"]
            },
            key=ord,
        ))

    def coverage_summary(self) -> dict[str, Any]:
        total = len(self.records)
        charset_set = set(self.charset.characters)
        occurrences = sum(
            sum(character not in charset_set for character in row["content"])
            for row in self.records
        )
        return {
            "samples": total,
            "eligible": self.eligible_count,
            "oov_samples": self.oov_sample_count,
            "oov_characters": occurrences,
            "oov_unique_characters": len(self.oov_characters),
            "oov_character_values": list(self.oov_characters),
            "oov_codepoints": [f"U+{ord(character):04X}" for character in self.oov_characters],
            "coverage": self.eligible_count / total,
            "charset_source": "train_only",
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = _require_torch()
        row = self.records[index]
        image, origin = self._load_crop(row)
        try:
            grayscale = image.convert("L")
            raw_width, raw_height = grayscale.size
            width = max(
                1,
                int(math.floor(raw_width * self.image_height / raw_height + 0.5)),
            )
            resized = grayscale.resize(
                (width, self.image_height), Image.Resampling.BILINEAR
            )
            array = np.asarray(resized, dtype=np.float32).copy() / np.float32(255.0)
            image_tensor = torch.from_numpy(array).unsqueeze(0).contiguous()
        finally:
            image.close()
        oov = tuple(row["oov_characters"])
        eligible = not oov
        if eligible:
            encoded = self.charset.encode(row["content"])
            target = torch.tensor(encoded, dtype=torch.long)
        else:
            # Empty is a sentinel paired with eligible=False, never an UNK or
            # silently shortened reference.
            target = torch.empty(0, dtype=torch.long)
        return {
            "image": image_tensor,
            "image_width": torch.tensor(width, dtype=torch.long),
            "target": target,
            "target_length": torch.tensor(target.numel(), dtype=torch.long),
            "text": row["content"],
            "eligible": eligible,
            "oov_characters": list(oov),
            "record_id": row["record_id"],
            "sample_id": row["sample_id"],
            "metadata": {
                "record_id": row["record_id"],
                "sample_id": row["sample_id"],
                "split": row["split"],
                "source_image": row["source_image"],
                "source_image_sha256": row["source_image_sha256"],
                "crop_box": dict(row["crop_box"]),
                "image_origin": origin,
                "raw_crop_size": [raw_width, raw_height],
                "resized_size": [width, self.image_height],
                "charset_source": "train_only",
                "eligible": eligible,
                "oov_characters": list(oov),
                "oov_codepoints": [f"U+{ord(character):04X}" for character in oov],
            },
        }

    def _load_crop(self, row: dict[str, Any]) -> tuple[Image.Image, str]:
        materialized = _evaluation_child(
            self.root, row["output_image"], require_file=False
        )
        if self.image_mode in ("materialized", "auto") and materialized.is_file():
            expected = row.get("output_image_sha256")
            if not _is_sha256(expected):
                raise TorchOCREvaluationDataError(
                    f"materialized crop {row['record_id']} lacks SHA-256"
                )
            self._verify_image(materialized, expected, row["record_id"])
            return _evaluation_decode_image(materialized), "materialized"
        if self.image_mode == "materialized":
            raise TorchOCREvaluationDataError(
                f"materialized crop is missing for {row['record_id']}"
            )
        if self.source_root is None:
            raise TorchOCREvaluationDataError(
                f"source_root is required for lazy crop {row['record_id']}"
            )
        source = _evaluation_child(self.source_root, row["source_image"])
        self._verify_image(source, row["source_image_sha256"], row["record_id"])
        decoded = _evaluation_decode_image(source)
        box = row["crop_box"]
        left, top, right, bottom = (
            box["left"], box["top"], box["right"], box["bottom"]
        )
        if left < 0 or top < 0 or right > decoded.width or bottom > decoded.height:
            decoded.close()
            raise TorchOCREvaluationDataError(
                f"lazy crop is outside source image for {row['record_id']}"
            )
        try:
            crop = decoded.crop((left, top, right, bottom)).convert("RGB")
        finally:
            decoded.close()
        return crop, "lazy"

    def _verify_image(self, path: Path, expected: str, record_id: str) -> None:
        if not self.verify_image_hashes:
            return
        stat = path.stat()
        identity = (expected, int(stat.st_size), int(stat.st_mtime_ns))
        if self._verified_images.get(str(path)) == identity:
            return
        actual = _sha256_file(path)
        if actual != expected:
            raise TorchOCREvaluationDataError(
                f"image SHA-256 mismatch for {record_id}: expected {expected}, got {actual}"
            )
        self._verified_images[str(path)] = identity


def ocr_evaluation_collate(
    batch: Sequence[dict[str, Any]], *, pad_value: float = 1.0
) -> dict[str, Any]:
    """Collate every evaluation row while retaining unencodable references."""

    torch = _require_torch()
    if not batch:
        raise TorchOCREvaluationDataError("cannot collate an empty evaluation batch")
    if isinstance(pad_value, bool) or not isinstance(pad_value, (int, float)) or not math.isfinite(float(pad_value)):
        raise TorchOCREvaluationDataError("evaluation pad_value must be finite")
    images = [item.get("image") for item in batch]
    if any(not torch.is_tensor(image) or image.ndim != 3 for image in images):
        raise TorchOCREvaluationDataError("evaluation rows lack [C,H,W] image tensors")
    channels, height = int(images[0].shape[0]), int(images[0].shape[1])
    if any(int(image.shape[0]) != channels or int(image.shape[1]) != height for image in images):
        raise TorchOCREvaluationDataError("evaluation image channels/heights differ")
    widths = [int(image.shape[2]) for image in images]
    padded = images[0].new_full(
        (len(images), channels, height, max(widths)), float(pad_value)
    )
    for index, (image, width) in enumerate(zip(images, widths, strict=True)):
        padded[index, :, :, :width] = image
    eligibility = [item.get("eligible") is True for item in batch]
    targets: list[Any] = []
    for item, eligible in zip(batch, eligibility, strict=True):
        target = item.get("target")
        if not torch.is_tensor(target) or target.ndim != 1:
            raise TorchOCREvaluationDataError("evaluation row target is invalid")
        if eligible and target.numel() <= 0:
            raise TorchOCREvaluationDataError("eligible evaluation target is empty")
        if not eligible and target.numel() != 0:
            raise TorchOCREvaluationDataError("OOV evaluation target must remain unencoded")
        targets.append(target.to(dtype=torch.long))
    return {
        "evaluation_contract": TorchOCREvaluationDataset.contract,
        "images": padded,
        "image_widths": torch.tensor(widths, dtype=torch.long),
        "targets_by_sample": targets,
        "eligibility": eligibility,
        "oov_characters_by_sample": [list(item["oov_characters"]) for item in batch],
        "texts": [item["text"] for item in batch],
        "record_ids": [item["record_id"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


class TinyOCRDetector:
    """Lazy constructor returning an ordinary ``torch.nn.Module`` Tiny U-Net."""

    def __new__(
        cls, in_channels: int = 3, base_channels: int = 8
    ) -> Any:
        return _detector_implementation()(
            in_channels=in_channels, base_channels=base_channels
        )


class TinyCRNN:
    """Lazy constructor returning a dynamic-width CNN/BiLSTM CTC model."""

    def __new__(
        cls,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 16,
        hidden_size: int = 64,
        lstm_layers: int = 1,
    ) -> Any:
        return _recognizer_implementation()(
            num_classes=num_classes,
            in_channels=in_channels,
            base_channels=base_channels,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
        )


def build_detector(*, in_channels: int = 3, base_channels: int = 8) -> Any:
    return TinyOCRDetector(in_channels=in_channels, base_channels=base_channels)


def build_recognizer(
    charset: CTCCharset | Sequence[str] | int,
    *,
    in_channels: int = 1,
    base_channels: int = 16,
    hidden_size: int = 64,
    lstm_layers: int = 1,
) -> Any:
    if isinstance(charset, CTCCharset):
        num_classes = charset.num_classes
    elif type(charset) is int:
        num_classes = charset
    else:
        num_classes = CTCCharset(tuple(charset)).num_classes
    return TinyCRNN(
        num_classes=num_classes,
        in_channels=in_channels,
        base_channels=base_channels,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
    )


def _detector_implementation() -> type[Any]:
    global _DETECTOR_CLASS
    if _DETECTOR_CLASS is not None:
        return _DETECTOR_CLASS
    torch = _require_torch()
    nn = torch.nn
    functional = torch.nn.functional

    class _Block(nn.Module):
        def __init__(self, incoming: int, outgoing: int) -> None:
            super().__init__()
            groups = max(1, math.gcd(4, outgoing))
            self.layers = nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.GroupNorm(groups, outgoing),
                nn.ReLU(inplace=True),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                nn.GroupNorm(groups, outgoing),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    class _TinyOCRDetector(nn.Module):
        def __init__(self, in_channels: int = 3, base_channels: int = 8) -> None:
            super().__init__()
            if in_channels <= 0 or base_channels <= 0:
                raise ValueError("OCR detector channel counts must be positive")
            self.encoder1 = _Block(in_channels, base_channels)
            self.encoder2 = _Block(base_channels, base_channels * 2)
            self.bottleneck = _Block(base_channels * 2, base_channels * 4)
            self.decoder2 = _Block(base_channels * 6, base_channels * 2)
            self.decoder1 = _Block(base_channels * 3, base_channels)
            self.head = nn.Conv2d(base_channels, 1, 1)
            self.model_config = {
                "name": "tiny_ocr_unet",
                "in_channels": int(in_channels),
                "base_channels": int(base_channels),
            }

        def forward(self, inputs: Any) -> Any:
            first = self.encoder1(inputs)
            second = self.encoder2(
                functional.max_pool2d(first, 2, 2, ceil_mode=True)
            )
            middle = self.bottleneck(
                functional.max_pool2d(second, 2, 2, ceil_mode=True)
            )
            up_second = functional.interpolate(
                middle, size=second.shape[-2:], mode="bilinear", align_corners=False
            )
            up_second = self.decoder2(torch.cat((up_second, second), dim=1))
            up_first = functional.interpolate(
                up_second, size=first.shape[-2:], mode="bilinear", align_corners=False
            )
            return self.head(self.decoder1(torch.cat((up_first, first), dim=1)))

    _TinyOCRDetector.__name__ = "TinyOCRDetector"
    _TinyOCRDetector.__qualname__ = "TinyOCRDetector"
    _DETECTOR_CLASS = _TinyOCRDetector
    return _DETECTOR_CLASS


def _recognizer_implementation() -> type[Any]:
    global _RECOGNIZER_CLASS
    if _RECOGNIZER_CLASS is not None:
        return _RECOGNIZER_CLASS
    torch = _require_torch()
    nn = torch.nn

    class _TinyCRNN(nn.Module):
        def __init__(
            self,
            num_classes: int,
            in_channels: int = 1,
            base_channels: int = 16,
            hidden_size: int = 64,
            lstm_layers: int = 1,
        ) -> None:
            super().__init__()
            if num_classes <= 1:
                raise ValueError("CRNN num_classes must include blank and characters")
            if min(in_channels, base_channels, hidden_size, lstm_layers) <= 0:
                raise ValueError("CRNN dimensions must be positive")
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2, ceil_mode=True),
                nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2, ceil_mode=True),
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.sequence = nn.LSTM(
                input_size=base_channels * 4,
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True,
            )
            self.classifier = nn.Linear(hidden_size * 2, num_classes)
            self.model_config = {
                "name": "tiny_crnn_bilstm_ctc",
                "num_classes": int(num_classes),
                "in_channels": int(in_channels),
                "base_channels": int(base_channels),
                "hidden_size": int(hidden_size),
                "lstm_layers": int(lstm_layers),
            }

        def forward(self, images: Any, image_widths: Any | None = None) -> Any:
            if image_widths is None:
                features = self.features(images)
                # Collapse only height. Width remains the CTC time dimension.
                sequence = features.mean(dim=2).permute(0, 2, 1).contiguous()
                encoded, _state = self.sequence(sequence)
            else:
                if image_widths.ndim != 1 or int(image_widths.numel()) != int(images.shape[0]):
                    raise ValueError(
                        "CRNN image_widths cardinality differs from image batch"
                    )
                if bool((image_widths <= 0).any()) or bool((image_widths > images.shape[3]).any()):
                    raise ValueError("CRNN image_widths are outside padded image width")
                raw_widths = [int(value) for value in image_widths.detach().to("cpu").tolist()]
                # Run each distinct-width crop through the CNN at its true
                # boundary.  Merely masking a white-padded tensor is not enough:
                # convolution biases in the padding columns can still leak back
                # into the final valid feature cell.
                feature_sequences: list[Any | None] = [None] * len(raw_widths)
                for width in sorted(set(raw_widths)):
                    indices = [
                        index for index, value in enumerate(raw_widths) if value == width
                    ]
                    features = self.features(images[indices, :, :, :width])
                    grouped = features.mean(dim=2).permute(0, 2, 1).contiguous()
                    for group_index, source_index in enumerate(indices):
                        feature_sequences[source_index] = grouped[group_index]
                if any(item is None for item in feature_sequences):
                    raise RuntimeError("CRNN failed to create every width-group feature")
                sequences = [item for item in feature_sequences if item is not None]
                output_lengths = torch.tensor(
                    [int(item.shape[0]) for item in sequences],
                    dtype=torch.long,
                )
                expected_lengths = recognition_output_lengths(image_widths).detach().to("cpu")
                if not bool(torch.equal(output_lengths, expected_lengths)):
                    raise ValueError("CRNN feature widths differ from the certified width transform")
                sequence = nn.utils.rnn.pad_sequence(
                    sequences, batch_first=True, padding_value=0.0
                )
                packed = nn.utils.rnn.pack_padded_sequence(
                    sequence,
                    output_lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_encoded, _state = self.sequence(packed)
                encoded, _unpacked_lengths = nn.utils.rnn.pad_packed_sequence(
                    packed_encoded,
                    batch_first=True,
                    total_length=sequence.shape[1],
                )
            return self.classifier(encoded).permute(1, 0, 2).contiguous()

        def output_lengths(self, image_widths: Any) -> Any:
            return recognition_output_lengths(image_widths)

    _TinyCRNN.__name__ = "TinyCRNN"
    _TinyCRNN.__qualname__ = "TinyCRNN"
    _RECOGNIZER_CLASS = _TinyCRNN
    return _RECOGNIZER_CLASS


def compute_detector_loss(
    logits: Any,
    target: Any,
    ignore: Any | None = None,
    *,
    bce_weight: float = 0.6,
    dice_weight: float = 0.4,
    epsilon: float = 1.0e-6,
) -> dict[str, Any]:
    """Compute ignore-aware BCE + soft Dice without hidden supervision."""

    torch = _require_torch()
    functional = torch.nn.functional
    if not torch.is_tensor(logits) or not torch.is_tensor(target):
        raise TypeError("detector logits and target must be torch tensors")
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("detector logits/target must share shape [N,1,H,W]")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in (bce_weight, dice_weight)):
        raise ValueError("detector loss weights must be finite and non-negative")
    if bce_weight + dice_weight <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("detector loss configuration is invalid")
    target = target.to(device=logits.device, dtype=logits.dtype)
    if bool(((target < 0) | (target > 1)).any()):
        raise ValueError("detector target must be in [0,1]")
    if ignore is None:
        valid = torch.ones_like(target)
    else:
        if not torch.is_tensor(ignore):
            raise TypeError("detector ignore mask must be a torch tensor")
        if ignore.ndim == 3:
            ignore = ignore.unsqueeze(1)
        if ignore.shape != target.shape:
            raise ValueError("detector ignore mask shape differs from target")
        ignore = ignore.to(device=logits.device, dtype=logits.dtype)
        if bool(((ignore < 0) | (ignore > 1)).any()):
            raise ValueError("detector ignore mask must be in [0,1]")
        valid = 1.0 - ignore
    valid_pixels = valid.sum()
    if float(valid_pixels.detach().cpu()) <= 0:
        zero = (logits * 0.0).sum()
        return {"loss": zero, "bce": zero, "dice": zero, "valid_pixels": valid_pixels}
    bce_map = functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    bce = (bce_map * valid).sum() / valid_pixels
    probabilities = torch.sigmoid(logits) * valid
    selected_target = target * valid
    numerator = 2.0 * (probabilities * selected_target).sum(dim=(1, 2, 3)) + epsilon
    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + selected_target.sum(dim=(1, 2, 3))
        + epsilon
    )
    dice = (1.0 - numerator / denominator).mean()
    loss = float(bce_weight) * bce + float(dice_weight) * dice
    return {"loss": loss, "bce": bce, "dice": dice, "valid_pixels": valid_pixels}


def detector_train_step(
    model: Any,
    batch: Mapping[str, Any],
    optimizer: Any,
    *,
    device: str | Any = "cpu",
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    torch = _require_torch()
    images, target, ignore = _detector_batch(batch, torch, device)
    model.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    terms = compute_detector_loss(model(images), target, ignore)
    terms["loss"].backward()
    if max_grad_norm is not None:
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {name: float(value.detach().cpu()) for name, value in terms.items()}


def detector_eval_step(
    model: Any,
    batch: Mapping[str, Any],
    *,
    device: str | Any = "cpu",
    threshold: float = 0.5,
    min_area: int = 4,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    torch = _require_torch()
    images, target, ignore = _detector_batch(batch, torch, device)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(images)
        terms = compute_detector_loss(logits, target, ignore)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    ground_truth = batch.get("boxes")
    ignore_boxes = batch.get("ignore_boxes")
    metrics = detection_box_metrics(
        probabilities,
        ground_truth_boxes=ground_truth,
        target_masks=target.detach().cpu().numpy(),
        ignore_masks=ignore.detach().cpu().numpy(),
        ignore_boxes=ignore_boxes,
        threshold=threshold,
        min_area=min_area,
        iou_threshold=iou_threshold,
    )
    output: dict[str, float | int] = {
        name: float(value.detach().cpu()) for name, value in terms.items()
    }
    output.update(metrics)
    return output


def extract_text_boxes(
    probabilities: Any,
    *,
    threshold: float = 0.5,
    min_area: int = 4,
) -> list[DetectionBox]:
    """Extract deterministic 4-connected components as exclusive ``xyxy`` boxes."""

    threshold = _probability(threshold, "threshold")
    if type(min_area) is not int or min_area <= 0:
        raise ValueError("min_area must be a positive integer")
    if hasattr(probabilities, "detach"):
        probabilities = probabilities.detach().cpu().numpy()
    array = np.asarray(probabilities, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("text-box extraction expects one 2-D probability map")
    if not np.isfinite(array).all():
        raise ValueError("probability map contains non-finite values")
    mask = array >= threshold
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    boxes: list[DetectionBox] = []
    for start_y, start_x in zip(*np.nonzero(mask), strict=True):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        area = 0
        score_sum = 0.0
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        while stack:
            y, x = stack.pop()
            area += 1
            score_sum += float(array[y, x])
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for neighbour_y, neighbour_x in (
                (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)
            ):
                if (
                    0 <= neighbour_y < height
                    and 0 <= neighbour_x < width
                    and mask[neighbour_y, neighbour_x]
                    and not visited[neighbour_y, neighbour_x]
                ):
                    visited[neighbour_y, neighbour_x] = True
                    stack.append((neighbour_y, neighbour_x))
        if area < min_area:
            continue
        score = score_sum / area
        boxes.append(
            DetectionBox(
                x1=float(min_x), y1=float(min_y),
                x2=float(max_x + 1), y2=float(max_y + 1),
                score=max(0.0, min(1.0, score)), area=area,
            )
        )
    boxes.sort(key=lambda box: (box.y1, box.x1, box.y2, box.x2, -box.score))
    return boxes


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("IoU boxes must contain four coordinates")
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    union = max(0.0, (lx2 - lx1) * (ly2 - ly1)) + max(
        0.0, (rx2 - rx1) * (ry2 - ry1)
    ) - intersection
    return intersection / union if union > 0 else 0.0


def detection_box_metrics(
    probabilities: Any,
    *,
    ground_truth_boxes: Any | None = None,
    target_masks: Any | None = None,
    ignore_masks: Any | None = None,
    ignore_boxes: Any | None = None,
    threshold: float = 0.5,
    min_area: int = 4,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    """One-to-one detector IoU precision/recall/Hmean over a batch."""

    iou_threshold = _probability(iou_threshold, "iou_threshold")
    array = np.asarray(
        probabilities.detach().cpu().numpy()
        if hasattr(probabilities, "detach") else probabilities,
        dtype=np.float32,
    )
    if array.ndim == 3:
        array = array[:, None, :, :]
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError("detector probabilities must have shape [N,1,H,W]")
    target_array = None if target_masks is None else np.asarray(target_masks)
    ignore_array = None if ignore_masks is None else np.asarray(ignore_masks)
    true_positives = false_positives = false_negatives = 0
    matched_ious: list[float] = []
    prediction_count = target_count = ignored_predictions = 0
    for index in range(array.shape[0]):
        predictions = extract_text_boxes(
            array[index, 0], threshold=threshold, min_area=min_area
        )
        local_ignore = _box_rows_at(ignore_boxes, index)
        if ignore_array is not None:
            mask = ignore_array[index]
            while mask.ndim > 2 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 2:
                raise ValueError("ignore masks must resolve to 2-D")
            kept: list[DetectionBox] = []
            for prediction in predictions:
                x1, y1, x2, y2 = (int(value) for value in prediction.xyxy())
                region = mask[y1:y2, x1:x2]
                overlap = float((region != 0).mean()) if region.size else 0.0
                if overlap >= 0.5:
                    ignored_predictions += 1
                else:
                    kept.append(prediction)
            predictions = kept
        elif local_ignore:
            kept = []
            for prediction in predictions:
                area = max(1.0, (prediction.x2 - prediction.x1) * (prediction.y2 - prediction.y1))
                overlap = max(
                    (_intersection_area(prediction.xyxy(), item) / area for item in local_ignore),
                    default=0.0,
                )
                if overlap >= 0.5:
                    ignored_predictions += 1
                else:
                    kept.append(prediction)
            predictions = kept
        truth = _box_rows_at(ground_truth_boxes, index)
        if ground_truth_boxes is None:
            if target_array is None:
                raise ValueError("ground_truth_boxes or target_masks are required")
            target = target_array[index]
            while target.ndim > 2 and target.shape[0] == 1:
                target = target[0]
            truth = [list(box.xyxy()) for box in extract_text_boxes(target, threshold=0.5, min_area=1)]
        prediction_count += len(predictions)
        target_count += len(truth)
        unmatched = set(range(len(truth)))
        # Global score order is deterministic and prevents duplicate matches.
        score_order = sorted(
            range(len(predictions)),
            key=lambda item: (
                -predictions[item].score,
                predictions[item].y1,
                predictions[item].x1,
            ),
        )
        for prediction_index in score_order:
            prediction = predictions[prediction_index]
            candidates = sorted(
                (
                    (box_iou(prediction.xyxy(), truth[item]), item)
                    for item in unmatched
                ),
                key=lambda item: (-item[0], item[1]),
            )
            if candidates and candidates[0][0] >= iou_threshold:
                iou, truth_index = candidates[0]
                unmatched.remove(truth_index)
                true_positives += 1
                matched_ious.append(iou)
            else:
                false_positives += 1
        false_negatives += len(unmatched)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    hmean = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "hmean": hmean,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "predictions": prediction_count,
        "targets": target_count,
        "ignored_predictions": ignored_predictions,
    }


def recognition_output_lengths(image_widths: Any) -> Any:
    """Exact width transform for two ceil-mode 2x2 CRNN pooling layers."""

    torch = _require_torch()
    if not torch.is_tensor(image_widths):
        image_widths = torch.as_tensor(image_widths, dtype=torch.long)
    elif image_widths.dtype == torch.bool or image_widths.dtype.is_floating_point or image_widths.dtype.is_complex:
        raise ValueError("recognition image_widths must use an integer dtype")
    if image_widths.ndim != 1 or bool((image_widths <= 0).any()):
        raise ValueError("recognition image_widths must be positive one-dimensional values")
    values = image_widths.to(dtype=torch.long)
    return torch.div(torch.div(values + 1, 2, rounding_mode="floor") + 1, 2, rounding_mode="floor")


def validate_ctc_batch(
    logits: Any,
    targets: Any,
    target_lengths: Any,
    input_lengths: Any,
    *,
    blank_index: int = 0,
) -> None:
    """Fail before CTC on every cardinality, class, and alignment violation."""

    torch = _require_torch()
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError("recognizer logits must have shape [T,N,C]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("recognizer logits contain non-finite values")
    time_steps, batch_size, classes = map(int, logits.shape)
    if classes <= 1 or blank_index != 0:
        raise ValueError("CTC requires class 0 blank plus at least one character")
    for value, name in (
        (targets, "targets"),
        (target_lengths, "target_lengths"),
        (input_lengths, "input_lengths"),
    ):
        if not torch.is_tensor(value) or value.ndim != 1:
            raise ValueError(f"CTC {name} must be a one-dimensional tensor")
        if value.dtype == torch.bool or value.dtype.is_floating_point or value.dtype.is_complex:
            raise ValueError(f"CTC {name} must use an integer dtype")
    if int(target_lengths.numel()) != batch_size or int(input_lengths.numel()) != batch_size:
        raise ValueError("CTC length cardinality differs from logits batch size")
    target_lengths_cpu = target_lengths.detach().to("cpu", dtype=torch.long)
    input_lengths_cpu = input_lengths.detach().to("cpu", dtype=torch.long)
    targets_cpu = targets.detach().to("cpu", dtype=torch.long)
    if bool((target_lengths_cpu <= 0).any()):
        raise ValueError("CTC targets must be non-empty")
    if int(target_lengths_cpu.sum()) != int(targets_cpu.numel()):
        raise ValueError("sum(target_lengths) differs from flattened target count")
    if bool((input_lengths_cpu <= 0).any()) or bool((input_lengths_cpu > time_steps).any()):
        raise ValueError("CTC input length is outside available model timesteps")
    if bool((targets_cpu <= blank_index).any()) or bool((targets_cpu >= classes).any()):
        raise ValueError("CTC targets contain blank or out-of-range class indices")
    offset = 0
    for index, (target_length, input_length) in enumerate(
        zip(target_lengths_cpu.tolist(), input_lengths_cpu.tolist(), strict=True)
    ):
        sequence = targets_cpu[offset : offset + target_length].tolist()
        offset += target_length
        adjacent_repeats = sum(
            left == right for left, right in zip(sequence, sequence[1:])
        )
        required = target_length + adjacent_repeats
        if input_length < required:
            raise ValueError(
                f"CTC sample {index} needs {required} timesteps for target length "
                f"{target_length} with repeats, but only {input_length} are valid"
            )


def compute_ctc_loss(
    logits: Any,
    targets: Any,
    target_lengths: Any,
    input_lengths: Any,
    *,
    blank_index: int = 0,
) -> Any:
    torch = _require_torch()
    # Filter out samples that don't have enough timesteps
    valid_indices = _get_ctc_valid_indices(torch, targets, target_lengths, input_lengths)
    if len(valid_indices) == 0:
        # Return zero loss if no valid samples
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    if len(valid_indices) < int(target_lengths.shape[0]):
        # Filter to valid samples only.  Targets must be rebuilt from the
        # ORIGINAL per-sample offsets; re-slicing filtered lengths against the
        # unfiltered flat target tensor would silently misalign every sample
        # after the first invalid one.
        valid_set = set(valid_indices)
        original_lengths = target_lengths.detach().cpu().tolist()
        flat_targets = []
        offset = 0
        for index, target_len in enumerate(original_lengths):
            if index in valid_set:
                flat_targets.append(targets[offset:offset + target_len])
            offset += target_len
        targets = torch.cat(flat_targets, dim=0)
        logits = logits[:, valid_indices, :]
        input_lengths = input_lengths[valid_indices]
        target_lengths = target_lengths[valid_indices]
    validate_ctc_batch(
        logits, targets, target_lengths, input_lengths, blank_index=blank_index
    )
    return torch.nn.functional.ctc_loss(
        logits.log_softmax(dim=2),
        targets.to(device=logits.device, dtype=torch.long),
        input_lengths.to(device="cpu", dtype=torch.long),
        target_lengths.to(device="cpu", dtype=torch.long),
        blank=blank_index,
        reduction="mean",
        zero_infinity=False,
    )


def _get_ctc_valid_indices(
    torch: Any,
    targets: Any,
    target_lengths: Any,
    input_lengths: Any,
) -> list[int]:
    """Get indices of samples that have enough CTC timesteps for their targets."""
    batch_size = int(target_lengths.shape[0])
    valid_indices = []

    offset = 0
    for i in range(batch_size):
        target_len = int(target_lengths[i].item())
        sequence = targets[offset:offset + target_len].detach().cpu().tolist()
        offset += target_len
        adjacent_repeats = sum(
            left == right for left, right in zip(sequence, sequence[1:])
        )
        required = target_len + adjacent_repeats
        if int(input_lengths[i].item()) >= required:
            valid_indices.append(i)

    return valid_indices


def _filter_ctc_valid_samples(
    torch: Any,
    targets: Any,
    target_lengths: Any,
    input_lengths: Any,
) -> Any:
    """Filter samples that have enough CTC timesteps for their targets."""
    batch_size = int(target_lengths.shape[0])
    valid_mask = torch.ones(batch_size, dtype=torch.bool, device=target_lengths.device)

    offset = 0
    for i in range(batch_size):
        target_len = int(target_lengths[i].item())
        sequence = targets[offset:offset + target_len].detach().cpu().tolist()
        offset += target_len
        adjacent_repeats = sum(
            left == right for left, right in zip(sequence, sequence[1:])
        )
        required = target_len + adjacent_repeats
        if int(input_lengths[i].item()) < required:
            valid_mask[i] = False

    return valid_mask


def recognizer_train_step(
    model: Any,
    batch: Mapping[str, Any],
    optimizer: Any,
    *,
    device: str | Any = "cpu",
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    torch = _require_torch()
    images, targets, target_lengths, image_widths = _recognition_batch(batch, torch, device)
    model.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images, image_widths.to(device))
    input_lengths = recognition_output_lengths(image_widths.detach().to("cpu"))
    loss = compute_ctc_loss(logits, targets, target_lengths, input_lengths)
    if not bool(torch.isfinite(loss)):
        raise TorchOCRRuntimeError("CTC loss is non-finite")
    loss.backward()
    if max_grad_norm is not None:
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        gradient_norm = float(norm.detach().cpu())
    else:
        gradient_norm = 0.0
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "gradient_norm": gradient_norm,
        "samples": float(images.shape[0]),
        "timesteps": float(input_lengths.sum()),
        "target_codepoints": float(target_lengths.sum()),
    }


def recognizer_eval_step(
    model: Any,
    batch: Mapping[str, Any],
    charset: CTCCharset,
    *,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    if batch.get("evaluation_contract") == TorchOCREvaluationDataset.contract:
        return recognizer_coverage_eval_step(
            model, batch, charset, device=device
        )
    torch = _require_torch()
    images, targets, target_lengths, image_widths = _recognition_batch(batch, torch, device)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(images, image_widths.to(device))
        input_lengths = recognition_output_lengths(image_widths.detach().to("cpu"))
        loss = compute_ctc_loss(logits, targets, target_lengths, input_lengths)
    predictions, _scores = greedy_ctc_decode(logits, input_lengths, charset)
    references = batch.get("texts")
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        references = _decode_flat_targets(targets, target_lengths, charset)
    references = [str(item) for item in references]
    if len(references) != len(predictions):
        raise ValueError("recognition reference cardinality differs from predictions")
    exact = sum(prediction == reference for prediction, reference in zip(predictions, references, strict=True))
    edit_distance = sum(
        levenshtein_distance(prediction, reference)
        for prediction, reference in zip(predictions, references, strict=True)
    )
    reference_codepoints = sum(len(reference) for reference in references)
    return {
        "loss": float(loss.detach().cpu()),
        "strict_exact": exact / max(1, len(references)),
        "cer": edit_distance / max(1, reference_codepoints),
        "exact_count": exact,
        "samples": len(references),
        "edit_distance": edit_distance,
        "reference_codepoints": reference_codepoints,
    }


def recognizer_coverage_eval_step(
    model: Any,
    batch: Mapping[str, Any],
    charset: CTCCharset,
    *,
    device: str | Any = "cpu",
) -> dict[str, Any]:
    """Evaluate all rows while restricting CTC loss to encodable alignments.

    Exact accuracy and CER use every preserved reference.  An OOV reference is
    explicitly forced to an exact-match error.  ``loss`` is computed only over
    train-charset rows whose target also fits the available CTC timesteps;
    ``loss_coverage`` makes that reduced denominator visible.
    """

    torch = _require_torch()
    if batch.get("evaluation_contract") != TorchOCREvaluationDataset.contract:
        raise ValueError("batch does not carry the OCR evaluation contract")
    images = batch.get("images")
    image_widths = batch.get("image_widths")
    targets_by_sample = batch.get("targets_by_sample")
    eligibility = batch.get("eligibility")
    references = batch.get("texts")
    oov_by_sample = batch.get("oov_characters_by_sample")
    if not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("evaluation images must have shape [N,1,H,W]")
    if not torch.is_tensor(image_widths) or image_widths.ndim != 1:
        raise ValueError("evaluation image_widths must be one-dimensional")
    if not isinstance(targets_by_sample, Sequence) or isinstance(targets_by_sample, (str, bytes)):
        raise ValueError("evaluation targets_by_sample must be a sequence")
    if not isinstance(eligibility, Sequence) or isinstance(eligibility, (str, bytes)):
        raise ValueError("evaluation eligibility must be a sequence")
    if not isinstance(references, Sequence) or isinstance(references, (str, bytes)):
        raise ValueError("evaluation references must be a sequence")
    if not isinstance(oov_by_sample, Sequence) or isinstance(oov_by_sample, (str, bytes)):
        raise ValueError("evaluation OOV evidence must be a sequence")
    sample_count = int(images.shape[0])
    if any(
        len(value) != sample_count
        for value in (targets_by_sample, eligibility, references, oov_by_sample)
    ) or int(image_widths.numel()) != sample_count:
        raise ValueError("evaluation batch cardinalities differ")
    if any(type(value) is not bool for value in eligibility):
        raise ValueError("evaluation eligibility values must be booleans")
    if any(not isinstance(value, str) or not value for value in references):
        raise ValueError("evaluation references must be non-empty raw strings")
    eligible_flags = list(eligibility)
    reference_texts = list(references)
    charset_set = set(charset.characters)
    for index, (eligible, target, oov) in enumerate(
        zip(eligible_flags, targets_by_sample, oov_by_sample, strict=True)
    ):
        if not torch.is_tensor(target) or target.ndim != 1:
            raise ValueError(f"evaluation target {index} is invalid")
        if not isinstance(oov, Sequence) or isinstance(oov, (str, bytes)):
            raise ValueError(f"evaluation OOV evidence {index} is invalid")
        if any(not isinstance(character, str) or len(character) != 1 for character in oov):
            raise ValueError(f"evaluation OOV evidence {index} contains invalid characters")
        actual_oov = sorted(set(reference_texts[index]) - charset_set, key=ord)
        if list(oov) != actual_oov:
            raise ValueError(f"evaluation OOV evidence {index} differs from raw reference")
        if eligible != (not actual_oov):
            raise ValueError(f"evaluation eligibility {index} differs from raw reference")
        if eligible:
            expected = list(charset.encode(reference_texts[index]))
            if target.detach().cpu().tolist() != expected:
                raise ValueError(f"evaluation target {index} differs from raw reference")
        elif target.numel() != 0:
            raise ValueError("OOV references must never be partially encoded")

    model.to(device)
    model.eval()
    widths_cpu = image_widths.to(dtype=torch.long, device="cpu")
    with torch.inference_mode():
        logits = model(
            images.to(device=device, dtype=torch.float32),
            image_widths.to(device=device, dtype=torch.long),
        )
    input_lengths = recognition_output_lengths(widths_cpu)
    predictions, _scores = greedy_ctc_decode(logits, input_lengths, charset)

    # OOV exact matches are errors by contract even if an invalid external
    # model somehow emitted a character outside this classifier's vocabulary.
    exact_flags = [
        eligible and prediction == reference
        for eligible, prediction, reference in zip(
            eligible_flags, predictions, reference_texts, strict=True
        )
    ]
    edit_distances = [
        levenshtein_distance(prediction, reference)
        for prediction, reference in zip(predictions, reference_texts, strict=True)
    ]
    eligible_indices = [index for index, value in enumerate(eligible_flags) if value]
    oov_indices = [index for index, value in enumerate(eligible_flags) if not value]
    eligible_exact = sum(exact_flags[index] for index in eligible_indices)
    eligible_edits = sum(edit_distances[index] for index in eligible_indices)
    eligible_codepoints = sum(len(reference_texts[index]) for index in eligible_indices)
    total_codepoints = sum(len(reference) for reference in reference_texts)

    loss_indices: list[int] = []
    ctc_ineligible = 0
    for index in eligible_indices:
        values = targets_by_sample[index].detach().cpu().tolist()
        minimum_steps = len(values) + sum(
            left == right for left, right in zip(values, values[1:])
        )
        if int(input_lengths[index]) >= minimum_steps:
            loss_indices.append(index)
        else:
            ctc_ineligible += 1
    if loss_indices:
        loss_logits = logits[:, loss_indices, :]
        loss_targets = torch.cat(
            [targets_by_sample[index].to(dtype=torch.long) for index in loss_indices],
            dim=0,
        )
        loss_target_lengths = torch.tensor(
            [int(targets_by_sample[index].numel()) for index in loss_indices],
            dtype=torch.long,
        )
        loss_input_lengths = input_lengths[loss_indices]
        loss_value: float | None = float(
            compute_ctc_loss(
                loss_logits,
                loss_targets,
                loss_target_lengths,
                loss_input_lengths,
            ).detach().cpu()
        )
    else:
        loss_value = None
    oov_occurrences = sum(
        sum(character not in charset_set for character in reference_texts[index])
        for index in oov_indices
    )
    unique_oov = sorted(
        {
            character
            for index in oov_indices
            for character in reference_texts[index]
            if character not in charset_set
        },
        key=ord,
    )
    return {
        "loss": loss_value,
        "strict_exact": sum(exact_flags) / sample_count,
        "strict_exact_eligible": (
            eligible_exact / len(eligible_indices) if eligible_indices else None
        ),
        "cer": sum(edit_distances) / max(1, total_codepoints),
        "cer_eligible": (
            eligible_edits / max(1, eligible_codepoints)
            if eligible_indices else None
        ),
        "exact_count": sum(exact_flags),
        "samples": sample_count,
        "eligible": len(eligible_indices),
        "oov_samples": len(oov_indices),
        "oov_characters": oov_occurrences,
        "oov_unique_characters": len(unique_oov),
        "oov_character_values": unique_oov,
        "oov_codepoints": [f"U+{ord(character):04X}" for character in unique_oov],
        "coverage": len(eligible_indices) / sample_count,
        "loss_eligible": len(loss_indices),
        "loss_ineligible_ctc": ctc_ineligible,
        "loss_coverage": len(loss_indices) / sample_count,
        "loss_coverage_eligible": (
            len(loss_indices) / len(eligible_indices) if eligible_indices else 0.0
        ),
        "edit_distance": sum(edit_distances),
        "reference_codepoints": total_codepoints,
        "eligible_reference_codepoints": eligible_codepoints,
        "charset_source": "train_only",
    }


def greedy_ctc_decode(
    logits: Any,
    input_lengths: Any,
    charset: CTCCharset,
) -> tuple[list[str], list[float]]:
    torch = _require_torch()
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError("recognizer logits must have shape [T,N,C]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("recognizer logits contain non-finite values")
    if int(logits.shape[2]) != charset.num_classes:
        raise ValueError("recognizer class count differs from charset")
    if not torch.is_tensor(input_lengths):
        input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    if input_lengths.ndim != 1 or int(input_lengths.numel()) != int(logits.shape[1]):
        raise ValueError("decode input_lengths cardinality differs from logits")
    if bool((input_lengths <= 0).any()) or bool((input_lengths > logits.shape[0]).any()):
        raise ValueError("decode input length exceeds available timesteps")
    probabilities = logits.softmax(dim=2).detach().to("cpu")
    maximum, indices = probabilities.max(dim=2)
    outputs: list[str] = []
    scores: list[float] = []
    for batch_index, raw_length in enumerate(input_lengths.detach().cpu().tolist()):
        sequence = indices[:raw_length, batch_index].tolist()
        confidence = maximum[:raw_length, batch_index].tolist()
        emitted: list[int] = []
        emitted_confidence: list[float] = []
        previous: int | None = None
        for class_index, class_score in zip(sequence, confidence, strict=True):
            if class_index != charset.blank_index and class_index != previous:
                emitted.append(int(class_index))
                emitted_confidence.append(float(class_score))
            previous = int(class_index)
        outputs.append(charset.decode(emitted))
        scores.append(
            float(sum(emitted_confidence) / len(emitted_confidence))
            if emitted_confidence else 0.0
        )
    return outputs, scores


def ctc_beam_decode(
    logits: Any,
    input_lengths: Any,
    charset: CTCCharset,
    *,
    candidate_count: int = 3,
    beam_width: int = 5,
    top_classes: int = 8,
) -> list[list[str]]:
    """Return deterministic non-empty CTC prefix-beam candidates.

    The greedy result remains the authoritative raw_text. This helper is
    an opt-in recovery channel for ambiguous OCR crops: it keeps a small
    prefix beam and only expands the top local classes at each timestep. The
    bounded local expansion is intentional because the production charset is
    large and candidate generation must not turn into an unbounded decoder.
    """

    torch = _require_torch()
    if not torch.is_tensor(logits) or logits.ndim != 3:
        raise ValueError("recognizer logits must have shape [T,N,C]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("recognizer logits contain non-finite values")
    if int(logits.shape[2]) != charset.num_classes:
        raise ValueError("recognizer class count differs from charset")
    if not torch.is_tensor(input_lengths):
        input_lengths = torch.as_tensor(input_lengths, dtype=torch.long)
    if input_lengths.ndim != 1 or int(input_lengths.numel()) != int(logits.shape[1]):
        raise ValueError("decode input_lengths cardinality differs from logits")
    if bool((input_lengths <= 0).any()) or bool((input_lengths > logits.shape[0]).any()):
        raise ValueError("decode input length exceeds available timesteps")
    if (
        isinstance(candidate_count, bool)
        or type(candidate_count) is not int
        or candidate_count < 0
    ):
        raise ValueError("candidate_count must be a non-negative integer")
    if type(beam_width) is not int or beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if type(top_classes) is not int or top_classes <= 0:
        raise ValueError("top_classes must be a positive integer")
    if candidate_count == 0:
        return [[] for _ in range(int(logits.shape[1]))]

    def logaddexp(left: float, right: float) -> float:
        if left == -math.inf:
            return right
        if right == -math.inf:
            return left
        if left < right:
            left, right = right, left
        return left + math.log1p(math.exp(right - left))

    def add_state(
        target: dict[tuple[int, ...], list[float]],
        prefix: tuple[int, ...],
        blank_score: float = -math.inf,
        nonblank_score: float = -math.inf,
    ) -> None:
        state = target.setdefault(prefix, [-math.inf, -math.inf])
        state[0] = logaddexp(state[0], blank_score)
        state[1] = logaddexp(state[1], nonblank_score)

    log_probabilities = logits.log_softmax(dim=2).detach().to("cpu")
    lengths = input_lengths.detach().to("cpu").tolist()
    outputs: list[list[str]] = []
    nonblank_top_k = min(top_classes, charset.num_classes - 1)
    for batch_index, raw_length in enumerate(lengths):
        per_step = log_probabilities[: int(raw_length), batch_index]
        values, indices = torch.topk(per_step, k=nonblank_top_k, dim=1)
        beams: dict[tuple[int, ...], tuple[float, float]] = {
            (): (0.0, -math.inf)
        }
        for timestep in range(int(raw_length)):
            local: list[tuple[int, float]] = [
                (int(class_index), float(class_score))
                for class_score, class_index in zip(
                    values[timestep].tolist(),
                    indices[timestep].tolist(),
                    strict=True,
                )
            ]
            local.sort(key=lambda item: (-item[1], item[0]))
            blank_score = float(per_step[timestep, charset.blank_index])
            if all(class_index != charset.blank_index for class_index, _ in local):
                local.append((charset.blank_index, blank_score))

            expanded: dict[tuple[int, ...], list[float]] = {}
            for prefix, (blank_prefix, nonblank_prefix) in beams.items():
                prefix_score = logaddexp(blank_prefix, nonblank_prefix)
                add_state(expanded, prefix, blank_score=prefix_score + blank_score)
                last_class = prefix[-1] if prefix else None
                for class_index, class_score in local:
                    if class_index == charset.blank_index:
                        continue
                    if class_index == last_class:
                        # A repeated class can either remain the same CTC
                        # prefix (when it follows a non-blank) or append a
                        # new character (when separated by a blank).
                        add_state(
                            expanded,
                            prefix,
                            nonblank_score=nonblank_prefix + class_score,
                        )
                        add_state(
                            expanded,
                            prefix + (class_index,),
                            nonblank_score=blank_prefix + class_score,
                        )
                    else:
                        add_state(
                            expanded,
                            prefix + (class_index,),
                            nonblank_score=prefix_score + class_score,
                        )
            ranked = sorted(
                expanded.items(),
                key=lambda item: (
                    -logaddexp(item[1][0], item[1][1]),
                    item[0],
                ),
            )[:beam_width]
            beams = {
                prefix: (state[0], state[1])
                for prefix, state in ranked
            }

        ranked = sorted(
            beams.items(),
            key=lambda item: (
                -logaddexp(item[1][0], item[1][1]),
                item[0],
            ),
        )
        seen: set[str] = set()
        candidates: list[str] = []
        for prefix, _state in ranked:
            text = charset.decode(prefix)
            if not text or text in seen:
                continue
            seen.add(text)
            candidates.append(text)
            if len(candidates) >= candidate_count:
                break
        outputs.append(candidates)
    return outputs


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def default_ocr_preprocess() -> dict[str, Any]:
    return {
        "detector_size": [512, 512],
        "detector_colour": "RGB",
        "detector_scale": "uint8_to_0_1",
        "detector_threshold": 0.5,
        "detector_min_area": 4,
        "recognition_height": 48,
        "recognition_colour": "L",
        "recognition_scale": "uint8_to_0_1",
        "recognition_pad_value": 1.0,
        "crop_padding": 2,
    }


def save_ocr_bundle(
    path: str | Path,
    detector: Any,
    recognizer: Any,
    *,
    charset: CTCCharset,
    preprocess: Mapping[str, Any],
    lineage: Mapping[str, Any],
    detector_optimizer: Any | None = None,
    recognizer_optimizer: Any | None = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Mapping[str, Any] | None = None,
) -> OCRBundleInfo:
    torch = _require_torch()
    if type(epoch) is not int or epoch < 0 or type(global_step) is not int or global_step < 0:
        raise ValueError("OCR bundle epoch/global_step must be non-negative integers")
    detector_spec = _model_spec(detector, "tiny_ocr_unet", "detector")
    recognizer_spec = _model_spec(recognizer, "tiny_crnn_bilstm_ctc", "recognizer")
    if recognizer_spec["num_classes"] != charset.num_classes:
        raise OCRBundleError("recognizer class count differs from charset")
    preprocess_copy = _validate_preprocess(preprocess)
    lineage_copy = _json_mapping(lineage, "lineage", require_nonempty=True)
    metrics_copy = _json_mapping(metrics or {}, "metrics", require_nonempty=False)
    document = {
        "schema": BUNDLE_SCHEMA,
        "detector": {
            "model": detector_spec,
            "state_dict": detector.state_dict(),
            "optimizer_state_dict": detector_optimizer.state_dict() if detector_optimizer is not None else None,
        },
        "recognizer": {
            "model": recognizer_spec,
            "state_dict": recognizer.state_dict(),
            "optimizer_state_dict": recognizer_optimizer.state_dict() if recognizer_optimizer is not None else None,
        },
        "charset": {
            "blank_index": charset.blank_index,
            "characters": list(charset.characters),
            "sha256": hashlib.sha256("".join(charset.characters).encode("utf-8")).hexdigest(),
        },
        "preprocess": preprocess_copy,
        "preprocess_sha256": _canonical_sha256(preprocess_copy),
        "lineage": lineage_copy,
        "lineage_sha256": _canonical_sha256(lineage_copy),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics_copy,
        "torch_version": str(torch.__version__),
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(document, temporary)
        if temporary.stat().st_size <= 0:
            raise OCRBundleError("torch produced an empty OCR bundle")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = _sha256_file(destination)
    return OCRBundleInfo(
        path=str(destination), sha256=digest, bytes=destination.stat().st_size,
        version=f"tiny-ocr-{digest[:16]}", epoch=epoch, global_step=global_step,
        lineage=lineage_copy,
    )


def load_ocr_bundle(
    path: str | Path,
    *,
    device: str | Any = "cpu",
    expected_sha256: str | None = None,
    expected_lineage: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> LoadedOCRBundle:
    torch = _require_torch()
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise OCRBundleError(f"OCR bundle is missing or empty: {source}")
    digest = _sha256_file(source)
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise OCRBundleError("expected OCR bundle SHA-256 is invalid")
        if digest != expected_sha256:
            raise OCRBundleError(
                f"OCR bundle SHA-256 mismatch: expected {expected_sha256}, got {digest}"
            )
    try:
        try:
            document = torch.load(source, map_location=device, weights_only=False)
        except TypeError:
            document = torch.load(source, map_location=device)
    except Exception as exc:
        raise OCRBundleError(
            f"cannot deserialize OCR bundle {source}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(document, dict) or document.get("schema") != BUNDLE_SCHEMA:
        raise OCRBundleError("OCR bundle schema is missing or incompatible")
    lineage = _json_mapping(document.get("lineage"), "lineage", require_nonempty=True)
    if document.get("lineage_sha256") != _canonical_sha256(lineage):
        raise OCRBundleError("OCR bundle lineage digest mismatch")
    if expected_lineage is not None:
        expected = _json_mapping(expected_lineage, "expected_lineage", require_nonempty=True)
        if expected != lineage:
            raise OCRBundleError("OCR bundle lineage differs from requested frozen lineage")
    preprocess = _validate_preprocess(document.get("preprocess"))
    if document.get("preprocess_sha256") != _canonical_sha256(preprocess):
        raise OCRBundleError("OCR bundle preprocess digest mismatch")
    charset_payload = document.get("charset")
    if not isinstance(charset_payload, dict) or charset_payload.get("blank_index") != 0:
        raise OCRBundleError("OCR bundle charset is invalid")
    characters = charset_payload.get("characters")
    if not isinstance(characters, list):
        raise OCRBundleError("OCR bundle charset characters are missing")
    try:
        charset = CTCCharset(tuple(characters))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise OCRBundleError(f"OCR bundle charset is invalid: {exc}") from exc
    expected_charset_hash = hashlib.sha256("".join(charset.characters).encode("utf-8")).hexdigest()
    if charset_payload.get("sha256") != expected_charset_hash:
        raise OCRBundleError("OCR bundle charset digest mismatch")
    detector_payload = document.get("detector")
    recognizer_payload = document.get("recognizer")
    if not isinstance(detector_payload, dict) or not isinstance(recognizer_payload, dict):
        raise OCRBundleError("OCR bundle detector/recognizer payload is missing")
    detector_spec = detector_payload.get("model")
    recognizer_spec = recognizer_payload.get("model")
    if not isinstance(detector_spec, dict) or detector_spec.get("name") != "tiny_ocr_unet":
        raise OCRBundleError("OCR detector specification is invalid")
    if not isinstance(recognizer_spec, dict) or recognizer_spec.get("name") != "tiny_crnn_bilstm_ctc":
        raise OCRBundleError("OCR recognizer specification is invalid")
    try:
        detector = build_detector(
            in_channels=int(detector_spec["in_channels"]),
            base_channels=int(detector_spec["base_channels"]),
        )
        recognizer = build_recognizer(
            int(recognizer_spec["num_classes"]),
            in_channels=int(recognizer_spec["in_channels"]),
            base_channels=int(recognizer_spec["base_channels"]),
            hidden_size=int(recognizer_spec["hidden_size"]),
            lstm_layers=int(recognizer_spec["lstm_layers"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OCRBundleError("OCR bundle model dimensions are invalid") from exc
    if recognizer.model_config["num_classes"] != charset.num_classes:
        raise OCRBundleError("OCR bundle recognizer class count differs from charset")
    for model, payload, label in (
        (detector, detector_payload, "detector"),
        (recognizer, recognizer_payload, "recognizer"),
    ):
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise OCRBundleError(f"OCR {label} state_dict is missing")
        try:
            model.load_state_dict(state, strict=strict)
            model.to(device)
            model.eval()
        except Exception as exc:
            raise OCRBundleError(
                f"OCR {label} weights are incompatible: {type(exc).__name__}: {exc}"
            ) from exc
    epoch = document.get("epoch")
    global_step = document.get("global_step")
    if type(epoch) is not int or epoch < 0 or type(global_step) is not int or global_step < 0:
        raise OCRBundleError("OCR bundle epoch/global_step are invalid")
    metrics = _json_mapping(document.get("metrics", {}), "metrics", require_nonempty=False)
    return LoadedOCRBundle(
        detector=detector, recognizer=recognizer, charset=charset,
        preprocess=preprocess, lineage=lineage, metrics=metrics,
        epoch=epoch, global_step=global_step, checkpoint_sha256=digest,
        version=f"tiny-ocr-{digest[:16]}",
        detector_optimizer_state=detector_payload.get("optimizer_state_dict"),
        recognizer_optimizer_state=recognizer_payload.get("optimizer_state_dict"),
        path=source,
    )


def export_ocr_submodel(
    model: Any,
    path: str | Path,
    *,
    kind: Literal["detector", "recognizer"],
    format: Literal["torchscript", "onnx"] | None = None,
    input_shape: Sequence[int] | None = None,
    device: str | Any = "cpu",
    tolerance: float | None = None,
    opset_version: int = 17,
) -> OCRExportReport:
    """Export exactly one OCR submodel and compare its native runtime output."""

    torch = _require_torch()
    if kind not in ("detector", "recognizer"):
        raise ValueError("kind must be 'detector' or 'recognizer'")
    destination = Path(path).expanduser().resolve()
    resolved_format = _resolve_export_format(destination, format)
    default_shape = (1, 3, 64, 64) if kind == "detector" else (1, 1, 48, 80)
    shape = tuple(int(value) for value in (input_shape or default_shape))
    expected_channels = 3 if kind == "detector" else 1
    if len(shape) != 4 or any(value <= 0 for value in shape) or shape[1] != expected_channels:
        raise ValueError(f"{kind} input_shape must be positive NCHW with {expected_channels} channel(s)")
    maximum_error = float(tolerance) if tolerance is not None else (
        1.0e-5 if resolved_format == "torchscript" else 5.0e-4
    )
    if not math.isfinite(maximum_error) or maximum_error < 0:
        raise ValueError("export tolerance must be finite and non-negative")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = (
        tempfile.gettempdir()
        if resolved_format == "torchscript"
        else destination.parent
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=f".{resolved_format}.tmp", dir=temporary_directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")
    original_training = bool(model.training)
    count = int(np.prod(shape))
    example = torch.linspace(0.0, 1.0, steps=count, dtype=torch.float32, device=device).reshape(shape)
    try:
        model.to(device)
        model.eval()
        with torch.inference_mode():
            reference = _tensor_output(model(example), torch).detach().cpu().numpy()
        if resolved_format == "torchscript":
            traced = torch.jit.trace(model, example, strict=True)
            traced.save(str(temporary))
        else:
            dynamic_axes = {
                "image": {0: "batch", 2: "height", 3: "width"},
                "logits": ({0: "time", 1: "batch"} if kind == "recognizer" else {0: "batch", 2: "height", 3: "width"}),
            }
            kwargs = {
                "export_params": True,
                "opset_version": opset_version,
                "do_constant_folding": True,
                "input_names": ["image"],
                "output_names": ["logits"],
                "dynamic_axes": dynamic_axes,
            }
            try:
                torch.onnx.export(model, example, str(temporary), dynamo=False, **kwargs)
            except TypeError:
                torch.onnx.export(model, example, str(temporary), **kwargs)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise OCRExportError("OCR export produced no artifact bytes")
        candidate = _run_exported(temporary, resolved_format, example)
        if candidate.shape != reference.shape:
            raise OCRExportError(
                f"OCR export output shape {candidate.shape} differs from eager {reference.shape}"
            )
        difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        max_abs = float(difference.max(initial=0.0))
        mean_abs = float(difference.mean()) if difference.size else 0.0
        if not math.isfinite(max_abs) or max_abs > maximum_error:
            raise OCRExportError(
                f"OCR {kind} {resolved_format} max_abs_error {max_abs:.8g} exceeds {maximum_error:.8g}"
            )
        try:
            os.replace(temporary, destination)
        except OSError:
            # torch.jit.save cannot write non-ASCII (e.g. Chinese) paths on
            # Windows; stage in the ASCII %TEMP% directory then copy.
            shutil.copy2(temporary, destination)
            temporary.unlink(missing_ok=True)
    except OCRExportError:
        raise
    except Exception as exc:
        raise OCRExportError(
            f"cannot export OCR {kind} as {resolved_format}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
        model.to(original_device)
        model.train(original_training)
    digest = _sha256_file(destination)
    return OCRExportReport(
        schema=EXPORT_SCHEMA, path=str(destination), kind=kind,
        format=resolved_format, artifact_sha256=digest,
        artifact_bytes=destination.stat().st_size, input_shape=shape,
        output_shape=tuple(int(value) for value in reference.shape),
        max_abs_error=max_abs, mean_abs_error=mean_abs,
        tolerance=maximum_error, passed=True,
    )


def export_ocr_models(
    detector: Any,
    recognizer: Any,
    *,
    detector_path: str | Path,
    recognizer_path: str | Path,
    format: Literal["torchscript", "onnx"] | None = None,
    device: str | Any = "cpu",
) -> dict[str, OCRExportReport]:
    return {
        "detector": export_ocr_submodel(
            detector, detector_path, kind="detector", format=format, device=device
        ),
        "recognizer": export_ocr_submodel(
            recognizer, recognizer_path, kind="recognizer", format=format, device=device
        ),
    }


def verify_ocr_export(
    model: Any,
    path: str | Path,
    *,
    format: Literal["torchscript", "onnx"],
    inputs: Any,
    tolerance: float | None = None,
) -> dict[str, float | bool]:
    torch = _require_torch()
    maximum_error = float(tolerance) if tolerance is not None else (
        1.0e-5 if format == "torchscript" else 5.0e-4
    )
    if not math.isfinite(maximum_error) or maximum_error < 0:
        raise ValueError("export tolerance must be finite and non-negative")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    with torch.inference_mode():
        reference = _tensor_output(model(inputs.to(device)), torch).detach().cpu().numpy()
    resolved_path = Path(path).resolve()
    if format == "torchscript":
        # torch.jit.load cannot open non-ASCII (e.g. Chinese) paths on Windows.
        descriptor, staged_name = tempfile.mkstemp(
            prefix=".ocr-verify-", suffix=".pt", dir=tempfile.gettempdir()
        )
        os.close(descriptor)
        staged = Path(staged_name)
        staged.unlink()
        try:
            shutil.copy2(resolved_path, staged)
            candidate = _run_exported(staged, format, inputs)
        finally:
            staged.unlink(missing_ok=True)
    else:
        candidate = _run_exported(resolved_path, format, inputs)
    if candidate.shape != reference.shape:
        raise OCRExportError("OCR export output shape differs from eager output")
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    maximum = float(difference.max(initial=0.0))
    mean = float(difference.mean()) if difference.size else 0.0
    if not math.isfinite(maximum) or maximum > maximum_error:
        raise OCRExportError(
            f"OCR export max_abs_error {maximum:.8g} exceeds {maximum_error:.8g}"
        )
    return {"max_abs_error": maximum, "mean_abs_error": mean, "tolerance": maximum_error, "passed": True}


# Verb-first aliases are retained for callers that group train functions by task.
train_detector_step = detector_train_step
train_recognizer_step = recognizer_train_step
evaluate_detector_step = detector_eval_step
evaluate_recognizer_step = recognizer_eval_step


def _evaluation_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise TorchOCREvaluationDataError(
            f"cannot read strict OCR evaluation JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise TorchOCREvaluationDataError(f"JSON root is not an object: {path}")
    return value


def _evaluation_jsonl(path: Path, identifier: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise TorchOCREvaluationDataError(f"cannot open {path}: {exc}") from exc
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise TorchOCREvaluationDataError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TorchOCREvaluationDataError(
                    f"OCR evaluation row {line_number} is not an object"
                )
            identity = row.get(identifier)
            if not isinstance(identity, str) or not identity or identity in seen:
                raise TorchOCREvaluationDataError(
                    f"OCR evaluation row {line_number} has missing/duplicate {identifier}"
                )
            seen.add(identity)
            rows.append(row)
    return rows


def _verify_evaluation_artifact(
    root: Path, manifest: Mapping[str, Any], relative: str
) -> None:
    artifacts = manifest.get("artifacts")
    expected = artifacts.get(relative) if isinstance(artifacts, Mapping) else None
    if not _is_sha256(expected):
        raise TorchOCREvaluationDataError(
            f"OCR manifest lacks SHA-256 for {relative}"
        )
    path = _evaluation_child(root, relative)
    actual = _sha256_file(path)
    if actual != expected:
        raise TorchOCREvaluationDataError(
            f"committed OCR artifact {relative} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _evaluation_child(
    root: Path, relative: str, *, require_file: bool = True
) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise TorchOCREvaluationDataError(
            f"OCR evaluation path must be relative: {relative!r}"
        )
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TorchOCREvaluationDataError(
            f"OCR evaluation path escapes root: {relative!r}"
        ) from exc
    if require_file and not resolved.is_file():
        raise TorchOCREvaluationDataError(f"OCR evaluation file is missing: {resolved}")
    return resolved


def _evaluation_decode_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as probe:
            if not probe.format:
                raise TorchOCREvaluationDataError(
                    f"image magic did not identify a format: {path}"
                )
            probe.verify()
        with Image.open(path) as opened:
            opened.load()
            return opened.copy()
    except TorchOCREvaluationDataError:
        raise
    except (OSError, ValueError) as exc:
        raise TorchOCREvaluationDataError(f"cannot decode image {path}: {exc}") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _detector_batch(batch: Mapping[str, Any], torch: Any, device: Any) -> tuple[Any, Any, Any]:
    images = batch.get("images", batch.get("image"))
    target = batch.get("target")
    ignore = batch.get("ignore")
    if not torch.is_tensor(images) or not torch.is_tensor(target):
        raise TypeError("detector batch requires images and target tensors")
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if target.ndim == 3:
        target = target.unsqueeze(0)
    if ignore is None:
        ignore = torch.zeros_like(target)
    elif ignore.ndim == 3:
        ignore = ignore.unsqueeze(0)
    return (
        images.to(device=device, dtype=torch.float32),
        target.to(device=device, dtype=torch.float32),
        ignore.to(device=device, dtype=torch.float32),
    )


def _recognition_batch(batch: Mapping[str, Any], torch: Any, device: Any) -> tuple[Any, Any, Any, Any]:
    images = batch.get("images")
    targets = batch.get("targets")
    target_lengths = batch.get("target_lengths")
    image_widths = batch.get("image_widths")
    if any(not torch.is_tensor(item) for item in (images, targets, target_lengths, image_widths)):
        raise TypeError("recognition batch requires images/targets/target_lengths/image_widths tensors")
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("recognition images must have shape [N,1,H,W]")
    if image_widths.ndim != 1 or int(image_widths.numel()) != int(images.shape[0]):
        raise ValueError("recognition image_widths cardinality differs from images")
    if bool((image_widths <= 0).any()) or bool((image_widths > images.shape[3]).any()):
        raise ValueError("recognition image_widths are outside padded tensor width")
    return (
        images.to(device=device, dtype=torch.float32),
        targets.to(dtype=torch.long),
        target_lengths.to(dtype=torch.long),
        image_widths.to(dtype=torch.long),
    )


def _decode_flat_targets(targets: Any, lengths: Any, charset: CTCCharset) -> list[str]:
    values = targets.detach().cpu().tolist()
    output: list[str] = []
    offset = 0
    for length in lengths.detach().cpu().tolist():
        output.append(charset.decode(values[offset : offset + length], remove_blank=True))
        offset += length
    return output


def _box_rows_at(value: Any, index: int) -> list[list[float]]:
    if value is None:
        return []
    row = value[index]
    if hasattr(row, "detach"):
        row = row.detach().cpu().tolist()
    result: list[list[float]] = []
    for item in row:
        values = [float(number) for number in item]
        if len(values) != 4 or any(not math.isfinite(number) for number in values):
            raise ValueError("box rows must contain four finite coordinates")
        result.append(values)
    return result


def _intersection_area(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _model_spec(model: Any, expected_name: str, label: str) -> dict[str, Any]:
    spec = getattr(model, "model_config", None)
    if not isinstance(spec, dict) or spec.get("name") != expected_name:
        raise OCRBundleError(f"OCR bundle requires a serializable Tiny {label} model")
    return _json_mapping(spec, f"{label}.model_config", require_nonempty=True)


def _validate_preprocess(value: Any) -> dict[str, Any]:
    result = _json_mapping(value, "preprocess", require_nonempty=True)
    detector_size = result.get("detector_size")
    if (
        not isinstance(detector_size, list)
        or len(detector_size) != 2
        or any(type(item) is not int or item <= 0 for item in detector_size)
    ):
        raise OCRBundleError("preprocess.detector_size must be [height,width]")
    if result.get("detector_colour") != "RGB" or result.get("detector_scale") != "uint8_to_0_1":
        raise OCRBundleError("unsupported detector preprocessing")
    if result.get("recognition_colour") != "L" or result.get("recognition_scale") != "uint8_to_0_1":
        raise OCRBundleError("unsupported recognizer preprocessing")
    if type(result.get("recognition_height")) is not int or result["recognition_height"] <= 0:
        raise OCRBundleError("preprocess.recognition_height must be positive")
    _probability(result.get("detector_threshold"), "preprocess.detector_threshold")
    if type(result.get("detector_min_area")) is not int or result["detector_min_area"] <= 0:
        raise OCRBundleError("preprocess.detector_min_area must be positive")
    pad = result.get("recognition_pad_value")
    if isinstance(pad, bool) or not isinstance(pad, (int, float)) or not math.isfinite(float(pad)):
        raise OCRBundleError("preprocess.recognition_pad_value must be finite")
    if type(result.get("crop_padding")) is not int or result["crop_padding"] < 0:
        raise OCRBundleError("preprocess.crop_padding must be non-negative")
    return result


def _json_mapping(value: Any, name: str, *, require_nonempty: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise OCRBundleError(f"{name} must be a mapping")
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        result = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OCRBundleError(f"{name} must be finite JSON data: {exc}") from exc
    if not isinstance(result, dict) or (require_nonempty and not result):
        raise OCRBundleError(f"{name} must be a non-empty JSON object")
    return result


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_export_format(path: Path, explicit: str | None) -> Literal["torchscript", "onnx"]:
    if explicit is not None:
        value = explicit.lower()
    else:
        value = "onnx" if path.suffix.lower() == ".onnx" else "torchscript"
    if value not in ("torchscript", "onnx"):
        raise ValueError("export format must be 'torchscript' or 'onnx'")
    return value  # type: ignore[return-value]


def _run_exported(path: Path, format: str, inputs: Any) -> np.ndarray:
    torch = _require_torch()
    if format == "torchscript":
        module = torch.jit.load(str(path), map_location="cpu")
        module.eval()
        with torch.inference_mode():
            return _tensor_output(module(inputs.detach().cpu()), torch).detach().cpu().numpy()
    try:
        runtime = importlib.import_module("onnxruntime")
    except (ImportError, OSError) as exc:
        raise OCRExportError("onnxruntime is required to verify OCR ONNX exports") from exc
    session = runtime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return np.asarray(session.run(None, {name: inputs.detach().cpu().numpy()})[0])


def _tensor_output(value: Any, torch: Any) -> Any:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and torch.is_tensor(value[0]):
        return value[0]
    if isinstance(value, Mapping):
        for key in ("logits", "output"):
            if torch.is_tensor(value.get(key)):
                return value[key]
    raise OCRExportError("OCR model output is not one logits tensor")


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric in [0,1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite in [0,1]")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError) as exc:
        raise TorchOCRRuntimeError(
            "PyTorch is required for the OCR model lifecycle; use the project .venv-train runtime"
        ) from exc


__all__ = [
    "DetectionBox", "LoadedOCRBundle", "OCRBundleError", "OCRBundleInfo",
    "OCRExportError", "OCRExportReport", "TinyCRNN", "TinyOCRDetector",
    "TorchOCREvaluationDataError", "TorchOCREvaluationDataset",
    "TorchOCRRuntimeError", "box_iou", "build_detector", "build_recognizer",
    "compute_ctc_loss", "compute_detector_loss", "default_ocr_preprocess",
    "detection_box_metrics", "detector_eval_step", "detector_train_step",
    "evaluate_detector_step", "evaluate_recognizer_step", "export_ocr_models",
    "export_ocr_submodel", "extract_text_boxes", "ctc_beam_decode", "greedy_ctc_decode",
    "levenshtein_distance", "load_ocr_bundle", "ocr_evaluation_collate",
    "recognition_output_lengths", "recognizer_coverage_eval_step",
    "recognizer_eval_step", "recognizer_train_step", "save_ocr_bundle",
    "train_detector_step", "train_recognizer_step", "validate_ctc_batch",
    "verify_ocr_export",
]
