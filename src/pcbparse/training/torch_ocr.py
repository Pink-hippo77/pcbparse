from __future__ import annotations

import builtins
import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image


VALID_SPLITS = ("train", "val", "test")
VALID_IMAGE_MODES = ("materialized", "lazy", "auto")


class TorchOCRDataError(RuntimeError):
    """Raised when frozen OCR data cannot be consumed without ambiguity."""


def require_torch() -> Any:
    """Import PyTorch only when tensor functionality is actually requested."""

    try:
        return builtins.__import__("torch")
    except (ImportError, ModuleNotFoundError) as exc:
        raise TorchOCRDataError(
            "PyTorch is required for OCR recognition tensors but is not installed. "
            "Install the project training runtime before constructing this dataset."
        ) from exc


@dataclass(frozen=True, slots=True)
class CTCCharset:
    """Stable one-codepoint-to-one-class mapping with CTC blank at index 0."""

    characters: tuple[str, ...]
    blank_index: int = 0

    def __post_init__(self) -> None:
        if self.blank_index != 0:
            raise TorchOCRDataError("CTC blank index is fixed at 0")
        if not self.characters:
            raise TorchOCRDataError("charset must contain at least one character")
        if any(not isinstance(character, str) or len(character) != 1 for character in self.characters):
            raise TorchOCRDataError(
                "each charset entry must be exactly one Unicode code point"
            )
        if len(set(self.characters)) != len(self.characters):
            raise TorchOCRDataError("charset contains duplicate characters")
        if tuple(sorted(self.characters, key=ord)) != self.characters:
            raise TorchOCRDataError(
                "charset is not in stable ascending Unicode-codepoint order"
            )
        if any(character in "\t\r\n" for character in self.characters):
            raise TorchOCRDataError("charset contains a line-control character")

    @classmethod
    def from_file(cls, path: str | Path) -> CTCCharset:
        charset_path = Path(path).expanduser().resolve(strict=True)
        raw = charset_path.read_text(encoding="utf-8")
        if raw and not raw.endswith("\n"):
            raise TorchOCRDataError("charset.txt must end with a newline")
        # The materializer writes exactly one code point followed by LF. A
        # literal space therefore remains " ", not an empty/trimmed record.
        characters = tuple(raw.split("\n")[:-1]) if raw else ()
        return cls(characters=characters)

    @property
    def num_classes(self) -> int:
        return len(self.characters) + 1

    @property
    def char_to_index(self) -> dict[str, int]:
        return {
            character: index
            for index, character in enumerate(self.characters, start=1)
        }

    def encode(self, text: str) -> tuple[int, ...]:
        _validate_raw_text(text, record_id="<encode>")
        mapping = self.char_to_index
        unknown = sorted(set(text) - set(mapping), key=ord)
        if unknown:
            codepoints = ", ".join(f"U+{ord(character):04X}" for character in unknown)
            raise TorchOCRDataError(
                f"text contains {len(unknown)} charset-OOV character(s): {codepoints}"
            )
        indices = tuple(mapping[character] for character in text)
        if any(index == self.blank_index for index in indices):
            raise TorchOCRDataError("encoded target unexpectedly contains CTC blank")
        return indices

    def decode(
        self,
        indices: Iterable[int],
        *,
        collapse_repeats: bool = False,
        remove_blank: bool = True,
    ) -> str:
        result: list[str] = []
        previous: int | None = None
        for raw_index in indices:
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise TorchOCRDataError("CTC indices must be integers")
            index = raw_index
            repeated = collapse_repeats and previous == index
            previous = index
            if repeated:
                continue
            if index == self.blank_index and remove_blank:
                continue
            if index <= self.blank_index or index >= self.num_classes:
                raise TorchOCRDataError(f"CTC class index out of range: {index}")
            result.append(self.characters[index - 1])
        return "".join(result)


class TorchOCRRecognitionDataset:
    """PyTorch-compatible recognition dataset backed by frozen materialization.

    This class intentionally does not inherit from ``torch.utils.data.Dataset``
    so importing this module never imports PyTorch. It implements the map-style
    ``__len__``/``__getitem__`` protocol consumed by ``DataLoader``.

    ``image_mode='materialized'`` reads and verifies frozen recognition PNGs.
    ``image_mode='lazy'`` verifies the original source image and applies the
    exact integer ``crop_box`` recorded by the materializer. ``auto`` prefers a
    valid materialized crop and otherwise requires ``source_root``.
    """

    def __init__(
        self,
        materialization_root: str | Path,
        *,
        split: str,
        source_root: str | Path | None = None,
        image_mode: str = "auto",
        image_height: int = 48,
        verify_image_hashes: bool = True,
    ) -> None:
        # Fail before accepting a dataset that the runtime cannot tensorize.
        require_torch()
        if split not in VALID_SPLITS:
            raise TorchOCRDataError(
                f"split must be one of {VALID_SPLITS}, got {split!r}"
            )
        if image_mode not in VALID_IMAGE_MODES:
            raise TorchOCRDataError(
                f"image_mode must be one of {VALID_IMAGE_MODES}, got {image_mode!r}"
            )
        if type(image_height) is not int or image_height <= 0:
            raise TorchOCRDataError("image_height must be a positive integer")
        if type(verify_image_hashes) is not bool:
            raise TorchOCRDataError("verify_image_hashes must be boolean")

        self.root = Path(materialization_root).expanduser().resolve(strict=True)
        self.split = split
        self.image_mode = image_mode
        self.image_height = image_height
        self.verify_image_hashes = verify_image_hashes
        self.source_root = (
            Path(source_root).expanduser().resolve(strict=True)
            if source_root is not None
            else None
        )

        self.manifest = _read_json_object(self.root / "manifest.json")
        _validate_commit_manifest(self.manifest)
        _verify_committed_artifact(self.root, self.manifest, "recognition_index.jsonl")
        _verify_committed_artifact(self.root, self.manifest, "charset.txt")
        _verify_committed_artifact(self.root, self.manifest, "report.json")

        report = _read_json_object(self.root / "report.json")
        policy = report.get("policy")
        if not isinstance(policy, dict):
            raise TorchOCRDataError("materialization report lacks a policy object")
        if _canonical_sha256(policy) != self.manifest.get("policy_sha256"):
            raise TorchOCRDataError("materialization policy hash mismatch")
        max_length = policy.get("max_text_length")
        if type(max_length) is not int or max_length <= 0:
            raise TorchOCRDataError("materialization policy has invalid max_text_length")
        if policy.get("overlength") != "exclude_not_truncate":
            raise TorchOCRDataError("materialization does not certify no-truncation policy")
        if policy.get("raw_text") != "preserved_without_strip_normalization_or_truncation":
            raise TorchOCRDataError("materialization does not certify strict raw text")
        self.max_text_length = max_length

        self.charset = CTCCharset.from_file(self.root / "charset.txt")
        all_rows = _read_jsonl(self.root / "recognition_index.jsonl", "record_id")
        manifest_counts = self.manifest.get("counts")
        if not isinstance(manifest_counts, dict):
            raise TorchOCRDataError("materialization manifest lacks record counts")
        indexed_count = manifest_counts.get("recognition_index")
        included_count = manifest_counts.get("recognition_included")
        excluded_count = manifest_counts.get("recognition_excluded")
        if indexed_count != len(all_rows):
            raise TorchOCRDataError(
                "recognition index count does not match materialization manifest"
            )
        actual_included = sum(row.get("status") == "included" for row in all_rows)
        actual_excluded = sum(row.get("status") == "excluded" for row in all_rows)
        if included_count != actual_included or excluded_count != actual_excluded:
            raise TorchOCRDataError(
                "included/excluded recognition counts do not match materialization manifest"
            )
        included: list[dict[str, Any]] = []
        for row in all_rows:
            _validate_index_record(row, max_text_length=max_length)
            if row["split"] != split:
                continue
            if row["status"] == "excluded":
                continue
            # Encode eagerly. This makes an OOV split fail at construction,
            # rather than after workers have already begun a training epoch.
            self.charset.encode(row["content"])
            included.append(row)
        included.sort(key=lambda row: row["record_id"])
        if not included:
            raise TorchOCRDataError(f"no included OCR recognition rows for split {split!r}")
        self.records = tuple(included)

        if image_mode == "lazy" and self.source_root is None:
            raise TorchOCRDataError("source_root is required for lazy OCR crops")
        if image_mode == "materialized" and self.manifest.get("materialize_images") is not True:
            raise TorchOCRDataError(
                "materialized mode requires a generation with materialize_images=true"
            )
        if image_mode == "auto" and self.source_root is None:
            missing = [
                row["record_id"]
                for row in self.records
                if not _materialized_path_if_available(self.root, row)
            ]
            if missing:
                raise TorchOCRDataError(
                    "auto mode found no materialized crop and source_root was not supplied; "
                    f"first record: {missing[0]}"
                )

    def __len__(self) -> int:
        return len(self.records)

    def alphanumeric_prefix_sampling_weights(
        self,
        *,
        power: float = 0.5,
        maximum_weight: float = 8.0,
    ) -> tuple[float, ...]:
        """Return deterministic train-only weights for rare text prefixes.

        A record is eligible when its complete raw text is an ASCII letter
        prefix followed by one or more digits (for example ``R12`` or ``K3``).
        Prefix frequencies are computed only from this dataset's frozen train
        rows. Non-matching text retains weight one, and a finite cap prevents
        singleton prefixes from dominating an epoch.
        """

        if self.split != "train":
            raise TorchOCRDataError(
                "OCR recognition balancing is restricted to the train split"
            )
        if (
            isinstance(power, bool)
            or not isinstance(power, (int, float))
            or not math.isfinite(float(power))
            or not 0.0 <= float(power) <= 1.0
        ):
            raise TorchOCRDataError("power must be finite and in [0, 1]")
        if (
            isinstance(maximum_weight, bool)
            or not isinstance(maximum_weight, (int, float))
            or not math.isfinite(float(maximum_weight))
            or float(maximum_weight) < 1.0
        ):
            raise TorchOCRDataError(
                "maximum_weight must be finite and >= 1"
            )
        prefixes: list[str | None] = []
        counts: dict[str, int] = {}
        for row in self.records:
            content = str(row["content"])
            match = re.fullmatch(r"([A-Za-z]+)[0-9]+", content)
            prefix = match.group(1).casefold() if match is not None else None
            prefixes.append(prefix)
            if prefix is not None:
                counts[prefix] = counts.get(prefix, 0) + 1
        largest = max(counts.values(), default=1)
        selected_power = float(power)
        selected_cap = float(maximum_weight)
        return tuple(
            1.0
            if prefix is None
            else min(
                selected_cap,
                max(1.0, (largest / counts[prefix]) ** selected_power),
            )
            for prefix in prefixes
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = require_torch()
        row = self.records[index]
        image, image_origin = self._load_crop(row)
        try:
            grayscale = image.convert("L")
            source_width, source_height = grayscale.size
            if source_width <= 0 or source_height <= 0:
                raise TorchOCRDataError(
                    f"record {row['record_id']} decoded to an empty image"
                )
            resized_width = max(
                1,
                int(math.floor(source_width * self.image_height / source_height + 0.5)),
            )
            resized = grayscale.resize(
                (resized_width, self.image_height), Image.Resampling.BILINEAR
            )
            array = np.array(resized, dtype=np.float32, copy=True) / np.float32(255.0)
            image_tensor = torch.from_numpy(array).unsqueeze(0).contiguous()
        finally:
            image.close()

        indices = self.charset.encode(row["content"])
        target = torch.tensor(indices, dtype=torch.long)
        target_length = torch.tensor(len(indices), dtype=torch.long)
        image_width = torch.tensor(resized_width, dtype=torch.long)
        metadata = {
            "record_id": row["record_id"],
            "sample_id": row["sample_id"],
            "split": row["split"],
            "source_domain": row.get("source_domain"),
            "source_image": row["source_image"],
            "source_image_sha256": row["source_image_sha256"],
            "crop_box": dict(row["crop_box"]),
            "vertical_long": row.get("vertical_long") is True,
            "image_origin": image_origin,
            "raw_crop_size": [source_width, source_height],
            "resized_size": [resized_width, self.image_height],
        }
        return {
            "image": image_tensor,
            "target": target,
            "target_length": target_length,
            "image_width": image_width,
            "text": row["content"],
            "record_id": row["record_id"],
            "sample_id": row["sample_id"],
            "metadata": metadata,
        }

    def _load_crop(self, row: dict[str, Any]) -> tuple[Image.Image, str]:
        materialized_path = _materialized_path_if_available(self.root, row)
        if self.image_mode in ("materialized", "auto") and materialized_path is not None:
            expected_hash = row.get("output_image_sha256")
            if not _is_sha256(expected_hash):
                if self.image_mode == "materialized":
                    raise TorchOCRDataError(
                        f"record {row['record_id']} lacks materialized image SHA-256"
                    )
            else:
                if self.verify_image_hashes:
                    _verify_file_hash(
                        materialized_path,
                        expected_hash,
                        f"materialized crop for {row['record_id']}",
                    )
                return _decode_image_by_magic(materialized_path), "materialized"
        if self.image_mode == "materialized":
            raise TorchOCRDataError(
                f"materialized crop is unavailable for record {row['record_id']}"
            )
        if self.source_root is None:
            raise TorchOCRDataError(
                f"source_root is required to lazily crop record {row['record_id']}"
            )
        source_path = _safe_child(self.source_root, row["source_image"])
        if self.verify_image_hashes:
            _verify_file_hash(
                source_path,
                row["source_image_sha256"],
                f"source image for {row['record_id']}",
            )
        source = _decode_image_by_magic(source_path)
        try:
            crop_box = row["crop_box"]
            left, top, right, bottom = (
                crop_box["left"],
                crop_box["top"],
                crop_box["right"],
                crop_box["bottom"],
            )
            if left < 0 or top < 0 or right > source.width or bottom > source.height:
                raise TorchOCRDataError(
                    f"lazy crop for {row['record_id']} is outside decoded source image"
                )
            # Match the materializer's crop pipeline exactly before grayscale.
            crop = source.crop((left, top, right, bottom)).convert("RGB")
        finally:
            source.close()
        return crop, "lazy"


def ctc_collate(batch: Sequence[dict[str, Any]], *, pad_value: float = 1.0) -> dict[str, Any]:
    """Pad dynamic-width recognition tensors and concatenate CTC targets."""

    torch = require_torch()
    if not batch:
        raise TorchOCRDataError("cannot CTC-collate an empty batch")
    if not isinstance(pad_value, (int, float)) or isinstance(pad_value, bool):
        raise TorchOCRDataError("pad_value must be numeric")
    pad_number = float(pad_value)
    if not math.isfinite(pad_number):
        raise TorchOCRDataError("pad_value must be finite")

    images = [item.get("image") for item in batch]
    if any(not torch.is_tensor(image) for image in images):
        raise TorchOCRDataError("every batch item must contain an image tensor")
    first = images[0]
    if first.ndim != 3:
        raise TorchOCRDataError("OCR image tensors must have shape [C,H,W]")
    channels, height = int(first.shape[0]), int(first.shape[1])
    widths: list[int] = []
    for image in images:
        if image.ndim != 3 or int(image.shape[0]) != channels or int(image.shape[1]) != height:
            raise TorchOCRDataError(
                "all OCR image tensors in a batch must share channel and height dimensions"
            )
        width = int(image.shape[2])
        if width <= 0:
            raise TorchOCRDataError("OCR image tensor width must be positive")
        widths.append(width)
    maximum_width = max(widths)
    padded = first.new_full(
        (len(images), channels, height, maximum_width), pad_number
    )
    for index, (image, width) in enumerate(zip(images, widths, strict=True)):
        padded[index, :, :, :width] = image

    targets = [item.get("target") for item in batch]
    if any(not torch.is_tensor(target) or target.ndim != 1 for target in targets):
        raise TorchOCRDataError("every batch item must contain a one-dimensional target")
    target_lengths = torch.tensor(
        [int(target.numel()) for target in targets], dtype=torch.long
    )
    if torch.any(target_lengths <= 0):
        raise TorchOCRDataError("CTC targets cannot be empty")
    flat_targets = torch.cat(targets, dim=0).to(dtype=torch.long)
    image_widths = torch.tensor(widths, dtype=torch.long)
    return {
        "images": padded,
        "targets": flat_targets,
        "target_lengths": target_lengths,
        "image_widths": image_widths,
        "texts": [item["text"] for item in batch],
        "record_ids": [item["record_id"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


def make_ctc_collate(*, pad_value: float = 1.0) -> Callable[[Sequence[dict[str, Any]]], dict[str, Any]]:
    """Create a spawn-picklable partially configured CTC collator."""

    if not isinstance(pad_value, (int, float)) or isinstance(pad_value, bool):
        raise TorchOCRDataError("pad_value must be numeric")

    return partial(ctc_collate, pad_value=float(pad_value))


def _validate_commit_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise TorchOCRDataError("unsupported OCR materialization manifest version")
    if manifest.get("kind") != "pcbparse_ocr_paddle_materialization":
        raise TorchOCRDataError("not a pcbparse OCR materialization manifest")
    if manifest.get("complete") is not True:
        raise TorchOCRDataError("OCR materialization is incomplete")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TorchOCRDataError("OCR materialization manifest lacks artifact hashes")


def _verify_committed_artifact(root: Path, manifest: dict[str, Any], relative: str) -> None:
    artifacts = manifest["artifacts"]
    expected = artifacts.get(relative)
    if not _is_sha256(expected):
        raise TorchOCRDataError(f"manifest lacks SHA-256 for {relative}")
    path = _safe_child(root, relative)
    _verify_file_hash(path, expected, f"committed artifact {relative}")


def _validate_index_record(row: dict[str, Any], *, max_text_length: int) -> None:
    record_id = row.get("record_id")
    if row.get("schema_version") != 1:
        raise TorchOCRDataError(f"record {record_id} has unsupported schema version")
    if row.get("split") not in VALID_SPLITS:
        raise TorchOCRDataError(f"record {record_id} has invalid split")
    status = row.get("status")
    if status not in ("included", "excluded"):
        raise TorchOCRDataError(f"record {record_id} has invalid status")
    content = row.get("content")
    if status == "included":
        _validate_raw_text(content, record_id=str(record_id))
        if len(content) > max_text_length:
            raise TorchOCRDataError(
                f"included record {record_id} exceeds max_text_length; "
                "long labels must remain excluded, never truncated"
            )
        if row.get("exclusion_reason") is not None:
            raise TorchOCRDataError(f"included record {record_id} has exclusion_reason")
    else:
        reason = row.get("exclusion_reason")
        if not isinstance(reason, str) or not reason:
            raise TorchOCRDataError(f"excluded record {record_id} lacks exclusion_reason")
    if row.get("normalization_applied") is not False:
        raise TorchOCRDataError(f"record {record_id} does not certify strict raw text")
    if row.get("content_sha256") != _text_sha256(content):
        raise TorchOCRDataError(f"record {record_id} content SHA-256 mismatch")
    text_length = row.get("text_length_codepoints")
    expected_length = len(content) if isinstance(content, str) else None
    if text_length != expected_length:
        raise TorchOCRDataError(f"record {record_id} text length metadata mismatch")
    if not _is_sha256(row.get("source_image_sha256")):
        raise TorchOCRDataError(f"record {record_id} lacks source image SHA-256")
    source_image = row.get("source_image")
    if not isinstance(source_image, str) or not source_image:
        raise TorchOCRDataError(f"record {record_id} lacks source image path")
    output_image = row.get("output_image")
    if not isinstance(output_image, str) or not output_image:
        raise TorchOCRDataError(f"record {record_id} lacks output image path")
    crop = row.get("crop_box")
    if not isinstance(crop, dict):
        raise TorchOCRDataError(f"record {record_id} lacks crop_box")
    values: list[int] = []
    for key in ("left", "top", "right", "bottom"):
        value = crop.get(key)
        if type(value) is not int:
            raise TorchOCRDataError(f"record {record_id} crop_box {key} is not integer")
        values.append(value)
    left, top, right, bottom = values
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise TorchOCRDataError(f"record {record_id} has invalid crop_box")


def _validate_raw_text(text: Any, *, record_id: str) -> None:
    if not isinstance(text, str) or not text:
        raise TorchOCRDataError(f"record {record_id} text must be a non-empty string")
    controls = [name for character, name in (("\t", "TAB"), ("\r", "CR"), ("\n", "LF")) if character in text]
    if controls:
        raise TorchOCRDataError(
            f"record {record_id} contains forbidden raw control(s): {', '.join(controls)}"
        )


def _materialized_path_if_available(root: Path, row: dict[str, Any]) -> Path | None:
    relative = row.get("output_image")
    if not isinstance(relative, str) or not relative:
        return None
    try:
        path = _safe_child(root, relative, require_file=False)
    except TorchOCRDataError:
        raise
    return path if path.is_file() else None


def _decode_image_by_magic(path: Path) -> Image.Image:
    try:
        with Image.open(path) as probe:
            if not probe.format:
                raise TorchOCRDataError(f"image magic did not identify a format: {path}")
            probe.verify()
        with Image.open(path) as opened:
            opened.load()
            return opened.copy()
    except TorchOCRDataError:
        raise
    except (OSError, ValueError) as exc:
        raise TorchOCRDataError(f"cannot decode image {path}: {exc}") from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise TorchOCRDataError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TorchOCRDataError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path, identifier: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise TorchOCRDataError(f"cannot open JSONL {path}: {exc}") from exc
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise TorchOCRDataError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TorchOCRDataError(f"JSONL row {line_number} is not an object")
            identity = row.get(identifier)
            if not isinstance(identity, str) or not identity:
                raise TorchOCRDataError(
                    f"JSONL row {line_number} lacks {identifier}"
                )
            if identity in seen:
                raise TorchOCRDataError(f"duplicate {identifier} {identity!r}")
            seen.add(identity)
            rows.append(row)
    return rows


def _safe_child(root: Path, relative: str, *, require_file: bool = True) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise TorchOCRDataError(f"artifact path must be relative: {relative!r}")
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TorchOCRDataError(f"artifact path escapes root: {relative!r}") from exc
    if require_file and not resolved.is_file():
        raise TorchOCRDataError(f"artifact does not exist: {resolved}")
    return resolved


def _verify_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise TorchOCRDataError(f"{label} does not exist: {path}")
    actual = _sha256_file(path)
    if actual != expected:
        raise TorchOCRDataError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


__all__ = [
    "CTCCharset",
    "TorchOCRDataError",
    "TorchOCRRecognitionDataset",
    "ctc_collate",
    "make_ctc_collate",
    "require_torch",
]
