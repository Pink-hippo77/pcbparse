"""Evidence-constrained offline component-library knowledge (A07).

The public library file `赛题六公开数据集/cpntLibrary` is a single JSON object
without an extension. Pin arrays are multi-source symbol aggregations: row
count is not physical pin count, and `pin_id` is not unique. This package
builds a deterministic index and a retrieval matcher that never invents
coordinates or invisible pins and keeps UNKNOWN when evidence is ambiguous.
"""

from pcbparse.knowledge.library_index import (
    LibraryIndexError,
    build_library_index,
    default_library_path,
    load_raw_library,
    normalize_model_name,
    split_variant_suffix,
    write_library_index,
)
from pcbparse.knowledge.library_matcher import (
    RetrievalDecision,
    RetrievalQuery,
    RetrievalResult,
    VisiblePinEvidence,
    build_retrieval_ab_report,
    retrieve_component,
)

__all__ = [
    "LibraryIndexError",
    "RetrievalDecision",
    "RetrievalQuery",
    "RetrievalResult",
    "VisiblePinEvidence",
    "build_library_index",
    "build_retrieval_ab_report",
    "default_library_path",
    "load_raw_library",
    "normalize_model_name",
    "retrieve_component",
    "split_variant_suffix",
    "write_library_index",
]
