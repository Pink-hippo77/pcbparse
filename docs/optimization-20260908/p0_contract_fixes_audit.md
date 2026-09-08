# P0 Contract Fixes Audit — 2026-09-08

## Summary

Implemented three blocking fixes identified in the architecture review (`docs/架构阻塞与硬件优化审查_2026-09-08.md`):

| ID | Issue | Status | Files Changed |
|---|---|---|---|
| P0-A | Graph builder/validator connectivity contract mismatch | ✅ Fixed | `graph/physical.py`, `validation/graph.py`, `pipeline/orchestrator.py` |
| P0-B | Association rejects valid designators, collapses duplicates | ✅ Fixed | `association/text_component.py` |
| P1-A | ONNX Pin backend silently falls back to CPU | ✅ Fixed | `perception/torch_pin_adapter.py` |

## Files Modified

| File | Change | Reason |
|---|---|---|
| `src/pcbparse/graph/physical.py` | Renamed `_segments_have_collinear_overlap` → `segments_have_collinear_overlap` (public) | Allow validator to reuse the same geometric check |
| `src/pcbparse/validation/graph.py` | Added `collinear_merge_tolerance_px` and `collinear_gap_merge_tolerance_px` parameters to `_has_disconnected_segments`, `validate_graph`, and `GraphValidator`; imported `segments_have_collinear_overlap` from `physical.py` | Validator now matches builder's connectivity logic |
| `src/pcbparse/pipeline/orchestrator.py` | Replaced `validate_and_repair_once` with explicit validator construction passing graph tolerances from config; imported `repair_ir_once`, `merge_diagnostics`, `stable_diagnostics` | Pipeline passes correct tolerances to validator |
| `src/pcbparse/association/text_component.py` | Added `_ALL_LETTER_DESIGNATOR` regex, `_ALL_LETTER_COMPATIBLE_CATEGORIES`, `_is_valid_designator()` helper; updated `resolve_component_keys` to allow duplicate designators; updated `eligible` check in `associate_component_texts` to use category-aware validation | Accept "RUN", "VCC", etc. for compatible categories; allow two components with same visible label |
| `src/pcbparse/perception/torch_pin_adapter.py` | Added post-creation session provider verification in `_ONNXBackend.__init__`; explicit CUDA requests now fail if session doesn't have CUDA provider; auto mode logs warning on fallback | Prevent silent CUDA→CPU fallback |
| `artifacts/audits/architecture_review_20260908/reproduce_contract_findings.py` | Updated connectivity probes to test both matching tolerances and legacy behavior; updated naming probes to capture new behavior | Document before/after evidence |

## Evidence: Before/After Contract Probes

### P0-A: Graph Connectivity

| Case | Builder | Validator (matching tolerances) | Validator (legacy, no tolerances) |
|---|---|---|---|
| `collinear_overlap` ([0,20] & [10,30], overlap=0.75px) | 1 net | **[] (no error)** ✅ | `DISCONNECTED_SEGMENTS` (expected) |
| `collinear_gap_5px` ([0,10] & [15,25], gap=6px) | 1 net | **[] (no error)** ✅ | `DISCONNECTED_SEGMENTS` (expected) |
| `distant_reused_polyline` (50px apart, same polyline ID) | 2 nets | `[]` (correct — builder doesn't merge) | `[]` (correct) |

**Key behavior**: When the pipeline passes `collinear_merge_tolerance_px` and `collinear_gap_merge_tolerance_px` from config to the `GraphValidator`, collinear-connected segments are no longer falsely flagged as disconnected. Legacy callers (no tolerances) retain the old behavior for backward compatibility.

### P0-B: Component Naming

| Case | Before | After |
|---|---|---|
| `R1` (standard designator) | `["R1"]`, no diagnostics | `["R1"]`, no diagnostics |
| `RUN` (all-letter, category="vbus_in") | `[null]` + `DESIGNATOR_FORMAT_REJECTED` | **`["RUN"]`**, no diagnostics ✅ |
| `duplicate_R1` (two components, same label) | `["R1", null]` + `COMPONENT_KEY_CONFLICT` | **`["R1", "R1"]`**, no diagnostics ✅ |

**Key behavior**: 
1. All-letter designators (2-6 uppercase letters) are accepted when the component category is compatible (starts with V, G, BUS, SIG, or is in the explicit set).
2. Two components with independent evidence for the same visible designator both receive the key. The serializer handles `_1`/`_2` suffixing for official output.

### P1-A: ONNX CUDA Verification

| Scenario | Before | After |
|---|---|---|
| `device='cuda'`, CUDA DLLs missing | Silent fallback to CPU | **`TorchPinAdapterError`** with clear message ✅ |
| `device='auto'`, CUDA DLLs missing | Silent fallback to CPU | Warning logged, continues on CPU ✅ |
| `device='cpu'` | CPU session | CPU session (unchanged) |

**Key behavior**: Explicit CUDA requests are validated post-session-creation. The `session.get_providers()` check catches cases where `ort.get_available_providers()` reports CUDA but the actual session creation silently falls back.

## Test Results

```
Pre-fix:  407 passed, 70 skipped, 76 subtests (112s)
Post-fix: 407 passed, 70 skipped, 76 subtests (97s)
```

No regressions. All existing unit tests pass. The 70 skipped tests require GPU models or external data not available in the test environment.

## Model Lock Verification

```
SHA-256: af35e1299b755d03ff37889f9bb26e33601373f0f7f19db3a84468e896461413
```

Unchanged. No model weights, configurations, or blessed status were modified.

## What These Fixes Do NOT Change

1. **No new models or training** — These are code-level contract fixes only.
2. **No tolerance value changes** — The default tolerances remain 0.0 for collinear merge. The pipeline reads them from config; existing configs with 0.0 produce identical behavior to before.
3. **No evaluation metric changes** — raw-v1, guide-v1, and strict scoring are untouched.
4. **No Docker/ORT version changes** — The CUDA verification is runtime behavior; the ORT version compatibility issue remains a separate task.
5. **No split/data changes** — release50 exposure status is unchanged.

## Remaining Work (per architecture review)

| Priority | Item | This change? |
|---|---|---|
| P0-C | End-to-end accuracy gap (designator exact 17.5%, pin number 4.77%) | No — requires training |
| P0-D |评测集污染与评分口径 | No — requires new data |
| P1-A (partial) | ORT CUDA 12 version compatibility | No — requires environment work |
| P1-B | Domain differences (KiCad weak) | No — requires training |
| P1-C | Failure IR replay, disk budget | No — requires engineering |

## Rollback

All changes are in source files only. To revert:
```bash
git checkout HEAD -- src/pcbparse/graph/physical.py src/pcbparse/validation/graph.py src/pcbparse/pipeline/orchestrator.py src/pcbparse/association/text_component.py src/pcbparse/perception/torch_pin_adapter.py
```

The reproduce script should also be reverted:
```bash
git checkout HEAD -- artifacts/audits/architecture_review_20260908/reproduce_contract_findings.py
```
