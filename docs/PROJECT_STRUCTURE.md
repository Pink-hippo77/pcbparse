# Project structure

`E:\EDA挑战题目六` is the only project root.  The former sibling folders
`E:\EDA比赛测试` and `E:\EDA官方dockers` were consolidated here on
2026-09-22 without dropping files.

```text
EDA挑战题目六/
├── .github/                     # CI workflow
├── src/                         # Active Python package
├── tests/                       # Unit and integration tests
├── configs/                     # Runtime, training, and candidate configs
├── scripts/                     # Active operational scripts
├── tools/                       # Interactive/report-building utilities
├── data_tools/                  # Dataset command-line tooling
├── data/                        # Manifests, splits, and derived datasets
├── Dataset300/                  # Official Dataset300 source
├── 赛题六公开数据集/             # Additional public challenge dataset
├── models/                      # Registry, locks, candidates, and releases
├── weights/                     # Standalone local initialization weights
├── docker/                      # Project image build and entrypoint
├── artifacts/                   # Audits and durable generated evidence
├── outputs/                     # User-facing inference outputs
├── output/                      # Generated report/PDF exports
├── work/                        # Large reproducible experiment runs
├── tmp/                         # Disposable rendering and analysis scratch data
├── docs/                        # Current documentation
├── archive/
│   └── legacy-training/         # Former EDA比赛测试 workspace
│       ├── scripts/             # Archived, path-portable entry points
│       ├── runs/                # Historical checkpoints and logs
│       ├── results/inference/   # Historical predictions
│       └── documents/           # Historical/reference documents
└── external/
    └── official-docker/         # Former EDA官方dockers workspace
        ├── edarace-agent-base_ubuntu24.tar
        └── edarace-agent-base_ubuntu24/  # Unpacked OCI layout
```

## Directory rules

- Active code and configuration stay in the established package layout; they
  are not flattened because imports, tests, release locks, and Docker builds
  rely on it.
- `archive/` is read-only historical material.  New experiments belong in
  `work/`, and durable audit results belong in `artifacts/`.
- `external/` stores vendor or competition-provided assets.  Do not modify the
  official Docker archive or OCI blobs in place.
- Virtual environments and caches remain local implementation details and are
  ignored by Git.

## Consolidation verification

| Former root | Destination | Files | Bytes | Result |
|---|---|---:|---:|---|
| `E:\EDA比赛测试` | `archive/legacy-training` | 602 | 2,791,419,432 | source removed after exact count/size match |
| `E:\EDA官方dockers` | `external/official-docker` | 19 | 6,215,533,044 | source removed after exact count/size match |

The destination counts above describe the migrated payload before adding the
new README and path-helper files.
