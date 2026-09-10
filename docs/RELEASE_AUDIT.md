# Release audit

This audit covers the tracked public-release workspace. Generated cache directories and the Git metadata directory are excluded. The secret-pattern scan excludes this report itself because the required pass labels are self-referential.

```text
NO_PRIVATE_PATHS = TRUE
NO_DATASETS = TRUE
NO_CHECKPOINTS = TRUE
NO_EXPERIMENT_RESULTS = TRUE
NO_SECRETS = TRUE
SYNTHETIC_TESTS_ONLY = TRUE
SOURCE_ONLY_TRAINING_CONTRACT = TRUE
```

Checks performed:

- No private absolute path or internal experiment identifier occurs in release source or documentation.
- No PCAP, processed dataset body, real manifest, model weight, checkpoint, prediction, log, or result artifact is present.
- The TSV example contains only its schema header.
- Dependency declarations are derived from release imports rather than an environment dump.
- The license text and copyright are inherited from the source-project root.
- Synthetic CPU tests cover K=16 masked aggregation, CCIF dimensions, CECC pair semantics, finite training forward, and the stable-only inference path.
