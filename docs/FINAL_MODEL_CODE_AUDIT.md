# Final model code audit

## Outcome

The release follows the final executable model chain rather than reconstructing a method from prose. The extracted chain is:

```text
processed traffic-window rows
-> flow grouping and deterministic K=16 window selection
-> HPTF window encoder
-> masked arithmetic mean
-> CCIF stable and factor-specific projectors
-> stable cosine classifier
```

During training, a separate source-only support batch enters a detached HPTF path. CCIF auxiliaries, CECC, and CEI update the factorization-side modules; the main classification path updates HPTF, the stable projector, and the classifier. Inference computes only HPTF, masked mean, the stable projector, and the classifier.

## Implementation mapping

- Training entry: `scripts/train.py` and `difter/training/trainer.py`.
- HPTF: `difter/models/hptf.py`.
- Traffic-window construction and selection: `difter/data/traffic_windows.py`.
- K=16 aggregation: `difter/models/aggregation.py`.
- CCIF: `difter/models/ccif.py` and `difter/losses/hsic.py`.
- CECC: `difter/models/cecc.py` and `difter/losses/contrastive.py`.
- CEI: `difter/models/cei.py` and `difter/losses/consistency.py`.
- Stable classifier: `difter/models/classifier.py`.
- Progressive schedule: `difter/training/schedule.py`.
- Evaluation and inference: `difter/evaluation/evaluator.py`, `scripts/evaluate.py`, and `scripts/infer.py`.

## Confirmed mathematical contract

- Stable projector: `768 -> 512 -> 256`.
- Factor-specific projector: `768 -> 256 -> 128`.
- CECC projection: `256 -> 256 -> 128`, followed by L2 normalization.
- CECC positives require the same class and a different source environment; negatives require different classes.
- CEI donors are same-class and different in the swapped factor. Donor environment representations are stop-gradient.
- K=16 denotes traffic windows per flow, not packets.
- Model selection uses source-validation Macro-F1.

## Provenance and license

The extracted research implementation is covered by the MIT license present at the source-project root. This release preserves its copyright and license text. No external baseline source, model weights, data, result artifacts, or private paths are included.
