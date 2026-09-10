# Reproducibility

The default training contract uses seed 42, K=16 window units, AdamW, 6000 successful optimizer steps, gradient clipping at 1.0, and source-validation Macro-F1 checkpoint selection every 250 successful steps.

The progressive optimization schedule leaves classification active from step zero:

- CCIF auxiliaries ramp linearly from zero to canonical weights over steps 0--500.
- CECC is off before step 500 and ramps to its canonical coefficient over steps 500--2000.
- CEI is off before step 2000 and ramps to its canonical coefficients over steps 2000--3500.
- The complete objective is active from step 3500 onward.

The main sampler and the source-only class/environment support sampler are separate. The support path does not propagate auxiliary gradients into HPTF. Target data are reserved for final evaluation and cannot select preprocessing, hyperparameters, training duration, or checkpoints.

The released configuration expects a compatible pretrained HPTF backbone path to be supplied locally. Pretrained weights are not redistributed. The final optimization uses the last two contextual encoder blocks at learning rates `3e-6` and `5e-6`; factorization, classifier, and training-only heads use `5e-5`. Source-train EMA class-prototype alignment begins at step 500 with coefficient `0.05` for both feature and classifier-weight terms.
