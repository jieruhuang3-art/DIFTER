# Method

DIFTER uses one shared HPTF encoder for traffic-window representations. Up to 16 window representations from each flow are aggregated by a mask-aware arithmetic mean to produce `h`.

CCIF maps `h` through a `768 -> 512 -> 256` stable projector and one `768 -> 256 -> 128` projector for each source-environment factor. The stable classifier sees only `z_s`. The canonical auxiliary objective is

```text
L_CCIF = 0.10 L_inv + 0.20 L_env + 0.05 L_orth + 0.10 L_rec
```

`L_inv` is a class-conditional finite-sample HSIC surrogate targeting stable/environment-factor independence conditional on class. It is neither an identifiability result nor a proof of statistical independence.

CECC uses a `256 -> 256 -> 128` normalized projection and temperature `0.07`. It treats same-class/different-source-environment pairs as positives, different-class pairs as negatives, and ignores same-class/same-environment pairs.

CEI keeps each anchor's stable representation and substitutes detached factor-specific environment representations from same-class donors whose corresponding source-environment factor differs. A shared reconstructor produces a counterfactual flow representation, which is factorized again. Its loss is

```text
L_CEI = 0.20 L_cls_cei + 0.10 L_sem_cei + 0.05 L_env_cei
```

The overall objective is

```text
L = L_cls + scheduled(L_CCIF) + scheduled(0.20 L_CECC) + scheduled(L_CEI)
```

The main classification path updates HPTF. Auxiliary support representations are computed through a detached HPTF path, preserving the gradient routing used by the final implementation.
