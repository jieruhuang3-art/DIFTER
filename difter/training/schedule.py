from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveScales:
    ccif: float
    cecc: float
    cei: float


def progressive_scales(step: int) -> ObjectiveScales:
    ccif = min(1.0, max(0.0, step / 500.0))
    cecc = 0.0 if step < 500 else min(1.0, (step - 500) / 1500.0)
    cei = 0.0 if step < 2000 else min(1.0, (step - 2000) / 1500.0)
    return ObjectiveScales(ccif, cecc, cei)
