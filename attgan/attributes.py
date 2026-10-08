"""Attribute order is part of the checkpoint format, never alphabetize it."""
import numpy as np

ATTRIBUTES = (
    "Bald", "Bangs", "Black_Hair", "Blond_Hair", "Brown_Hair",
    "Bushy_Eyebrows", "Eyeglasses", "Male", "Mouth_Slightly_Open",
    "Mustache", "No_Beard", "Pale_Skin", "Young",
)


def binary_labels(values):
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1:] != (len(ATTRIBUTES),) or not np.isin(values, [0, 1]).all():
        raise ValueError("Provide 13 binary labels in the documented checkpoint order.")
    return values


def targets(source, edits, intensity=1.0):
    """Set requested attributes explicitly; preserve the remaining conditioning.

    Resolve the upstream hair/bangs conflicts, but reject mutually contradictory
    explicit edits instead of silently allowing argument order to choose a winner.
    """
    result = binary_labels(source).copy()
    if not np.isfinite(intensity) or not 0 < intensity <= 3:
        raise ValueError("Intensity must be finite and in (0, 3].")
    if any(name not in ATTRIBUTES or value not in (0, 1) for name, value in edits.items()):
        raise ValueError("Unknown attribute or nonbinary edit value.")
    conflicts = [("Bald", "Bangs"), ("Black_Hair", "Blond_Hair"),
                 ("Black_Hair", "Brown_Hair"), ("Blond_Hair", "Brown_Hair")]
    for a, b in conflicts:
        if edits.get(a) == edits.get(b) == 1:
            raise ValueError(f"Conflicting requested attributes: {a}, {b}")
    for name, value in edits.items():
        result[..., ATTRIBUTES.index(name)] = value
        if value:
            for a, b in conflicts:
                if name in (a, b):
                    other = b if name == a else a
                    result[..., ATTRIBUTES.index(other)] = 0
    conditioning = result * 2 - 1
    for name in edits:
        conditioning[..., ATTRIBUTES.index(name)] *= intensity
    return conditioning.astype(np.float32)
