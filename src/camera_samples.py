"""Camera-specific steering-label helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CameraSample:
    """An image path paired with its corrected steering label."""

    image_path: str
    steering: float
    camera: str


_CAMERA_OFFSETS = {"center": 0.0, "left": 1.0, "right": -1.0}


def correct_steering(angle: float, camera: str, correction: float = 0.2) -> float:
    """Apply the conventional steering correction for a camera position."""

    angle = float(angle)
    correction = float(correction)
    if camera == "center":
        return angle
    if camera == "left":
        return angle + correction
    if camera == "right":
        return angle - correction
    raise ValueError("camera must be center, left, or right")


def correct_steering_lookup(
    angle: float,
    camera: str,
    correction: float = 0.2,
) -> float:
    """Apply the same correction through a camera-to-offset lookup table."""

    try:
        offset = _CAMERA_OFFSETS[camera]
    except KeyError:
        raise ValueError("camera must be center, left, or right")
    return float(angle) + offset * float(correction)


def make_camera_samples(
    center_path: str,
    left_path: str,
    right_path: str,
    angle: float,
    correction: float = 0.2,
) -> Tuple[CameraSample, ...]:
    """Return center, left, and right samples in a stable order."""

    return tuple(
        CameraSample(path, correct_steering(angle, camera, correction), camera)
        for path, camera in (
            (center_path, "center"),
            (left_path, "left"),
            (right_path, "right"),
        )
    )
