"""Shared activity-wave shape for curses and inline terminal renderers."""

from __future__ import annotations

import math

from anomx.agent.ui.constants import (
    ACTIVITY_WAVE_PAUSE_FRAMES,
    ACTIVITY_WAVE_TRAVEL_FRAMES,
    ACTIVITY_WAVE_WIDTH,
)


def activity_wave_strengths(spans: list[tuple[int, str, int]], frame: int) -> list[float]:
    """Return a soft cosine-shaped crest strength (0-1) for every terminal cell span."""
    if not spans:
        return []
    phase = frame % (ACTIVITY_WAVE_TRAVEL_FRAMES + ACTIVITY_WAVE_PAUSE_FRAMES)
    label_width = spans[-1][0] + spans[-1][2]
    radius = ACTIVITY_WAVE_WIDTH / 2
    center = -radius + (label_width + 2 * radius) * phase / (ACTIVITY_WAVE_TRAVEL_FRAMES - 1)
    strengths = []
    for column, _glyph, cells in spans:
        distance = abs(column + cells / 2 - center) / radius
        strengths.append(
            (1 + math.cos(math.pi * distance)) / 2
            if phase < ACTIVITY_WAVE_TRAVEL_FRAMES and distance < 1 else 0
        )
    return strengths
