"""
Pure geometry and formatting helpers — no UI dependencies.
"""

import numpy as np
from math import atan2, degrees, acos, sqrt


def line_distance(p1, p2) -> float:
    """Euclidean distance between two points in pixels."""
    return float(np.linalg.norm(np.array(p2) - np.array(p1)))


def angle_between(p1, p2, p3) -> float:
    """Angle at vertex p2 formed by rays p2→p1 and p2→p3, in degrees [0, 180]."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    rad = atan2(v2[1], v2[0]) - atan2(v1[1], v1[0])
    deg = abs(degrees(rad))
    return 360 - deg if deg > 180 else deg


def preview_angle(p1, p2, mx, my):
    """
    Live angle preview between arm p2→p1 and arm p2→(mx, my).
    Returns degrees, or None if either arm has zero length.
    """
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (mx - p2[0],    my - p2[1])
    mag = sqrt(v1[0]**2 + v1[1]**2) * sqrt(v2[0]**2 + v2[1]**2)
    if mag > 0:
        return degrees(acos(max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / mag))))
    return None


def arc_canvas_points(center, start, end, radius=50, n=100):
    """
    Compute canvas-space arc points (Y-axis inverted) for drawing on a tk.Canvas.
    Returns a list of (x, y) integer tuples.
    """
    sx, sy = start[0] - center[0], center[1] - start[1]
    ex, ey = end[0]   - center[0], center[1] - end[1]
    sa = atan2(sy, sx)
    ea = atan2(ey, ex)
    span = (ea - sa) % (2 * np.pi)   # counterclockwise span in [0, 2π)
    # shorter arc: CCW if span ≤ π, CW if span > π
    angle_end = sa + span if span <= np.pi else sa + span - 2 * np.pi
    return [
        (int(center[0] + radius * np.cos(a)),
         int(center[1] - radius * np.sin(a)))
        for a in np.linspace(sa, angle_end, n)
    ]


def arc_pil_params(center, start, end):
    """
    Compute (start_deg, end_deg, radius) for drawing an arc with PIL.
    The arc represents the smaller angle at center between the two arms.
    """
    sa = atan2(start[1] - center[1], start[0] - center[0])
    ea = atan2(end[1]   - center[1], end[0]   - center[0])
    if sa < 0: sa += 2 * np.pi
    if ea < 0: ea += 2 * np.pi
    span = ea - sa
    if span < 0: span += 2 * np.pi
    if span > np.pi:
        sa, ea = ea, sa
    r = int(min(
        np.linalg.norm(np.array(start) - np.array(center)),
        np.linalg.norm(np.array(end)   - np.array(center))
    ) * 0.25)
    return float(np.degrees(sa)), float(np.degrees(ea)), r


def format_distance(px_dist: float, scale_factor, unit: str) -> str:
    """Format a pixel distance using the current calibration and unit selection."""
    if unit == "px" or scale_factor is None:
        return f"{px_dist:.1f} px"
    mm = px_dist * scale_factor
    if unit == "mm": return f"{mm:.2f} mm"
    if unit == "cm": return f"{mm / 10:.3f} cm"
    if unit == "in": return f"{mm / 25.4:.4f} in"
    return f"{mm:.2f} mm"