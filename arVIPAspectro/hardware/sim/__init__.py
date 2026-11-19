"""
Simulation frame generators for offline development, dry-run testing, and UI verification.
"""

from .camera_sim import generate_vipa_frame, generate_brightfield_frame

__all__ = [
    "generate_vipa_frame",
    "generate_brightfield_frame",
]
