"""Explicit rig state machine states and transition validation."""

from enum import Enum


class RigState(str, Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    LIVE = "LIVE"
    CALIBRATING = "CALIBRATING"
    SCANNING = "SCANNING"
    FAULT = "FAULT"
