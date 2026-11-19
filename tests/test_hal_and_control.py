"""
Tests for the Hardware Abstraction Layer (HAL), ASI MS-2000 E-stop preemption,
protocol framing, simulation drivers, and Control layer.
"""

import threading
import time
import unittest
from typing import List
from unittest.mock import patch
import numpy as np

from arVIPAspectro.hardware import (
    StageProtocol,
    CameraProtocol,
    LiveCameraProtocol,
    MS2000Stage,
    OrcaFlashCamera,
    TeledyneBrightfieldCamera,
)
from arVIPAspectro.control import (
    RigState,
    DryRunStage,
    RigController,
)
from arVIPAspectro.control.rig_controller import move_and_settle


class _FakeSerial:
    """In-memory mock of `serial.Serial` for testing exact ASI MS-2000 byte framing."""

    def __init__(self, busy_duration_s: float = 0.0):
        self.is_open = True
        self.writes: List[bytes] = []
        self._busy_until = 0.0
        self._busy_duration_s = busy_duration_s
        self._last_cmd = b""
        self._lock = threading.Lock()

    def reset_input_buffer(self) -> None:
        pass

    def reset_output_buffer(self) -> None:
        pass

    def write(self, data: bytes) -> int:
        with self._lock:
            self.writes.append(data)
            self._last_cmd = data.strip()
            if self._last_cmd.startswith(b"M ") and self._busy_duration_s > 0:
                self._busy_until = time.monotonic() + self._busy_duration_s
            if self._last_cmd == b"\\":
                self._busy_until = 0.0
        return len(data)

    def flush(self) -> None:
        pass

    def read_until(self, expected: bytes = b"\n") -> bytes:
        with self._lock:
            cmd = self._last_cmd
            if cmd == b"W X Y":
                return b":A 0.0 0.0\r\n"
            if cmd == b"W X Y Z":
                return b":A 0.0 0.0 0.0\r\n"
            if cmd == b"W Z":
                return b":A 0.0\r\n"
            if cmd == b"V":
                return b":A Version:9.2\r\n"
            if cmd == b"/":
                if time.monotonic() < self._busy_until:
                    return b"B\r\n"
                return b"N\r\n"
            if cmd.startswith((b"M ", b"R ", b"H ", b"SPIN ")):
                return b":A\r\n"
        return b":A\r\n"

    def close(self) -> None:
        self.is_open = False


class TestHALAndControl(unittest.TestCase):
    def test_protocol_conformance(self):
        self.assertIsInstance(MS2000Stage(), StageProtocol)
        self.assertIsInstance(DryRunStage(), StageProtocol)
        self.assertIsInstance(OrcaFlashCamera(), CameraProtocol)
        self.assertIsInstance(TeledyneBrightfieldCamera(), LiveCameraProtocol)

    def test_ms2000_protocol_framing_and_halt_preemption(self):
        fake_ser = _FakeSerial(busy_duration_s=5.0)
        import serial

        with patch.object(serial, "Serial", return_value=fake_ser):
            stage = MS2000Stage()
            stage.connect(port="COM_MOCK", baudrate=115200, simulate=False)

            self.assertIn(b"W X Y\r", fake_ser.writes)
            self.assertIn(b"V\r", fake_ser.writes)
            self.assertIn(b"W Z\r", fake_ser.writes)

            move_done = threading.Event()
            move_exc = []

            def _run_move():
                try:
                    stage.move_to(x_um=10.0, y_um=20.0, wait=True, timeout_s=10.0)
                except Exception as exc:
                    move_exc.append(exc)
                finally:
                    move_done.set()

            t = threading.Thread(target=_run_move, daemon=True)
            t.start()

            deadline = time.monotonic() + 1.0
            while b"/\r" not in fake_ser.writes and time.monotonic() < deadline:
                time.sleep(0.01)

            t0 = time.monotonic()
            stage.halt()
            halt_elapsed = time.monotonic() - t0

            self.assertLess(halt_elapsed, 0.2, f"halt() blocked for {halt_elapsed:.3f}s!")
            self.assertIn(b"\\\r", fake_ser.writes)
            self.assertIn(b"M X=100.0 Y=200.0\r", fake_ser.writes)

            self.assertTrue(move_done.wait(timeout=1.0), "move_to did not exit after halt()")
            self.assertEqual(move_exc, [])

    def test_simulated_cameras_and_stage(self):
        stage = MS2000Stage()
        stage.connect(simulate=True)
        pos = move_and_settle(stage, x_um=15.0, y_um=-25.0, settling_ms=0.0)
        self.assertEqual(pos["X"], 15.0)
        self.assertEqual(pos["Y"], -25.0)

        orca = OrcaFlashCamera()
        orca.connect(simulate=True)
        orca.set_roi(0, 128, 0, 128, binning=1)
        frame_orca = orca.snap_frame(n_frames=2, stage_pos_um=pos, sim_spectrum_type="laser")
        self.assertEqual(frame_orca.shape, (128, 128))
        self.assertEqual(frame_orca.dtype, np.uint16)

        bf = TeledyneBrightfieldCamera()
        bf.connect(simulate=True)
        bf.set_roi(0, 160, 0, 120)
        frame_bf = bf.snap_frame(n_frames=2, stage_pos_um=pos)
        self.assertEqual(frame_bf.shape, (120, 160))
        self.assertEqual(frame_bf.dtype, np.uint16)

        bf.start_live()
        live_frame = bf.read_live_frame(timeout_s=0.1)
        self.assertIsNotNone(live_frame)
        self.assertEqual(live_frame.shape, (120, 160))
        bf.stop_live()

    def test_dry_run_stage_and_rig_controller(self):
        logs = []
        dry_stage = DryRunStage(logger=logs.append)
        orca = OrcaFlashCamera()
        orca.connect(simulate=True)

        rig = RigController(camera=orca, stage=dry_stage)
        self.assertEqual(rig.state, RigState.IDLE)

        pos = move_and_settle(dry_stage, x_um=42.0, y_um=84.0, settling_ms=0.0)
        self.assertEqual(pos["X"], 42.0)
        self.assertTrue(any("move_to(X=42.0, Y=84.0" in m for m in logs))

        rig.emergency_halt()
        self.assertEqual(rig.state, RigState.IDLE)
        self.assertTrue(any("halt()" in m for m in logs))

    def test_gui_imports_clean(self):
        from arVIPAspectro.gui.app import MainWindow
        from arVIPAspectro.gui.pipeline_panel import PipelineAcquisitionWidget
        from arVIPAspectro.gui.multi_angle_panel import MultiAngleAcquisitionWidget

        self.assertIsNotNone(MainWindow)
        self.assertIsNotNone(PipelineAcquisitionWidget)
        self.assertIsNotNone(MultiAngleAcquisitionWidget)


if __name__ == "__main__":
    unittest.main()
