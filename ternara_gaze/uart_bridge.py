"""Framed UART protocol and unified backend interface (Local Q27 Oracle <-> Live FPGA UART).

Packet format (Host -> FPGA):
  [0..1]   SYNC = 0x54 0x47 ('T', 'G')
  [2]      FLAGS (bit 0 = request 32-byte h2 vector for calibration)
  [3]      SEQ (uint8 wrapping sequence counter)
  [4..515] PAYLOAD = 512 bytes packed dual-eye 4-bit nibbles (left 16x32 + right 16x32)
  [516..517] CRC16-CCITT (big-endian) over bytes [0..515]
Total TX: 518 bytes (~5.6 ms at 921,600 baud -> >170 FPS wire ceiling).

Packet format (FPGA -> Host):
  [0..1]   SYNC = 0x47 0x54 ('G', 'T')
  [2]      STATUS (0x00 = OK, bit 0 = h2 included, bit 7 = pack5/trit FAULT)
  [3]      SEQ (echoed)
  [4..11]  4 x int16 little-endian: (gaze_x, gaze_y, blink_left, blink_right)
  [12..43] optional 32 x int8 h2 vector (present if STATUS bit 0 set)
  [last 2] CRC16-CCITT (big-endian)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .oracle import (
    FRAME_PACKED_BYTES,
    OracleOutput,
    oracle_forward_fast,
    pack_dual_eye_nibbles,
    unpack_dual_eye_nibbles,
)
from .pack5 import TernaryCartridge

TX_SYNC = b"TG"
RX_SYNC = b"GT"
FLAG_WANT_H2 = 0x01
STATUS_HAS_H2 = 0x01
STATUS_FAULT = 0x80


def crc16_ccitt(data: bytes | bytearray | memoryview) -> int:
    """Standard CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_tx_frame(
    left_u4: np.ndarray,
    right_u4: np.ndarray,
    seq: int = 0,
    want_h2: bool = True,
) -> bytes:
    flags = FLAG_WANT_H2 if want_h2 else 0
    payload = pack_dual_eye_nibbles(left_u4, right_u4)
    body = TX_SYNC + bytes([flags, seq & 0xFF]) + payload
    crc = crc16_ccitt(body)
    return body + struct.pack(">H", crc)


def decode_tx_frame(frame: bytes) -> tuple[np.ndarray, np.ndarray, int, bool]:
    expected_len = 4 + FRAME_PACKED_BYTES + 2
    if len(frame) != expected_len:
        raise ValueError(f"Expected TX frame length {expected_len}, got {len(frame)}")
    if frame[:2] != TX_SYNC:
        raise ValueError(f"Bad TX sync: {frame[:2]!r}")
    body, crc_bytes = frame[:-2], frame[-2:]
    (rx_crc,) = struct.unpack(">H", crc_bytes)
    if crc16_ccitt(body) != rx_crc:
        raise ValueError("TX frame CRC16 mismatch")
    flags = frame[2]
    seq = frame[3]
    left, right = unpack_dual_eye_nibbles(frame[4 : 4 + FRAME_PACKED_BYTES])
    return left, right, seq, bool(flags & FLAG_WANT_H2)


def encode_rx_frame(out: OracleOutput, seq: int, include_h2: bool = True) -> bytes:
    status = STATUS_HAS_H2 if include_h2 else 0
    g = struct.pack(
        "<hhhh",
        max(-32768, min(32767, out.gaze_x)),
        max(-32768, min(32767, out.gaze_y)),
        max(-32768, min(32767, out.blink_left)),
        max(-32768, min(32767, out.blink_right)),
    )
    h2_bytes = out.h2.astype(np.int8).tobytes() if include_h2 else b""
    body = RX_SYNC + bytes([status, seq & 0xFF]) + g + h2_bytes
    crc = crc16_ccitt(body)
    return body + struct.pack(">H", crc)


def decode_rx_frame(frame: bytes, expected_seq: int | None = None) -> OracleOutput:
    if len(frame) < 14:
        raise ValueError(f"RX frame too short: {len(frame)}")
    if frame[:2] != RX_SYNC:
        raise ValueError(f"Bad RX sync: {frame[:2]!r}")
    body, crc_bytes = frame[:-2], frame[-2:]
    (rx_crc,) = struct.unpack(">H", crc_bytes)
    if crc16_ccitt(body) != rx_crc:
        raise ValueError("RX frame CRC16 mismatch")
    status = frame[2]
    seq = frame[3]
    if status & STATUS_FAULT:
        raise RuntimeError("FPGA reported pack5/Q27 FAULT status")
    if expected_seq is not None and seq != (expected_seq & 0xFF):
        raise ValueError(f"Sequence mismatch: expected {expected_seq & 0xFF}, got {seq}")
    gx, gy, bl, br = struct.unpack("<hhhh", frame[4:12])
    if status & STATUS_HAS_H2:
        if len(frame) != 46:
            raise ValueError(f"Expected 46-byte RX frame with h2, got {len(frame)}")
        h2 = np.frombuffer(frame[12:44], dtype=np.int8).copy()
    else:
        h2 = np.zeros((32,), dtype=np.int8)
    return OracleOutput(gaze_x=gx, gaze_y=gy, blink_left=bl, blink_right=br, h2=h2)


class GazeInferenceBackend(Protocol):
    def infer(self, left_u4: np.ndarray, right_u4: np.ndarray, want_h2: bool = True) -> OracleOutput:
        ...


class OracleLoopbackBackend:
    """Exercises the full binary UART framing + CRC16 over the bit-exact Q27 integer oracle."""

    def __init__(self, cartridge: TernaryCartridge) -> None:
        self.cartridge = cartridge
        self._seq = 0

    def infer(self, left_u4: np.ndarray, right_u4: np.ndarray, want_h2: bool = True) -> OracleOutput:
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFF
        tx = encode_tx_frame(left_u4, right_u4, seq=seq, want_h2=want_h2)
        l_dec, r_dec, seq_dec, want_dec = decode_tx_frame(tx)
        out = oracle_forward_fast(self.cartridge, l_dec, r_dec)
        rx = encode_rx_frame(out, seq=seq_dec, include_h2=want_dec)
        return decode_rx_frame(rx, expected_seq=seq)


class FpgaUartBackend:
    """Talks to a live AX7203 / ECP5 FPGA running the Ternara-Gaze Q27 core over UART."""

    def __init__(self, port: str, baudrate: int = 921600, timeout_s: float = 0.25) -> None:
        import serial

        self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout_s)
        self._seq = 0

    def close(self) -> None:
        self._ser.close()

    def infer(self, left_u4: np.ndarray, right_u4: np.ndarray, want_h2: bool = True) -> OracleOutput:
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFF
        tx = encode_tx_frame(left_u4, right_u4, seq=seq, want_h2=want_h2)
        self._ser.reset_input_buffer()
        self._ser.write(tx)
        expected_len = 46 if want_h2 else 14
        rx = self._ser.read(expected_len)
        if len(rx) != expected_len:
            raise TimeoutError(f"FPGA UART timeout: expected {expected_len} bytes, got {len(rx)}")
        return decode_rx_frame(rx, expected_seq=seq)
