"""pack5: Dense 5-trits-per-byte (radix-243) codec and .tgaze.bin weight cartridge format.

Design invariants:
- Each trit t in {-1, 0, +1} maps to digit d = t + 1 in {0, 1, 2}.
- Five trits (t0..t4) pack into one unsigned 8-bit lane:
    byte = d0 + 3*d1 + 9*d2 + 27*d3 + 81*d4   in [0, 242].
- Bytes 243..255 are strictly invalid (fault-checked on decode), matching Q27 fault discipline.
- Strictly 8-bit memory lanes: avoids 9-bit BRAM parity-lane hazards on 7-Series/ECP5 while
  achieving 99.06% of the theoretical log2(3) = 1.585 bits/trit density (+56.25% capacity vs 2-bit/trit).
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

CARTRIDGE_MAGIC = b"TGAZ"
CARTRIDGE_VERSION = 1
POW3 = (1, 3, 9, 27, 81)
MAX_VALID_PACK5_BYTE = 242  # 3^5 - 1

# Precomputed 243x5 decode table (exact integer LUT, matches hardware ROM)
_DECODE_LUT = np.zeros((243, 5), dtype=np.int8)
for _b in range(243):
    _v = _b
    for _k in range(5):
        _DECODE_LUT[_b, _k] = (_v % 3) - 1
        _v //= 3


class Pack5FaultError(ValueError):
    """Raised when an invalid trit value or reserved pack5 byte (>=243) is encountered."""


def pack5_encode(trits: Sequence[int] | np.ndarray) -> bytes:
    """Pack a 1D sequence of trits in {-1, 0, +1} into radix-243 bytes (5 trits/byte)."""
    arr = np.asarray(trits, dtype=np.int16).reshape(-1)
    if arr.size == 0:
        return b""
    if np.any((arr < -1) | (arr > 1)):
        bad = arr[(arr < -1) | (arr > 1)][:5].tolist()
        raise Pack5FaultError(f"Invalid trit values outside {{-1, 0, +1}}: {bad}")

    n = arr.size
    pad = (-n) % 5
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.int16)])
    digits = (arr + 1).reshape(-1, 5)
    packed = (
        digits[:, 0]
        + 3 * digits[:, 1]
        + 9 * digits[:, 2]
        + 27 * digits[:, 3]
        + 81 * digits[:, 4]
    ).astype(np.uint8)
    return packed.tobytes()


def pack5_decode(data: bytes | bytearray | memoryview, count: int) -> np.ndarray:
    """Decode `count` trits in {-1, 0, +1} from radix-243 bytes, rejecting bytes > 242."""
    if count < 0:
        raise ValueError("count must be >= 0")
    if count == 0:
        return np.empty((0,), dtype=np.int8)

    expected_bytes = (count + 4) // 5
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size < expected_bytes:
        raise Pack5FaultError(
            f"Truncated pack5 payload: need {expected_bytes} bytes for {count} trits, got {raw.size}"
        )
    used = raw[:expected_bytes]
    invalid_idx = np.flatnonzero(used > MAX_VALID_PACK5_BYTE)
    if invalid_idx.size > 0:
        idx = int(invalid_idx[0])
        raise Pack5FaultError(
            f"Reserved pack5 byte {int(used[idx])} (> {MAX_VALID_PACK5_BYTE}) at offset {idx}"
        )

    trits = _DECODE_LUT[used].reshape(-1)
    # Enforce zero-padding invariant on trailing trits
    if trits.size > count:
        trailing = trits[count:]
        if np.any(trailing != 0):
            raise Pack5FaultError(f"Non-zero padding trits in final pack5 byte: {trailing.tolist()}")
    return trits[:count].copy()


@dataclass(frozen=True)
class TernaryCartridge:
    """Immutable, hardware-verifiable ternary model cartridge (.tgaze.bin)."""

    w1: np.ndarray  # (h1, in_dim) int8 in {-1, 0, +1}
    b1: np.ndarray  # (h1,) int16
    s1: int         # arithmetic right-shift after layer 1
    w2: np.ndarray  # (h2, h1) int8 in {-1, 0, +1}
    b2: np.ndarray  # (h2,) int16
    s2: int         # arithmetic right-shift after layer 2
    w3: np.ndarray  # (out_dim, h2) int8 in {-1, 0, +1}
    b3: np.ndarray  # (out_dim,) int16

    def __post_init__(self) -> None:
        if self.w1.ndim != 2 or self.w2.ndim != 2 or self.w3.ndim != 2:
            raise ValueError("Weight matrices must be 2D")
        h1, _ = self.w1.shape
        h2, h1_check = self.w2.shape
        out_dim, h2_check = self.w3.shape
        if h1 != h1_check or h2 != h2_check:
            raise ValueError("Layer dimension mismatch")
        if self.b1.shape != (h1,) or self.b2.shape != (h2,) or self.b3.shape != (out_dim,):
            raise ValueError("Bias vector shape mismatch")
        if not (0 <= self.s1 <= 15 and 0 <= self.s2 <= 15):
            raise ValueError("Shift amounts must be in [0, 15]")
        for name, w in (("w1", self.w1), ("w2", self.w2), ("w3", self.w3)):
            if np.any((w < -1) | (w > 1)):
                raise Pack5FaultError(f"Non-ternary weight in {name}")

    @property
    def in_dim(self) -> int:
        return int(self.w1.shape[1])

    @property
    def h1(self) -> int:
        return int(self.w1.shape[0])

    @property
    def h2(self) -> int:
        return int(self.w2.shape[0])

    @property
    def out_dim(self) -> int:
        return int(self.w3.shape[0])

    @property
    def total_trits(self) -> int:
        return int(self.w1.size + self.w2.size + self.w3.size)

    @property
    def zero_fraction(self) -> float:
        zeros = int(np.sum(self.w1 == 0) + np.sum(self.w2 == 0) + np.sum(self.w3 == 0))
        return zeros / float(self.total_trits)

    def to_bytes(self) -> bytes:
        all_trits = np.concatenate([
            self.w1.astype(np.int8).reshape(-1),
            self.w2.astype(np.int8).reshape(-1),
            self.w3.astype(np.int8).reshape(-1),
        ])
        packed_weights = pack5_encode(all_trits)
        biases_bytes = (
            self.b1.astype("<i2").tobytes()
            + self.b2.astype("<i2").tobytes()
            + self.b3.astype("<i2").tobytes()
        )
        # Header: magic(4s) ver(H) in_dim(H) h1(H) h2(H) out_dim(H) s1(B) s2(B)
        #         trit_count(I) packed_len(I) bias_len(I)
        header = struct.pack(
            "<4sHHHHHBBIII",
            CARTRIDGE_MAGIC,
            CARTRIDGE_VERSION,
            self.in_dim,
            self.h1,
            self.h2,
            self.out_dim,
            self.s1,
            self.s2,
            self.total_trits,
            len(packed_weights),
            len(biases_bytes),
        )
        body = header + biases_bytes + packed_weights
        crc = zlib.crc32(body) & 0xFFFFFFFF
        return body + struct.pack("<I", crc)

    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = self.to_bytes()
        p.write_bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()
        p.with_suffix(p.suffix + ".sha256").write_text(f"{digest}  {p.name}\n", encoding="utf-8")
        return digest

    @classmethod
    def from_bytes(cls, blob: bytes) -> "TernaryCartridge":
        hdr_fmt = "<4sHHHHHBBIII"
        hdr_size = struct.calcsize(hdr_fmt)
        if len(blob) < hdr_size + 4:
            raise Pack5FaultError("Cartridge too short")
        body, crc_bytes = blob[:-4], blob[-4:]
        (expected_crc,) = struct.unpack("<I", crc_bytes)
        actual_crc = zlib.crc32(body) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise Pack5FaultError(
                f"Cartridge CRC32 mismatch: expected 0x{expected_crc:08x}, got 0x{actual_crc:08x}"
            )

        magic, ver, in_dim, h1, h2, out_dim, s1, s2, trit_count, packed_len, bias_len = (
            struct.unpack(hdr_fmt, body[:hdr_size])
        )
        if magic != CARTRIDGE_MAGIC:
            raise Pack5FaultError(f"Bad cartridge magic: {magic!r}")
        if ver != CARTRIDGE_VERSION:
            raise Pack5FaultError(f"Unsupported cartridge version: {ver}")
        expected_trits = in_dim * h1 + h1 * h2 + h2 * out_dim
        if trit_count != expected_trits:
            raise Pack5FaultError(f"Trit count mismatch: header={trit_count}, dims={expected_trits}")
        expected_bias_len = 2 * (h1 + h2 + out_dim)
        if bias_len != expected_bias_len:
            raise Pack5FaultError("Bias section length mismatch")
        if hdr_size + bias_len + packed_len != len(body):
            raise Pack5FaultError("Cartridge body length mismatch")

        off = hdr_size
        b_arr = np.frombuffer(body[off : off + bias_len], dtype="<i2").copy()
        off += bias_len
        b1 = b_arr[:h1]
        b2 = b_arr[h1 : h1 + h2]
        b3 = b_arr[h1 + h2 :]

        trits = pack5_decode(body[off : off + packed_len], trit_count)
        n1 = in_dim * h1
        n2 = h1 * h2
        w1 = trits[:n1].reshape(h1, in_dim)
        w2 = trits[n1 : n1 + n2].reshape(h2, h1)
        w3 = trits[n1 + n2 :].reshape(out_dim, h2)
        return cls(w1=w1, b1=b1, s1=s1, w2=w2, b2=b2, s2=s2, w3=w3, b3=b3)

    @classmethod
    def load(cls, path: str | Path) -> "TernaryCartridge":
        return cls.from_bytes(Path(path).read_bytes())
