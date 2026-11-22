import numpy as np
import pytest

from ternara_gaze.pack5 import (
    MAX_VALID_PACK5_BYTE,
    Pack5FaultError,
    TernaryCartridge,
    pack5_decode,
    pack5_encode,
)


def test_pack5_roundtrip_all_lengths():
    rng = np.random.default_rng(123)
    for n in [0, 1, 2, 3, 4, 5, 6, 17, 255, 18560]:
        trits = rng.integers(-1, 2, size=n, dtype=np.int8)
        packed = pack5_encode(trits)
        assert len(packed) == (n + 4) // 5
        if packed:
            assert max(packed) <= MAX_VALID_PACK5_BYTE
        decoded = pack5_decode(packed, n)
        assert np.array_equal(trits, decoded)


def test_pack5_rejects_reserved_bytes_and_invalid_trits():
    for bad in range(243, 256):
        with pytest.raises(Pack5FaultError):
            pack5_decode(bytes([bad]), 5)

    with pytest.raises(Pack5FaultError):
        pack5_encode([0, 1, 2, -1, 0])

    with pytest.raises(Pack5FaultError):
        pack5_encode([0, -2, 1])

    # Non-zero trailing padding trit must fault
    packed = pack5_encode([1, 1, 1, 1, 1])
    with pytest.raises(Pack5FaultError):
        pack5_decode(packed, 3)


def test_cartridge_roundtrip_and_crc_fault(tmp_path):
    rng = np.random.default_rng(7)
    cart = TernaryCartridge(
        w1=rng.integers(-1, 2, size=(64, 256), dtype=np.int8),
        b1=rng.integers(-200, 200, size=64, dtype=np.int16),
        s1=5,
        w2=rng.integers(-1, 2, size=(32, 64), dtype=np.int8),
        b2=rng.integers(-200, 200, size=32, dtype=np.int16),
        s2=4,
        w3=rng.integers(-1, 2, size=(4, 32), dtype=np.int8),
        b3=rng.integers(-100, 100, size=4, dtype=np.int16),
    )
    out = tmp_path / "test.tgaze.bin"
    digest = cart.save(out)
    assert len(digest) == 64

    reloaded = TernaryCartridge.load(out)
    assert reloaded.sha256() == digest
    assert np.array_equal(reloaded.w1, cart.w1)
    assert np.array_equal(reloaded.w2, cart.w2)
    assert np.array_equal(reloaded.w3, cart.w3)
    assert np.array_equal(reloaded.b1, cart.b1)
    assert np.array_equal(reloaded.b2, cart.b2)
    assert np.array_equal(reloaded.b3, cart.b3)

    # Corrupt one byte and ensure CRC32 rejects it
    raw = bytearray(out.read_bytes())
    raw[40] ^= 0x01
    with pytest.raises(Pack5FaultError):
        TernaryCartridge.from_bytes(bytes(raw))
