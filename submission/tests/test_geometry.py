from __future__ import annotations

import pytest

from pushquantum.compiler.geometry import default_device
from pushquantum.compiler.ir import CZRound

dev = default_device()


def test_zones():
    assert dev.zone_of((0, 0)) == "storage"
    assert dev.zone_of((9, 9)) == "storage"
    assert dev.zone_of((0, 10)) == "entangling"
    assert dev.zone_of((10, 0)) is None
    assert dev.zone_of((0, 20)) is None


def test_site_lists():
    assert len(dev.storage_sites) == 100
    assert dev.storage_sites[0] == (0, 9)
    assert len(dev.pair_sites) == 50
    assert dev.pair_sites[0] == (10, 0)


def test_pairs():
    assert dev.pair_slots((10, 1)) == ((2, 10), (3, 10))
    assert dev.pair_of((3, 12)) == (12, 1)
    assert dev.pair_of((3, 9)) is None
    assert dev.is_pair((2, 10), (3, 10))
    assert not dev.is_pair((1, 10), (2, 10))
    assert not dev.is_pair((2, 9), (3, 9))


def test_distances():
    # validator uses physical distances (checked: storage col 5 -> entangling col 5 = sqrt(1 + 0.7^2) = 1.221)
    assert dev.dist((0, 9), (0, 10)) == 1
    assert dev.dist((5, 9), (5, 10)) == pytest.approx(1.2207, abs=1e-3)
    assert dev.dist((1, 10), (2, 10)) == pytest.approx(1.7)
    assert dev.dist((1, 10), (2, 10), model="grid") == 1
    assert dev.position((3, 10)) == pytest.approx((2.3, 10))


def test_round_rejects_shared_qubit():
    with pytest.raises(ValueError):
        CZRound(pairs=[(0, 1), (1, 2)], block=0)
