"""The device: zones, sites, CZ pair sites and distances. Everything is read from the spec, nothing is hard-coded."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache, cached_property

from pushquantum.architecture import Architecture, ArchitectureSpec, Zone
from pushquantum.compiler.config import DistModel
from pushquantum.move import Site

PairSite = tuple[int, int]  # (row, k): the k-th pair of that row


def _contains(zone: Zone, site: Site) -> bool:
    x, y = site
    return zone.x[0] <= x <= zone.x[1] and zone.y[0] <= y <= zone.y[1]


@dataclass(frozen=True)
class Device:
    storage: Zone
    entangling: Zone

    @classmethod
    def from_spec(cls, spec: ArchitectureSpec | None = None) -> Device:
        arch = Architecture({}, spec)
        storage = arch.zones[arch.loading_zone]
        entangling = next(z for z in arch.zones.values() if z.pair_axis is not None)
        if entangling.pair_axis != "x":
            msg = f"Only pairs along x are supported, got {entangling.pair_axis!r}."
            raise NotImplementedError(msg)
        return cls(storage, entangling)

    def zone_of(self, site: Site) -> str | None:
        if _contains(self.storage, site):
            return "storage"
        if _contains(self.entangling, site):
            return "entangling"
        return None

    def in_storage(self, site: Site) -> bool:
        return _contains(self.storage, site)

    def in_entangling(self, site: Site) -> bool:
        return _contains(self.entangling, site)

    @cached_property
    def pairs_per_row(self) -> int:
        return (self.entangling.x[1] - self.entangling.x[0] + 1) // 2

    def _rows_to_entangling(self, row: int) -> int:
        lo, hi = self.entangling.y
        return 0 if lo <= row <= hi else min(abs(row - lo), abs(row - hi))

    @cached_property
    def storage_sites(self) -> list[Site]:
        """All storage sites, closest to the entangling zone first."""
        sites = [
            (x, y)
            for y in range(self.storage.y[0], self.storage.y[1] + 1)
            for x in range(self.storage.x[0], self.storage.x[1] + 1)
        ]
        return sorted(sites, key=lambda s: (self._rows_to_entangling(s[1]), s[0]))

    @cached_property
    def pair_sites(self) -> list[PairSite]:
        """All pair sites, rows closest to storage first."""
        lo, hi = self.entangling.y
        rows = sorted(range(lo, hi + 1), key=lambda r: min(abs(r - y) for y in self.storage.y))
        return [(row, k) for row in rows for k in range(self.pairs_per_row)]

    def pair_slots(self, pair: PairSite) -> tuple[Site, Site]:
        row, k = pair
        left = self.entangling.x[0] + 2 * k
        return (left, row), (left + 1, row)

    def pair_of(self, site: Site) -> PairSite | None:
        if not self.in_entangling(site):
            return None
        k = (site[0] - self.entangling.x[0]) // 2
        return (site[1], k) if k < self.pairs_per_row else None

    def is_pair(self, a: Site, b: Site) -> bool:
        pa = self.pair_of(a)
        return pa is not None and pa == self.pair_of(b) and a != b

    def position(self, site: Site) -> tuple[float, float]:
        """Physical (x, y). Entangling columns: pairs every 2·spacing, the two sites of a pair `pair_distance` apart."""
        x, y = site
        if self.in_entangling(site):
            z = self.entangling
            offset = x - z.x[0]
            px = z.x[0] * z.spacing + (offset // 2) * 2 * z.spacing + (offset % 2) * (z.pair_distance or 0.0)
            return px, y * z.spacing
        return x * self.storage.spacing, y * self.storage.spacing

    def dist(self, a: Site, b: Site, model: DistModel = "physical") -> float:
        if model == "grid":
            return math.hypot(a[0] - b[0], a[1] - b[1])
        (ax, ay), (bx, by) = self.position(a), self.position(b)
        return math.hypot(ax - bx, ay - by)


@cache
def default_device() -> Device:
    return Device.from_spec(None)
