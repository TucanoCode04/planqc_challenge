"""Switches for the compiler. Defaults = safest settings (v0); optimisations are turned on one at a time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DistModel = Literal["grid", "physical"]
Colouring = Literal["greedy", "cpsat"]


@dataclass(frozen=True)
class Config:
    dist_model: DistModel = "physical"
    row_merge: bool = False
    parallel_moves: bool = False
    reuse: bool = False
    colouring: Colouring = "greedy"
    seed: int = 0


DEFAULT = Config()
