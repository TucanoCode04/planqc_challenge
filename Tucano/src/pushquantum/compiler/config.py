"""Switches for the compiler. Each optimisation is a switch, so a regression is one line to undo.

History:
- v0  everything off: one atom per move, atoms go home after every pulse. Valid everywhere, official score 65.4%.
- v1a merge_moves on: many atoms per `move`. Official score 72.1%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DistModel = Literal["grid", "physical"]
Colouring = Literal["greedy", "cpsat"]


@dataclass(frozen=True)
class Config:
    # "physical": the validator measures real positions (entangling cols at 0, 0.3, 2, 2.3, ...), checked by experiment.
    # "grid" (1 per cell) was the first assumption in v0; kept only to compare.
    dist_model: DistModel = "physical"

    # v1a: many atoms per `move`. Scored distance = sum over moves of the longest trip in each move.
    merge_moves: bool = True

    # v0 plan, NOT USED anymore (kept for the record):
    # - row_merge: merge only atoms on the same row. Replaced by merge_moves, which merges any legal rectangle.
    # - parallel_moves: put independent moves in the same parallel step. Experiments showed the validator does group
    #   them into one step, but the scored distance still adds every move, so steps only change time, not the score.
    # row_merge: bool = False
    # parallel_moves: bool = False

    reuse: bool = False
    colouring: Colouring = "greedy"
    seed: int = 0


DEFAULT = Config()
