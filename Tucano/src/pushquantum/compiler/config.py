"""Switches for the compiler. Each optimisation is a switch, so a regression is one line to undo.

History:
- v0  everything off: one atom per move, atoms go home after every pulse. Valid everywhere, official score 65.4%.
- v1a merge_moves on: many atoms per `move`. Official score 72.1%.
- v1b reuse + pick_best: atoms in consecutive pulses stay up; per circuit keep the cheaper. Official score 76.0%.
- v2a home="snake": Cuthill-McKee order laid out as a snake; pick_best also tries both home layouts. Score 76.7%.

REVIEW-COMMENT (why each step; maybe delete before submitting):
- v0 -> v1a: the validator scores the SUM over `move` instructions of the longest atom trip in each move. One move
  carrying many atoms costs only its longest trip, so merging moves (a + b -> max(a, b)) was the biggest lever.
  Putting separate moves in the same parallel step does not help: the distance still adds up (checked).
- v1a -> v1b: every atom went home after every pulse, even when its next gate was another CZ. Staying up saves a
  trip down and up. The validator forbids atoms waiting in the entangling zone during a pulse they are not part of,
  so only atoms of the NEXT pulse may stay. It helps iHVA but hurt some random circuits (sideways shifts cost up to
  1.7 and merge less well), so both versions are compiled and the cheaper one is kept.
- v1b -> v2a: the BFS home layout started from the busiest qubit, i.e. the middle of a chain, so HEA chains landed
  in storage as 1, 0, 2, 3, ... and the second round's pairs did not line up for one merged move. Cuthill-McKee
  starts from an end of the graph (a chain comes out as 0, 1, 2, ...), and the snake keeps the chain connected
  across rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DistModel = Literal["grid", "physical"]
Colouring = Literal["greedy", "cpsat"]
HomeMethod = Literal["bfs", "snake"]


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

    # v1b: after a pulse, atoms that are also in the next pulse stay in the entangling zone (only shift sideways).
    reuse: bool = False

    # v2a: initial loading. "bfs" = v0 order from the busiest qubit; "snake" = Cuthill-McKee order, snake rows.
    home: HomeMethod = "snake"

    # compile every combination of reuse (off/on) x home (bfs/snake), keep the cheapest (distance, then depth),
    # measured by our own copy of the metric (cost.py)
    pick_best: bool = True

    colouring: Colouring = "greedy"
    seed: int = 0


DEFAULT = Config()
