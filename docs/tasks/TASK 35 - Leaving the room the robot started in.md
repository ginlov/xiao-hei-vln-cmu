# TASK 35 — Leaving the room the robot started in

TASK 33 gave the executor an ordered plan and TASK 34 taught it to drive a
passage. Both were measured on scenes where every destination was visible from
somewhere in one room. `home_building_1` q5 is not that scene: its first
destination is a nightstand at `(-1.90, -12.92)`, thirteen metres from the
start and through two doorways, and every attempt at it failed. This task is
about why, and it turns out the model was never the problem.

## What the failing runs actually showed

Leg 1 of `home_building_1` q5, twice, before any change here:

| | `hm1_q2_reserve` (1st) | `hm1_q2_reserve` (2nd) |
|---|---|---|
| ground covered | x∈[-5.1, 0], y∈[-2.0, 1.3] | x∈[0, 7.3], y∈[-2.9, -7.6] |
| ended because | first return to a visited pose | 9 steps used |
| time used / allotted | 183 / 400 s | 212 / 400 s |

Both legs died with more than three minutes of their own slice unspent, and
both died of a proxy for time rather than time itself.

The model's reasoning was sound throughout. Its `explore.why` named a visible
doorway on seven of nine calls — "the wide framed doorway on the left of image 0
opens into a bedroom-like room with a bed and lamp" — and the leg walked 20 m
for 7 m of net displacement without entering a single room.

## Four causes, each measured

**1. Circling detection killed the search.** `revisited()` was written for a leg
that had already bound its target and was walking the ring around it, where
coming back means there is nothing more to gain. In a multi-room search it means
the opposite: backing out of a dead end and returning to the hall to try another
door is what searching a building *is*. It now counts returns instead of ending
on the first one, up to `MAX_LOOPS = 3`, and tells the model it has looped.

**2. Nothing remembered which door had been tried.** The model re-reasons from
scratch on every call and kept nominating the door it had just come out of; the
terrain cannot tell two doors apart. `spent` records each departure as
`(from, unit direction)` and `already_tried()` matches within `REVISIT_M` and
`SPENT_CONE_DEG = 40°`. A repeat is scored down by `SPENT_PENALTY = 0.25`, not
forbidden — a room with one exit must keep that exit reachable.

**3. `reach · cos(Δ)` was structurally unable to leave a room.** This is the
important one. A doorway is a *short*-reach bearing: an 0.9 m opening onto a
small room stops the ray a couple of metres in, while the corridor beside it
runs five or six. Maximising reach therefore prefers the open space to the door
every time, by construction. In the nine-step leg above, five steps were swung
15–30° off what the model asked and all five landed on bearings reaching
4.4–6.8 m. Reach is now a **gate, not an objective**: among bearings clearing
`MIN_EXPLORE_M`, the one nearest what was asked wins. The old rule survives as a
fallback for a vehicle hemmed in on all sides, so loft's "heading 270° reaches
0.00 m" fix is not lost.

**4. A bearing cannot express "through that door".** Prompt **v6-way-out** adds
a `way` field — the opening, boxed like any other object — and `lift_way()`
places it with the same `_lift_xy` that already places targets, anchors and
gates. Built by surgery on v5, so v5 stays byte-for-byte exact; the 117 cached
replies and every offline script are keyed to it.

`--goto-steps` went 9 → 20, since the leg's share of `--budget` is the real
governor and the step cap is a safety valve.

## The cap that v6 needed immediately

The first v6 run used the way-out branch on **7 of 8** exploration steps — the
mechanism works — and drove the robot into a bedroom, where it inspected both
nightstands, found no clock, and correctly left. No previous run had entered any
room at all.

But four of seven lifts came back beyond 9 m, and the first, at 13.69 m, landed
at `(+8.57, -10.68)` — outside the entire extent the reference trajectory ever
visits, which is what aimed the leg east from its first move. The cause is the
thing that makes the field work close up: the scanner sees *through* an opening
and returns whatever stands behind it. Near to hand that is usefully past the
threshold; far away it is the room after next.

`WAY_MAX_M = 7.0` keeps the bearing and drops the range. Replaying the run's own
eleven lifts, seven are clamped and the one that drove the robot into the
bedroom (4.47 m) is untouched.

## Where this leaves the scenes

`home_building_2` q1, driven end to end against `trajectory_q4.ply`:

| | GT | ours | gap |
|---|---|---|---|
| destination 1 (magazine on ottoman) | reversal at (+0.83, +3.58) | bound (+0.58, +4.16) | 0.52 m |
| destination 2 (plant on dressing table) | end (-4.07, +11.66) | bound (-3.37, +10.57) | 1.30 m |

The robot stopped 0.95 m from its binding where the converter's best legal point
in the whole terrain frame was 0.92 m — three centimetres off the platform's
floor, which `obstacleDisThre` and the vehicle footprint set. The reference
trajectory's own endpoint is 1.30 m from that binding, so the run parks closer
to the object than GT does. **2/2.**

`home_building_1` q5 still fails leg 1, and the last run failed it in a new way
worth recording: the model matched the *feature* and dropped the *head noun*.
Asked for "the nightstand with a clock on it", it answered NOT_VISIBLE three
times in a row while looking straight at two lamp-topped nightstands — exactly
what branch (b) asks for — then found a clock on a TV sideboard and reported
`visible: true` at `target_type: nightstand`, binding 12.10 m from the right
answer. Its own evidence says "sideboard", "under the wall-mounted TV",
"bedside-**type**", and "the only clock" — a scarcity argument overriding the
noun. Branch (b) covers "right type, feature missing → NOT_VISIBLE" and says
nothing about the converse. Not fixed here.

## Files

- `scripts/approach_loop.py` — `lift_way`, `already_tried`, rewritten
  `explore_direction`, loop counting; `WAY_MAX_M`, `SPENT_CONE_DEG`,
  `SPENT_PENALTY`, `MAX_LOOPS`
- `scripts/vlm_probe.py` — `_make_v6`, `_WAY_BRANCH`, `_WAY_FIELDS`
- `scripts/execute_plan.py` — `--goto-steps` default
- `scripts/show_run.py` — new; reads a run directory back as one screen per step
- `tests/test_execute_plan.py` — +52 tests, on the recorded numbers above

## Still open

- **No responder.** Nothing here is submittable; a submission is a ROS node
  under `ai_module/` reacting to `/challenge_question`.
- The feature-over-noun false positive above.
- A destination thirteen metres away through two doorways is still unreached.
- `KEEPOUT_M = 1.2` refuted in TASK 31, still a disc rather than a corridor.
