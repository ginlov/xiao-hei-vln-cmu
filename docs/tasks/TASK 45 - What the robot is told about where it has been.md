# TASK 45 — What the robot is told about where it has been

`VISITED_BLOCK` gives the model back its own `here` clauses from earlier calls,
under a heading that says *"Do not send it back to one of them… Prefer
somewhere it has not stood."* It is expensive twice over: `here` is 19.7% of
the characters the model writes, and the block is, a few steps in, the largest
thing in the prompt that prompt caching can never touch, because it changes on
every call.

This task measured whether it works, and put the alternatives behind a flag.


## Measuring it

**Observational, over 93 recorded runs.** For every `NOT_VISIBLE` step with an
explore heading and at least two places already stood (n=87), where the heading
lands relative to the nearest of them, against the same step scored over all 24
headings:

```
model's heading   median 0.94 m from the nearest place already stood
all 24 headings   median 0.94 m
model better on 44/87 (51%)      sign test p = 1.000
```

Chance. A second, geometry-independent cut agrees: the heading points *toward*
the centroid of everywhere the robot has stood on 55 of 85 steps (65%).

**That is not enough to conclude anything**, and the first version of this
report said "no measurable effect" on the strength of it. Observational data
cannot separate "the block does nothing" from "the block does something and
something else undoes it".

**Counterfactual, 123 API calls over 41 recorded steps.** Same step, same
faces, same previous crop, same mission, same model — the only difference is
whether the block is present. Any error in reconstructing the mission cancels,
because both arms get the same one.

A two-arm design would still not have been enough. Thinking is stochastic, so
two identical prompts already disagree, and that disagreement is what the
with/without difference has to beat. A third arm re-samples the *same* prompt
to establish it:

```
                     identical   median   mean   >45°   >90°
with vs WITHOUT            29%       7°    42°    29%    22%
with vs with (control)     44%       1°    25°    20%    12%

removing the block is more disruptive than re-rolling it:
   22/30 steps    sign test p = 0.016
```

**So the model reads it.** That refutes the observational reading.

What it does with it is the other half:

```
distance from the heading's landing point to the nearest place already stood
   with visited      1.73 m      (second sample of the same prompt: 1.79 m)
   without visited   1.67 m
   with vs without differ by 0.06 m
   the two with-samples differ by 0.06 m
```

The block moves the answer well past noise, and moves it nowhere in
particular — its advantage on the one thing it asks for is exactly the size of
the noise between two samples of an identical prompt.


## The flag

`--visited {prose,bearing,xy,off}` and `--drop-here`, on both
`approach_loop.py` and `execute_plan.py`, defaulting from `XIAO_HEI_VISITED`
so a sim run can be switched without editing anything.

| value | the block reads | 18-visit run |
|---|---|---|
| `prose` | the model's own `here` clauses — **the original** | 5471 chars |
| `bearing` | `5.6 m away at heading 211° (back-left)` | 600 chars |
| `xy` | `(+2.31, -1.44)` | 238 chars |
| `off` | nothing | 0 |

Whole prompt on that run: **4167 → 2859 tokens** under `bearing`, every one of
the 1308 off the uncacheable part. `--drop-here` takes ~100 more input tokens
and ~65 output tokens per call, and costs the run's best diagnostic.

### Bearings, not coordinates

The original docstring justified prose by arguing that map coordinates are
useless to something that reasons over images and has never seen the frame.
Half of that is right and the flag keeps it: `(2.31, -1.44)` names nothing the
model can act on. But the robot *has* stood in those places and we know exactly
where they are, and there is a frame both sides share — the headings that
number the four faces. "4.2 m at heading 211°" names one of the pictures in
front of it.

Headings therefore invert map yaw exactly as `explore_goal` does, so a heading
read out of the block and one handed back in `explore` mean the same thing.
Getting that sign backwards would tell the model to avoid the one direction it
should go, and nothing downstream would flag it; there is a round-trip test.

`xy` is kept as the control for the claim, not as a candidate.

### Rollback

`--visited prose` renders a **byte-identical** prompt to the code before the
flag existed — verified against `git show HEAD:scripts/vlm_probe.py` for the
bare, `+visited`, and `+visited +mission` shapes. The first version of that
check compared the new code against itself and passed while the block was
actually missing a blank line; comparing against git is what caught it.


## What this does not settle

- **Nothing here has been driven.** The measurements are on the model's reply,
  not on where the robot ended up. Which style is better is a sim question and
  that is what the flag is for.
- The counterfactual scores the block against **its own stated goal**. If
  `visited` helps some other way — better `here` prose, better `visible`
  judgements — this did not look. The replies are saved and cost nothing more
  to re-score.
- n = 41 steps from 20 runs, half the planned sample. p = 0.016 stands; the
  null on the avoidance metric means "effect ≤ noise", not "effect = 0".


## Changed

- `scripts/vlm_probe.py` — `APPROACH_HERE` split out of `APPROACH_BLOCK`;
  `VISITED_BLOCK_BEARING`, `VISITED_BLOCK_XY`, `VISITED_BLOCKS`;
  `build_prompt(visited_kind=…, ask_here=…)`.
- `scripts/approach_loop.py` — `VISITED_STYLE(S)`, `FACE_OF`;
  `Ctx.visited` now `{"xy", "text"}` records, with `note_visit`,
  `visited_for`, `prompt_visited_kind`; `ground(visited_kind=…, ask_here=…)`;
  both flags.
- `scripts/execute_plan.py` — both flags.
- `docs/guides/drive-loop-runbook.md` — the `--visited` section.
