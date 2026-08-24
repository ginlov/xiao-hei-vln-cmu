# TASK 56 — The scorer was reading the noun and ignoring the clause

Found by looking at a picture, not by running a test. The trajectory atlas drew
`chinese_room` q4 — *"Go near the potted plant on the table and stop at the
painting near the TV"* — and the run our own table scored **6.00/6** had driven
to a potted plant on the far side of the room, 5.2 m from the one the organisers'
reference path went to, and stopped short of it besides.

## What was wrong

`score_if.head_label` reduces a referring expression to its head noun so it can
be matched against the scene annotation's label set. *"the potted plant on the
table"* becomes `potted plant`. Every box carrying that label then satisfies the
constraint. The relative clause — which is the entire job the challenge is
setting — never enters the test.

This is not a rounding error in these scenes. Of **75 credited destinations
across the 52-run coverage corpus, 54 (72%) are in scenes holding more than one
object of the named class**, so for three destinations in four the clause is the
only thing that picks a referent.

## The fix, and why it is a second reading rather than a replacement

There is no ground truth for "which instance did the sentence mean" — the
annotations carry labels and boxes, not referents. But the organisers shipped a
reference trajectory per question, and it *does* pick one, by going there.

`reference_pins(question, scene, qi, by_label, labels)` returns, per ambiguous
GOTO, the single instance whose box their reference path approaches most closely.
`score_trajectory(..., pin=pins)` then restricts that constraint to it.
`paper_table.py --strict-instance` and `export_traj_figures.py` both use it.

Two honest limits, both stated wherever the number appears: a reference path can
brush a rival instance on its way past, and the pin only exists for questions
that ship a reference trajectory. It is, however, independent of anything we
built — which is the property that matters, since the alternative was our own
grounder grading its own homework.

## What it costs

| | strict instance | any instance |
|---|---|---|
| mean over 52 runs | **3.77 / 6** | 4.29 / 6 |
| full marks | 19 / 52 | 25 / 52 |
| pass 1 / pass 2 | 3.808 / 3.731 | 4.269 / 4.308 |
| identical across passes | 18 / 26 | 19 / 26 |
| missed destinations | 36 | 25 |
| of those, within 0.5 m of passing | **22%** | 44% |
| worst miss | **8.2 m** | 4.2 m |

11 of 52 runs move. **11 of 75 credited destinations (15%) were credited on a
different instance than the reference's.**

The row that changed a conclusion is the last two. Under the loose reading the
misses pile up against the tolerance and trail off at 4.2 m, which reads as *the
platform's standoff is the ceiling* — a limitation that is not ours. Under the
strict reading half the misses are in the tail, and a tail miss is a grounding
failure and ours. **Ignoring the relative clause does not merely inflate the
score; it relocates the blame**, and it relocates it away from us. §6.3 now says
so in those words.

## Shipped

* `scripts/score_if.py` — `reference_pins`, and `pin=` on `score_trajectory`.
* `scripts/paper_table.py` — `--strict-instance`.
* `scripts/export_traj_figures.py` — every run carries `score` and
  `score_strict`, every constraint carries `ok`/`ok_strict`, and every named box
  is tagged `role: "pinned" | "rival"` so the atlas can draw the difference.
* The atlas gained a strict/any-instance switch that redraws the tolerance
  rings, the constraint marks and the corpus mean:
  https://claude.ai/code/artifact/63b4a3f5-acc2-47a7-965e-5961a864d637
* `docs/paper/skeleton.md` §6.2, §6.3, §7 rewritten; `NEXT.md` updated. 679
  tests pass.

## What this is worth to the paper

A methodological point we can make honestly because we were caught by it: **a
proxy scorer should be published as trajectories a reader can disagree with, not
as a mean.** Both of this scorer's free parameters — the 1.25 m tolerance and
the instance rule — move the headline result by more than any mechanism in the
system does, and the second one was invisible in every table we had produced. It
took one drawn picture.

## Open

* The ordering false positive is still there: `livingroom_3` q5's own reference
  trajectory loses a mark to `order` at tau 1.25, with all three constraints
  `ok`. That is why tau is not widened to 1.50 even though our validation rule
  would license it.
* `score_if.py:200` warns on divide-by-zero when two PASS anchors coincide.
