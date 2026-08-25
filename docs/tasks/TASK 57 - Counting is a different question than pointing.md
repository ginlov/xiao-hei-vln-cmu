# TASK 57 — Counting is a different question than pointing

The numerical responder, built to the plan, plus the gate measurement that says
whether it was worth building. Also two submission-blocking defects found while
wiring it in, which had nothing to do with counting and would have cost the
whole run.

---

## 1. The shape of the question, read as a set

Fifteen released numerical questions, one per development scene. Read one at a
time they look like "count things in a room". Read together, **eleven of the
fifteen are anchor-local**: there is one piece of furniture and the count is of
small things on or above it. Two are scene-wide (`arabic_room`,
`chinese_room`); two turn on a determiner (`a sofa`, `a table`) and have two
defensible answers each.

That is the finding the design rests on. For eleven questions the robot does not
have to explore — it has to find one object, frame it, and count. Finding one
object is what `run_goto` already does 100 times a sweep.

## 2. The answer key, computed rather than typed

`scripts/score_numerical.py`. The challenge ships no answers, so the key comes
from the VLA-3D annotation the scenes were built from — same frame, same
objects as `viz/data`, checked object for object on `loft` where they agree to
1e-4 m, and with the colour columns `viz/data` drops.

The split is the point: **the reading of the English is declared, the count is
computed**. A human has to say which noun is the target and which relation
joins them, because there is nothing to check that against; no integer in the
file was typed.

A `--sens` sweep over the one free parameter — how far two footprints may miss
and still count as touching — is what grades the key's own confidence:

| reading | n | what it means |
|---|---|---|
| solid | 9 | the annotation answers it and the integer does not move with the pad |
| shaky | 4 | it answers under a judgement call, written down |
| open | 2 | two readings of the sentence give different integers |

The sweep demoted two questions that had been declared solid: `arabic_room`
gives 2 or 3 sofas depending on the pad, `home_building_1` 6 or 7 pillows.

**Colour turned out to be available, and half of it works.** The plan said the
annotation carried no colour; that was wrong. VLA-3D snaps every object to the
nearest CSS colour name with a percentage, which settles `home_building_2` —
exactly two pillows on the sofa, both `maroon 78%` (firebrick), so the answer to
"how many red pillows" is 2. It does **not** settle `loft`: eight pillows sit on
its sofas, seven `slategray`, one `darkolivegreen`, none `black`. The two
`darkslategray` (47,79,79) ones are dark enough that a person would say black,
which is where the key's 2 comes from and why it is marked shaky.

**Best constant guess: 3, which scores 4/15 (27%).** That is the bar.

## 3. The counting call

`scripts/count_view.py`. Two rules, and they are the design.

**It is never told the running total.** Not the tally, not what a previous view
counted. TASK 55 paid for this: the binding verifier, given its hypothesis in
the prompt, said `holds` on 83% of bindings beyond 5 m against 35% within 5 m —
agreeing more the less it could see — and falsely confirmed a wrong binding 5
times in 10; blinding took that to 0 in 10. So `count_view` takes no history
argument and there is nowhere to put one. A test asserts the signature stays
that way.

**The model counts pixels, the geometry counts objects.** Every instance comes
back as a box, every box is lifted by the same lidar cone the approach loop
uses, and the answer is the number of *clusters* over all views — never the sum.
`sufficient` may only withhold a commit; it can never raise the count.

## 4. The gate

`scripts/count_audit.py` replays the call over views the robot has already
taken — 175 of the sweep's 1,980 face images already frame a numerical anchor
within 5 m, so this costs no simulator time. Each view carries its own
expectation (how many counted objects project into a face from that pose), so a
robot in a doorway is not marked wrong for seeing three of four.

**The first run of this reported 71%, and that number was wrong.** It came from
five scenes whose answers are all 1-3, and it was compared against a
constant-guess baseline that needs no navigation at all. Extending the audit to
the six scenes it had skipped — the ones with answers of 3, 6, 6 and 8 — and
fixing a bug in the instrument (below) gives the real shape:

| | easy 5 (answers 1-3) | hard 6 (answers 3-8) | all 32 views |
|---|---|---|---|
| exact against the key | 10/14 = **71%** | 2/18 = **11%** | 12/32 = **38%** |
| exact on views seeing everything | 10/13 = 77% | **0/13 = 0%** | 10/26 = 38% |
| exact where 4+ objects are in frame | — | 1/11 = 9% | **1/11 = 9%** |
| said `sufficient` and was right | 7/9 = 78% | 1/7 = 14% | 8/16 = **50%** |
| **confident and wrong** | 2/14 = 14% | 6/18 = 33% | 8/32 = **25%** |
| instances the scanner could place | 28/31 | 53/56 | 81/87 = 93% |

**38% against a 27% constant-guess baseline, and 0 of 13 on full views of the
hard scenes.** The gate does not pass in the form it was set. What the model can
do is count one, two or three things; at four and above it is at 9%, and the
`sufficient` flag — which the whole commit rule leans on — degrades from 78%
right to 14% right and is a coin flip overall.

**The instrument had a bug and it had to be found before any of this was
readable.** `visible_from` filtered projected objects on `abs(across) < 0.42`,
reading `across` as a signed offset from the face centre. It is `u / FACE_SIZE`
in [0, 1), so 0.5 is the centre: the filter kept only the left 42% of each face
and discarded everything right of it. That under-counted `visible` on most
views, made "sees every counted object" mean "every object is in the left
half", and produced a 56% "over-count" rate that was almost entirely the metric.
`where_is` already returns None when a bearing falls in no face, so there was
nothing to filter in the first place. Corrected, the corpus holds 322 usable
views rather than 175, and 253 of them see every counted object.

Two things the easy half still said, and they survive:

**Abstention is well calibrated on small counts.** All three `japanese_room`
views answered 2 or 3 against a truth of 3, and *every one of them* said the
view was not sufficient, with the reason written out — "the rightmost darker
frame is too small/dim to confirm as calligraphy". It never claimed a wrong
number confidently. That is the designed behaviour, working.

**Both confident errors are the same failure, and it is the one the merge was
built for.** On `livingroom_2` the coffee table straddles two faces; the model
saw one mug in each and reasoned itself into two: *"their bearings differ enough
that they are two separate cups rather than one seen twice"*. The geometric
merge exists precisely to overrule that — and it could not, because **the
scanner placed 0 of those 2 boxes**. A mug is the smallest target in the set and
the lift is where it fails: 28 of 31 instances placed overall, and all three
failures are cups. De-duplication only protects what the lidar can reach.

(The second `livingroom_2` view describes "a white mug and a grey mug", which
may well be right against an annotation that labels one `cup`. Counted as an
error anyway — the key is the instrument and softening it to fit is how a
proxy stops measuring.)

## 5. Two defects that had nothing to do with counting

Both found by adding one line to `sync_ai_module.sh` that imports the synced
tree the way the container will.

**`verify_binding.py` was not in the sync list while `approach_loop.py` imported
it at module scope.** The submission would have died with `ModuleNotFoundError`
at launch — and instruction following is **36 of the 51 available points**, none
of which survive that. It stayed invisible only because the copy under
`ai_module/` had gone stale and predated the import; the copy that would have
shipped on the next sync is the one that breaks.

**`verify_binding.strip_of` used PIL, which the image does not install.** The
import was function-local, so it was a landmine rather than a crash: harmless
while the verifier is off, fatal the first time anyone sets `XIAO_HEI_VERIFY=1`
inside the container. Rewritten on cv2, which the Dockerfile does install.

The sync script now distinguishes the two cases — a missing module-level import
is fatal and stops the sync, a missing nested one is reported as a note.

## 6. Shipped

* `scripts/score_numerical.py` — the key, `--why <scene>` for the working.
* `scripts/count_view.py` — the blind counting call, the lift, the merge.
* `scripts/count_audit.py` — the gate; `--dry` prices it with no calls.
* `scripts/numerical_plan.py` — question → target/attribute/relation/anchor,
  model with a regex fallback, `--diff` to measure one against the other.
* `scripts/answer_numerical.py` — the live loop: find the anchor, frame it,
  look, reposition, commit.
* `challenge_node.handle_numerical` — no longer silent. It publishes, because
  once a responder exists silence can only lose the point: a wrong integer and
  no integer score the same 0.
* `classify.py` — default flipped from `stub` to `claude`. Routing a counting
  question to the drive loop guaranteed 0 of its 1 point.
* `scripts/sync_ai_module.sh` — the missing modules, and the import check.
* `tests/test_numerical.py`, 32 tests. **711 pass.**

## 6b. The first live runs, and what they moved

Two questions driven on the sim, 2026-08-22/23.

**`studio` — answered 3, correct, and it does not validate anything.**
`run_goto` timed out 2.58 m short of the couch, so `Outcome.xy` came back None,
the framing step never ran and repositioning had no anchor to work from. It
looked once from wherever it had stopped, counted three framed records and
committed. Right answer, failed navigation: exactly the confound that makes an
offline conditional untransferable.

It also exposed a defect **older than any of this work**: `robot_io.cmd_drive`
publishes its waypoint once and never retracts it, so when the process ends the
local planner still holds that goal. With an unreachable goal — which is what a
drive timeout means — the vehicle oscillates in place until the stack is
restarted. It is not cosmetic: instruction following is scored on the driven
trajectory, so a vehicle still moving after the run has ended is still writing
to the thing being marked, and can wander into a keep-out the run respected.
Fixed by publishing the current pose as the waypoint (`robot_io stop`,
`Robot.stop`, `RobotNode.stop`), called before the numerical loop reports and
before `challenge_node` idles.

**`office_1` — answered 3 against 6, and the diagnosis is not what it looked
like.** The suspicion was that the anchor had ignored the relative clause. It
had not: the split extracted `anchor_qualifier: "closest to the map wall
decal"`, `run_goto` was given the whole phrase, and both bindings landed 0.91 m
and 1.28 m from the correct table — the other one is 5-6 m away and was never
touched.

What went wrong is where it stood. That desk is **2.9 m long** and its six
monitors run 2.00 m along it. The vehicle finished at (+4.81, −3.54): past the
desk's far end and 1.7 m off its axis, viewing it end-on, with the near
monitors hiding the rest. All four looks said `too_far` or `occluded` and
**none of them claimed sufficiency** — the abstention was right and had nothing
to act on, because three separate things stopped the loop from fixing the view:

* `run_goto` ends on "no legal point closer", which is the correct answer to
  *approach* and the wrong one to *frame*;
* `anchor_offset` only backs off along the bearing the vehicle already has, so
  a corner viewpoint stays a corner viewpoint;
* the binding is a *point*. Nothing in the loop knew the table was 2.9 m long
  or which way it ran.

**Geometry says the framing was possible and we simply did not do it.** Angular
width of each question's counted set, viewed square-on from 2.6 m:

| | objects | spread | subtends |
|---|---|---|---|
| `office_1` | 6 | 2.00 m | 42° |
| `livingroom_1` | 8 | 2.41 m | 50° |
| `home_building_1` | 6 | 2.78 m | 56° |
| `chinese_room` | 6 | 6.45 m | **102°** — the only one that cannot fit |

Thirteen of fifteen fit inside a single 100° face. So "counting past three does
not work" is at least partly **"we never gave it a view holding more than
three"** — and the two are not the same defect. Sets with four or more members
average 2.99 m of spread against 1.11 m below four, so count and spread are
confounded in the offline result and it cannot separate them.

`frame_pose` decides the viewpoint from the instances already placed rather
than from a constant: centroid, long axis by SVD, and a standoff that makes the
extent subtend 70°, standing square to the axis on the side the vehicle is
already on. It fires once per run, only when the model reports `too_far`,
`occluded` or `anchor_cut_off`, and only when the move is real. On the
`office_1` numbers it returns **(+2.78, −3.72) — 8 cm from a pose that run had
already driven through** on its way to the corner it gave up in.

That is a prediction, not a result. `office_1` has to be re-driven, and if it
comes back 6 the offline 38% has to be re-measured rather than reinterpreted.

## 7. Open

* **Counting past three does not work — but the offline corpus cannot say
  whether that is the call or the viewpoint.** 1/11 where four or more objects
  are in frame. `arabic_room` answered 0 three
  times with `sufficient: true` against a truth of 3; `office_1` answered 3
  against 6 monitors it could all see. Nothing downstream fixes this — it is the
  call itself, and the next move is a prompt that makes the model enumerate and
  place before it totals, measured on the 253 full views the corrected corpus
  now offers.
* **`sufficient` does not survive the hard scenes** (78% right on small counts,
  14% on large). The commit rule leans on it, so it cannot be trusted as the
  only gate; a count that disagrees with the previous look is the other signal
  available and is not yet used.
* **Not yet driven live.** Every part is tested and the counting call is
  measured offline, but `answer_numerical` has never moved a robot. The framing
  stop rule (`VIEW_M = 2.6`) is derived from face geometry, not from a run.
* **The offline number is a conditional and cannot be multiplied out.** It is
  `P(right | already standing in front of the thing)`, measured on views the
  *instruction* sweep produced. For the navigation half the nearest number is
  the same `run_goto`, at 75% of GOTO legs reaching their object over the 52-run
  corpus (64% when it must be the reference's instance) — a harder test than
  this needs, so a floor rather than an estimate.
* **Small targets defeat the lift, and the lift is the de-duplication.** Cups
  placed 0/3. Either the cone needs widening for small boxes, or the merge needs
  a fallback that is not positional — and the honest version of the second is
  "take the maximum single view", which is what unplaced instances already get.
* `--diff` between the model split and the regex has not been run; the regex
  disagrees on `chinese_room` (it reads the anchor as "pillows on them"), which
  matters only when a reply is unparseable.
* The two `open` questions still have no policy. A decision has to be written
  down before the held-out scenes are seen, because there we will not get to
  look.
