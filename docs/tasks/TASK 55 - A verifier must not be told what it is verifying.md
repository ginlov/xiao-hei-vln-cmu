# TASK 55 — A verifier must not be told what it is verifying

Three things, in the order they happened: a crash that was costing whole
questions, a measurement that redirected the verifier design, and the verifier
itself — which failed in its first form for a reason worth the paper.

---

## 1. A leg running out of time destroyed the question's record

`home_building_1` q5 drove eleven steps on 2026-08-17 and kept none of them.

```
python3 /tmp/robot_io.py drive 6.2258 -3.3695 --timeout -26.4
subprocess.TimeoutExpired: ... timed out after -1.35 seconds
```

`Ctx.left()` returned `deadline - now`, which goes negative once a deadline has
passed. All four `drive_to` sites pass `min(something, ctx.left())` into
`subprocess.run(timeout=timeout + 15)`, so a negative value raises immediately,
nothing caught it, and `plan.json` — written last — was never written at all.

Two fixes, because either alone leaves a hole:

* `left()` clamps at zero. The crash becomes a drive with no time to make, and
  `out_of_time()` ends the leg cleanly on the next check. Nothing depended on
  the value being negative: expiry is detected by `out_of_time()`, and the only
  other reader prints it.
* `execute()` now appends into a list `main` owns, and `main` wraps the walk in
  `try/except` and writes `plan.json` either way, with a `crash` field and exit
  code 2. A `KeyboardInterrupt` is caught the same way, so a run stopped by hand
  also keeps its legs. TASK 52 noted this gap; it is now closed.

Regression tests in `TestExecuteKeepsWhatItDrove` drive a fake `run_goto` that
succeeds on leg 1 and raises on leg 2, and assert leg 1 survives.

## 2. The measurement that changed the design

TASK 54 built T1 — predicted range against odometry — and measured it offline at
~60% detection. `NEXT.md` then recorded that in live runs it almost never fires,
because the arbitration ladder reaches the branch it lives in on about 2% of
steps. That number held on the finalised corpus:

| situation | share of grounding steps |
|---|---|
| a rival reading disagrees and the cheap rungs cannot decide (**T1's home**) | **2.2%** |
| the lift is refused and the leg drives on an unmeasured binding | **21.3%** |

The second row is ten times the first, and **no comparison test can touch any of
it** — T1 needs a measured range to compare against, and there is none. The
shape is `jr5_p1` leg 3: the lift refused step after step, the leg circling and
giving up 2.40 m short of its own binding with nothing ever contradicting it.

Worth recording precisely, because an earlier note here got it wrong: that leg's
binding was **not** demonstrably wrong. The run scores 6.00/6 — the trajectory
passed within tolerance of the real object while the loop believed it had
failed. What the case shows is a leg spending its budget with no measurement
behind it, which is the cost the verifier is meant to cut, not a wrong binding
the verifier would have caught.

So the verifier was gated on *that*, not on disagreement. `should_verify` fires
on three triggers, ordered by how blind the geometry is:

| trigger | condition | measured rate |
|---|---|---|
| `carried` | 2+ consecutive steps driven on a binding no lift could measure | 5.6% |
| `rejected` | a reading landed >1 m away and the ladder kept the binding | 2.2% |
| `arriving` | arrival declared on a binding no measurement confirmed | 6.7% |

Union, measured over 560 grounding steps in 29 recorded runs: **5.2%** — one
extra call per twenty steps, not per step. The wall clock is the real budget and
it is not doubled.

## 3. The first design did not work, and the reason is the finding

`verify_binding` projects the binding into the current view — `where_is` inverts
`cam_dir_to_map`, round-tripped to 1e-16 in tests — and asks the model whether
the phrase is at that place. It may only *refute*; a refutation drops the
binding and the next step re-grounds. It can never nominate.

`verify_audit.py` replays it over recorded runs: no simulator, one call per
fired step, and the binding graded by distance to the nearest ground-truth
instance of the phrase's head noun, read from the scene annotation rather than
from our own scorer. The gate fired 29 times, on 13 bindings that were right and
13 that were wrong — a balanced set by accident, not design.

**It refuted right bindings more often than wrong ones.**

| | anchored | blind |
|---|---|---|
| said `holds` on a **wrong** binding | **5/10** | **0/10** |
| said `holds` on a right binding | 5/10 | 2/10 |
| said `refuted` on a wrong binding | 2/10 | 3/10 |
| said `refuted` on a right binding | 4/10 | 2/10 |
| abstained | 4/20 | 13/20 |

The mechanism is visible in one cut. Under the first design the model said
`holds` on **83%** of bindings more than 5 m away against **35%** within 5 m: it
agreed *more* the less it could see. The prompt stated the belief — *the robot
is driving to a position it believes is "the soccer ball near the couch"* —
before asking whether it held, and at 9 m the model reported seeing a soccer
ball that was 3.30 m from where it was looking.

The fix is to blind the eye. `blind_verdict` splits the question in two: one
call looks at a vertical strip of the face around the projected column and is
asked only *what is there*, never what it should find; a second, text-only call
matches that description against the phrase. Two calls, the second cheap.

That change takes false confirmation of a wrong binding from **5/10 to 0/10**
(Fisher exact p ≈ 0.03) and moves refutation into the right direction. It does
not make the verifier a good detector: 3/10 wrong bindings refuted, and n is 10
per cell. Abstention rises to 57%, which is the designed behaviour and not a
failure.

Two further honesties:

* **The threshold was not tuned.** `REFUTE_CONF` stayed at the 0.7 chosen before
  any of this was measured. On the audit both false alarms sit at exactly 0.60
  and all three true refutations at 0.62 or above, so 0.62 would score 3/3 with
  no false alarm — fitted to six points, and not adopted.
* **Two of the "false alarms" are label artefacts.** Both were the soccer ball
  seen through a glass door onto the lawn; the description named a soccer ball
  *on grass*, the phrase says *near the couch*, and the verifier refuted. The
  head-noun label calls that a false alarm because it only matches the noun. The
  verifier was arguably right and the label is the crude instrument.

## 4. Two claims corrected against the data

Both were in the draft and both are now wrong in the file only where this
records them.

**"2x run-to-run variance"** was written as though it were the norm. It is not.
The full two-pass corpus (26 questions driven twice, 52 runs) says **19 of 26
scored identically** and 7 moved, usually by 3 of 6 marks, while the **corpus
mean moved 0.038** (4.269 against 4.308). Per-question reproducibility is poor
and corpus-level reproducibility is excellent -- roughly a factor of eighty
apart -- which is a reason to buy coverage rather than repetition.

**A leg's verdict is not the rubric's.** `jr5_p1` scores 6.00/6 while its own
leg 3 reports "circling 2.40 m short of the binding". The loop was wrong about
failing. Destinations-reached and proxy score measure different things and
neither may be quoted for the other.

### What this is worth to the paper

Not "we built a verifier". The transferable claim is about how a foundation
model must be *asked*: **a verifier that is told the hypothesis will confirm
it, and the confirmation gets stronger exactly where the evidence gets
weaker.** That is measurable, has a mechanism, and costs one prompt-level
change to fix. It also sharpens the existing abstention claim — blinding is what
makes the abstention honest rather than polite.

## Shipped

* `scripts/verify_binding.py` — the verifier, both styles, default `blind`,
  default **off** (`XIAO_HEI_VERIFY=1`). Never raises; a failed call leaves the
  binding untouched.
* `scripts/verify_audit.py` — the offline instrument. `--dry` prices the gate
  with no calls.
* `scripts/approach_loop.py` — the hook, the `carried_run` counter, `verified`
  written into the binding record, and `falsify`/`verify` into run settings so
  the arm is recoverable from the log rather than the directory name.
* `scripts/execute_plan.py` — crash-safe `plan.json`.
* `scripts/sweep_cv.sh` — the coverage sweep, pinned to the shipped
  configuration, which is the corpus the paper's main table is built on.
* `scripts/paper_table.py` — §6.2's table, §6.3's miss decomposition and the
  repeat spread, generated from the runs. Three of four questions were wrong the
  last time anything here was typed out of a log by hand.
* `tests/test_verify_binding.py` (36 tests), `TestExecuteKeepsWhatItDrove`.
  679 pass.

## Open

* The live arm has not been driven. The audit is the right instrument for the
  mechanism — it isolates it from the per-question spread — but the loop
  integration itself has only been unit-tested.
* `arriving` never fired in the audit: the recorded binding did not carry
  `verified` until this task added it, so the reconstruction could not see it.
  The next corpus will.
