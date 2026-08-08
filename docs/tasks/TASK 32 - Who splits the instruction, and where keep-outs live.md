# TASK 32 — Who splits the instruction, and where keep-outs live

Instruction-following is the only question type this team is now working on: it
is **6 of the 17 points a scene carries twice over** — 2 questions × 6 points
against 1 numerical and 2 object-reference — so across the 3 held-out
evaluation scenes it is **36 of the 51 points on offer**. The other two types
are being handled separately.

The executor that walks a question's clauses is the next thing to build. Before
building it, one question had to be settled: **who turns the sentence into
clauses.**

## The two candidates

`scripts/instruction_plan.py` (TASK 31) does it with a regex. Its own docstring
argues for that, and the argument is sound as far as it goes: every training
question is available offline, so a deterministic parser can be measured on all
thirty for free, where a model call can only be measured by spending it.

The objection is about what it was measured *against*. `_START` is a closed set
of ten clause openers — `go to`, `go near`, `stop at`, `stop by`, `go`,
`take the path`, `go between`, `pass by`, `pass through`, `avoid` — read off
those same thirty sentences. The evaluation scenes are held out and README only
promises questions "of similar style". `head toward`, `make your way past`, or
`keeping the sofa on your left` all fall through to the single-`GOTO` fallback.

That fallback used to cost nothing. While the loop drove to one object and
stopped, collapsing a sequence to its first destination lost nothing that was
going to be attempted. With an executor it is no longer free: README §175
scores the trajectory on whether it "follows the path constraints in the
command and in the correct order", so a question that degrades to one `GOTO`
forfeits every ordering point it carried.

## What was measured

`scripts/decompose.py` asks a model for the same `Clause` objects, making it a
drop-in, and keeps `parse_instruction` as the fallback for an unparseable
reply. `--diff` runs both over the official thirty and compares.

The prompt's two worked examples are the README's own illustrations, chosen
deliberately from outside `questions.json`: seeding it with two of the thirty
would have scored the prompt against its own answer key.

| | |
|-|-|
| same drive order | **30 / 30** |
| identical clauses | 28 / 30 |
| model reply unusable | 0 / 30 |
| cost | 3.3 s per question, one call — 0.6% of the 600 s budget |

The two remaining differences are `text` phrasing on a `PASS` clause (whose
geometry reads `anchors`, not `text`, and those matched), and `pass by the
stairs`, where the model recovered `relation="near"` and the regex did not:
`by` lives inside the opener `pass by`, is stripped with it, and `default_rel`
was only ever wired for `between`.

**Decision: `decompose` is the primary path, `parse_instruction` the fallback.**
`decompose` returns `(clauses, from_model)` so a silent degradation mid-run
cannot corrupt a later measurement.

## The finding that mattered more

The first diff disagreed on the three keep-out questions, and the disagreement
turned out not to be about parsing at all.

`parse_instruction` emits clauses in the order the words appear. Two of the
three keep-outs are phrased `..., then stop at B, avoiding the path between X
and Y` — written last, but governing the drive *toward* B. A forbidden region
that only switches on once the robot has parked forbids nothing, so a fix was
added to hoist a trailing `AVOID` ahead of the last destination. It repaired
those two and broke the third:

    Go to the cup near the TV remote and avoid the path near the cabinet.

One destination, so there is no "before" to hoist into.

Both orderings were attempts to encode *when the constraint switches on* as a
position in a list, and the sentence never says. Nothing makes it start after A,
and there is not always an A. The keep-out is a constraint over the whole
trajectory, and README §175 penalises passing through a forbidden area with no
mention of when.

So `AVOID` was taken out of the sequence entirely:

- `steps(plan)` — destinations and passages, in drive order
- `keepouts(plan)` — the forbidden regions, unordered, active for the whole run

The hoist was reverted. With the comparison made against what the executor
actually consumes, the parsers agree 30/30 on drive order, and the ordering
question no longer exists to get wrong. This supersedes TASK 31's treatment of
keep-outs as positional clauses.

## Also corrected

`decompose` v1 read the preposition rather than the verb and returned `PASS`
for `Go near the stool under the picture` — a destination. The prompt now says
the verb decides: `go near` / `stop by` / `stop at` are destinations, and only
`take the path near` / `pass by` / `go between` are passages. Found by the
3-question smoke run before the full sweep.

## State

- 441 tests pass, 1 skipped.
- Question shapes unchanged: 30 questions → 59 destinations, 13 passages,
  3 keep-outs; 27 of 30 need two or more destinations in order.
- Nothing consumes `steps` / `keepouts` yet. The executor is the next task, and
  after it the responder wrapper — **no part of the current VLM stack is
  submittable**: `approach_loop.py` is driven over SSH from outside the
  container, while a submission is a ROS node under `ai_module/` reacting to
  `/challenge_question`.
