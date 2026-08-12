# TASK 38 — What an earlier leg already saw

`runs/exec_hb1_q2b`, step 6, leg 1, while the robot was looking for a
nightstand:

> middle of the kitchen floor: counter run with range hood and wall cabinets
> ahead, second counter run with window and **blue trash can** to the right,
> built-in oven/microwave column and **stainless fridge** behind, open glazed
> wood door to the left opening onto the living/dining hall

Leg 3's target is *"the trash can closest to the refridgerator"*. Both objects
were in one sentence, written unprompted, five steps before the leg that needed
them — and the sentence was thrown away. Leg 3 then bound a different bin on
every run: (+3.3, −10.9) once, (−5.4, −1.4) twice.

## Nothing carried a sighting, and the prompt forbade making one

`Ctx` kept five things across a leg boundary — `visited`, `avoid`, `spent`,
`crossed`, `done` — and none of them is "where the thing I will need later is".
The mission block said the opposite of what was wanted:

> Steps after it have not been attempted, so do not answer for them — you are
> being asked about step {k} only.

Worse, the one channel the information did arrive on was fed back with its sign
inverted. `here` goes into `VISITED_BLOCK`:

> PLACES ALREADY SEARCHED. Do not send it back to one of them unless the
> request can only be satisfied there and you say why.

So "the kitchen has a blue trash can and a stainless fridge" reached leg 3 as
*the kitchen has been searched, prefer somewhere else*.

It is not one run. Searching every recorded run for a later leg's noun in an
earlier leg's `here`: `exec_chinese_room` step 1 named the plant-on-table and
where it stood; `o2_001` step 1 named the only potted plant in sight and which
desk row it was beside. The model volunteers this without being asked.

## What changed

- **The prompt asks for it.** One exception to "step {k} only": an object a
  *later* step names, seen in passing, goes in `sightings` as
  `{step, what, image_index?, box_2d?}`. `what` is written for a reader who
  cannot see the image and will arrive from somewhere else, so it names the
  thing and what it stands next to. An empty list is the ordinary answer, and
  the prompt says to report only what is actually visible, not the room the
  object is probably in.
- **`Ctx.sightings` carries them.** Filed only for steps *after* the current
  one — a sighting of the current step is just the answer, and belongs in
  `box_2d` where the rest of the loop can see it. Deduplicated by
  `same_thing`, so the same bin seen on four calls is one lead.
- **Fed back on the leg it was for, as a lead.** `SIGHTINGS_BLOCK` is
  deliberately not merged into `VISITED_BLOCK`: one says go there and the other
  says do not, and on `home_building_1` they were the same sentence.
  `mission_for(k)` filters to step `k`, because a lead for step 5 shown on step
  2 is exactly the chasing the rest of the block forbids.
- **A lifted sighting steers, and only steers.** When the target is not in
  sight and nothing is boxed as a `way`, a sighting that carries a coordinate
  fills that slot — drive that way and look again — under the same `WAY_MAX_M`
  cap, and never as a binding. Today's measurements are the reason for the
  restraint: a lift from 11 m put the soccer ball 3.42 m from the truth and
  defended itself for the rest of the leg (TASK 37).

## Cost

113 input tokens when a lead is present, and an output field that is usually
`[]`. Against 29 s a step, and against a leg that has never once reached its
destination on `home_building_1`, that is not a number worth optimising.

604 tests pass, 8 new.

## Not measured yet

Whether it helps. The mechanism is tested and the evidence that the information
exists is four runs deep, but no run has yet been driven with it. The thing to
watch on `home_building_1` q2 is leg 3: today it binds a different trash can
every time, and the sighting should make it the same one — the one in the
kitchen, next to the fridge.
