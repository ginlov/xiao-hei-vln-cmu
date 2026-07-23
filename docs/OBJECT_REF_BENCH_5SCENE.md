# Object-reference 5-scene bench (100 questions)

Frozen GT: first 10 ref + first 10 num per scene for
`studio`, `chinese_room`, `livingroom_3`, `office_2`, `home_building_1`
(50 + 50 = 100 questions). Same IDs scored on baseline and improved dumps.

## Freeze command

```bash
cd /home/ubuntu/workspace/aryan/worktrees/xiao-hei-object-ref
uv run python scripts/freeze_bench_gt.py \
  --scenes studio chinese_room livingroom_3 office_2 home_building_1 \
  --num 10 --ref 10 \
  --out artifacts/bench_5scene_100q/gt
```

## Baseline (frontier, spatial off)

_Pending — command and metrics filled after the run._

## Improved (NBV + spatial on)

_Pending — command and metrics filled after the run._
