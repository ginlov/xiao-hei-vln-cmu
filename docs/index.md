# Xiao Hei VLN

Team Xiao Hei's VLM stack for the [CMU Vision-Language-Navigation Challenge 2026](https://www.ai-meets-autonomy.com/cmu-vln-challenge).

## What is this?

A typed Python framework that sits between the challenge's ROS 2 sensor topics and a Vision-Language Model (VLM). It provides:

- **Pydantic data contracts** for every input (camera, lidar, terrain, odometry, question) and output (numerical answer, object reference, waypoint path).
- **A 2 Hz tick loop** that snapshots all sensors into a single `VLMInput` and publishes the model's `VLMOutput` to the correct ROS topic.
- **A pluggable responder interface** — swap models by implementing one method (`respond()`).
- **Docker infrastructure** that drops into the official challenge compose stack.
- **An offline evaluation pipeline** with numerical and object-reference metrics.
- **VLM tick logging** for post-run debugging with HTML report generation.

## Quick links

| What you want to do | Where to go |
|---|---|
| Run the system for the first time | [Quickstart](getting-started/quickstart.md) |
| Understand the architecture | [Architecture](architecture.md) |
| Understand the exploration phase | [Exploration Phase](concepts/exploration.md) |
| Understand the frontier algorithm | [FrontierExplorer](concepts/frontier-explorer.md) |
| Add a new exploration strategy | [New Exploration Strategy](guides/new-exploration-strategy.md) |
| Plug in a new VLM | [New Model Guide](guides/new-model.md) |
| Configure parameters | [Configuration](getting-started/configuration.md) |
| Run evaluation | [Evaluation Guide](guides/evaluation.md) |
| Generate the training corpus | [Data Generation](guides/data-generation.md) |
| Explore the dataset (distributions, quality) | [Dataset EDA](eda_report.md) |
| Debug a VLM run | [VLM Logging](guides/vlm-logging.md) |

## Project status

The system currently uses **Qwen3.5-4B** served via a vLLM HTTP sidecar. The responder implements multi-tick reasoning for numerical questions (counting objects from multiple viewpoints) and single-tick waypoint emission for instruction-following questions.

Active areas of development:

- VLM-based question classification (replacing keyword heuristic)
- Prompt engineering for object-reference questions
- Evaluation on VLA-3D generated datasets
