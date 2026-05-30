# About the challenge

The CMU Vision-Language-Navigation Challenge combines computer vision and natural language understanding for navigation autonomy. The challenge aims to push the limits of embodied AI in real environments and on real robots — providing a robot platform and a working autonomy system to move higher-level reasoning and learning models closer to real-world deployment.

The challenge requires developing a model that takes natural language queries or commands about a scene and generates appropriate navigation responses by reasoning about semantic and spatial relationships. The environment is initially unknown, so the system must navigate to informative viewpoints to discover and validate spatial relations and object attributes. A real-robot platform equipped with a 3D LiDAR and a 360° camera is provided and includes onboard base autonomy that estimates sensor pose, analyzes terrain, avoids collisions, and navigates to waypoints. For 2026, the challenge will run first in a custom simulation environment and transition to the real-robot system in a later phase.

# Tasks

## Task 1 — Consolidate input/output format for the VLM model

Because we communicate with a ROS system where different sensors publish data at different frequencies, we need to consolidate the VLM model's input and output formats to:

- Standardize the data format.
- Synchronize or resample sensor data to a unified frequency for the VLM pipeline.

This task includes three phases:

- Run examples to determine current sensor data formats and publishing frequencies; consult the challenge repository and website for documentation.
- Propose the input/output formats and the execution frequency for the VLM system.
- Implement the proposed formats as Python data classes in this repository for development and testing.

## Task 2 - End-to-end integrate to docker container

This task is to enable core AI module development, now input and output format are standardzed, I want to build docker container from this project in order to:

- Make use of new input/output format.
- Build my own dummy VLM which is later replace by real VLM.

This task includes three phases:

- Investigate how dummy container in the challenge repo was built.
- Plan of building a dummy system for this repo so that I get use it from now.
- Implement it and write report on how to use, how to build, how to integrate VLM later and how to replace the ai module container in challenge repo by that container.

## Task 3 - Integrate Qwen3.5VL for testing
This task is to integrate Qwen3.5VL into the model to get initial results.

This task includes three phases:

- Research the proper library, framework to run Qwen3.5 VL for this challenge since we are gonna receive data frequently depends on the frequency we set.
- Implement that serving service.
- Design the prompt for the type one question which is numerical question, when we receive the question, Qwen model need to return the waypoint for the robot to navigate and also at the end, answer that numerical questions.

### Task 3.1
It seems like the dependencies of vllm are too different from the ROS system. Setting up a separate container for vllm only is better approach. For this task.
- Let set up a new container for Qwen3.5 VL only.
- Integrate the system so that we just need to start all in one.