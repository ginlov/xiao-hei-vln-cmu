"""ROS 2 adapters.

`rclpy` is imported lazily inside `subscribers`/`publishers` so the
core message contract remains importable in a plain Python venv.
"""
