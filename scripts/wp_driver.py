#!/usr/bin/env python3
"""Deterministic fixed-path waypoint driver for the A/B experiment.

Publishes a fixed trajectory to /way_point (map frame) and advances to the
next waypoint on pose-based arrival (or a per-waypoint timeout for
unreachable points). Robot motion is driven purely by the nav stack's
response to this identical command sequence, so two runs follow the same
path regardless of the ai_module tick rate. Exits when all waypoints done.
"""
import json, math, sys, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry

_d = json.load(open(sys.argv[1]))
WAYPOINTS = _d["waypoints"] if isinstance(_d, dict) else _d
ARRIVE_TOL = 0.7      # m — pose within this of commanded wp counts as arrived
ARRIVE_TICKS = 3      # sustained ticks near the wp before advancing
WP_TIMEOUT_S = 25.0   # give up on an unreachable wp after this
PUB_HZ = 5.0

class Driver(Node):
    def __init__(self):
        super().__init__("ab_wp_driver")
        self.pub = self.create_publisher(PointStamped, "/way_point", 5)
        self.sub = self.create_subscription(Odometry, "/state_estimation", self._on_pose, 10)
        self.pose = None
        self.idx = 0
        self.near = 0
        self.wp_start = time.monotonic()
        self.timer = self.create_timer(1.0 / PUB_HZ, self._tick)

    def _on_pose(self, msg):
        self.pose = msg.pose.pose.position

    def _tick(self):
        if self.idx >= len(WAYPOINTS):
            print("DRIVER_DONE all %d waypoints" % len(WAYPOINTS), flush=True)
            rclpy.shutdown(); return
        wp = WAYPOINTS[self.idx]
        m = PointStamped()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.point.x = float(wp["x"]); m.point.y = float(wp["y"])
        m.point.z = float(self.pose.z) if self.pose is not None else 0.75
        self.pub.publish(m)
        now = time.monotonic()
        if self.pose is not None:
            d = math.hypot(self.pose.x - wp["x"], self.pose.y - wp["y"])
            if d <= ARRIVE_TOL:
                self.near += 1
            else:
                self.near = 0
            arrived = self.near >= ARRIVE_TICKS
            timed_out = (now - self.wp_start) >= WP_TIMEOUT_S
            if arrived or timed_out:
                print("WP %d/%d %s at (%.2f,%.2f) target(%.2f,%.2f) d=%.2f" % (
                    self.idx + 1, len(WAYPOINTS),
                    "reached" if arrived else "timeout",
                    self.pose.x, self.pose.y, wp["x"], wp["y"], d), flush=True)
                self.idx += 1; self.near = 0; self.wp_start = now

def main():
    rclpy.init()
    n = Driver()
    try:
        rclpy.spin(n)
    except Exception:
        pass

if __name__ == "__main__":
    main()
