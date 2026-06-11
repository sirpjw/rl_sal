#!/usr/bin/env python3
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

RAW_W = 106
RAW_H = 60
OUT_W = 87
OUT_H = 58
CLIP_MIN = 0.0
CLIP_MAX = 2.0
FX_PX = RAW_W * 11.041 / 20.955
FY_PX = RAW_H * 11.041 / 12.240
CX_PX = (RAW_W - 1) * 0.5
CY_PX = (RAW_H - 1) * 0.5


def image_to_depth(msg: Image):
    if msg.width != RAW_W or msg.height != RAW_H:
        return None
    if msg.encoding == "32FC1":
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape((RAW_H, RAW_W)).copy()
    elif msg.encoding == "16UC1":
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape((RAW_H, RAW_W)).astype(np.float32) * 0.001
    else:
        return None
    return depth


def z_depth_to_ray_depth(z_depth: np.ndarray) -> np.ndarray:
    u = np.arange(RAW_W, dtype=np.float32)
    v = np.arange(RAW_H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    x = (uu - CX_PX) / FX_PX
    y = (vv - CY_PX) / FY_PX
    return z_depth * np.sqrt(1.0 + x * x + y * y)


def preprocess_depth(raw_depth: np.ndarray) -> np.ndarray:
    z = raw_depth.copy()
    invalid = ~np.isfinite(z)
    z[invalid] = CLIP_MAX
    ray = z_depth_to_ray_depth(z)
    clipped = np.clip(ray, 0.0, CLIP_MAX)
    cropped = clipped[:-2, 4:-4]
    resized = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0.0, CLIP_MAX)
    return resized / CLIP_MAX - 0.5


def colorize_depth(depth: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    vis = depth.copy()
    vis[~np.isfinite(vis)] = max_value
    vis = np.clip(vis, min_value, max_value)
    vis = ((vis - min_value) / (max_value - min_value) * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(255 - vis, cv2.COLORMAP_TURBO)
    return vis


class DepthDebugViewer(Node):
    def __init__(self):
        super().__init__("depth_debug_viewer")
        self.window_name = "go2 depth: raw | processed"
        self.got_first_frame = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 965, 300)
        cv2.moveWindow(self.window_name, 40, 40)
        self.show_waiting_panel()
        self.timer = self.create_timer(0.1, lambda: cv2.waitKey(1))
        self.sub = self.create_subscription(Image, "/depth_camera/depth/image_raw", self.callback, 10)
        self.get_logger().info("Depth debug viewer subscribed to /depth_camera/depth/image_raw")

    def show_waiting_panel(self):
        panel = np.zeros((300, 965, 3), dtype=np.uint8)
        cv2.putText(panel, "Waiting for /depth_camera/depth/image_raw ...", (24, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "raw z-depth", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(panel, "processed ray-depth", (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.imshow(self.window_name, panel)
        cv2.waitKey(1)

    def callback(self, msg: Image):
        raw = image_to_depth(msg)
        if raw is None:
            self.get_logger().warn(f"Unsupported depth image: {msg.encoding} {msg.width}x{msg.height}", throttle_duration_sec=2.0)
            return
        if not self.got_first_frame:
            self.got_first_frame = True
            self.get_logger().info(f"Received first depth frame: {msg.encoding} {msg.width}x{msg.height}")

        processed = preprocess_depth(raw)
        raw_vis = colorize_depth(raw, CLIP_MIN, CLIP_MAX)
        processed_depth = (processed + 0.5) * CLIP_MAX
        proc_vis = colorize_depth(processed_depth, CLIP_MIN, CLIP_MAX)

        scale = 5
        raw_vis = cv2.resize(raw_vis, (RAW_W * scale, RAW_H * scale), interpolation=cv2.INTER_NEAREST)
        proc_vis = cv2.resize(proc_vis, (OUT_W * scale, OUT_H * scale), interpolation=cv2.INTER_NEAREST)
        proc_vis = cv2.copyMakeBorder(proc_vis, 0, raw_vis.shape[0] - proc_vis.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        panel = np.hstack([raw_vis, proc_vis])
        cv2.putText(panel, "raw z-depth", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, "processed ray-depth", (RAW_W * scale + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.window_name, panel)
        cv2.waitKey(1)


def main():
    rclpy.init(args=sys.argv)
    node = DepthDebugViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
