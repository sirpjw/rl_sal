#!/usr/bin/env python3
"""
Subscribe to Gazebo depth camera and save raw & processed depth images.
Mimics the exact processing pipeline from rl_sim.cpp:
  z-depth -> ray-depth -> clip -> crop -> bicubic resize -> normalize

Usage:
  RL_DEPTH_SAVE_DIR=/tmp/depth_out ros2 run rl_sar save_depth.py
  (or just: python3 save_depth.py)
"""
import os
import sys
import struct
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv2 import resize, INTER_CUBIC


# ─── constants (matching rl_sim.hpp) ───
DEPTH_RAW_W, DEPTH_RAW_H = 106, 60
DEPTH_W, DEPTH_H = 87, 58
DEPTH_CLIP_MAX = 2.0
# Intrinsic params (matching IsaacLab D435i)
FX = 106.0 * 11.041 / 20.955   # ~55.85
FY = 60.0  * 11.041 / 12.240   # ~54.12
CX = (106.0 - 1.0) * 0.5        # 52.5
CY = (60.0  - 1.0) * 0.5        # 29.5

SAVE_INTERVAL = 10  # save every N frames


def image_plane_to_ray(z_depth, u, v):
    """Convert Gazebo z-depth to ray distance (matches rl_sim.cpp)."""
    x = (u - CX) / FX
    y = (v - CY) / FY
    return z_depth * np.sqrt(1.0 + x * x + y * y)


def process_depth(raw_z: np.ndarray) -> dict:
    """Full processing pipeline from rl_sim::ProcessAndStoreDepth."""
    h, w = raw_z.shape  # (60, 106)

    # 1. z-depth -> ray depth
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    ray = image_plane_to_ray(raw_z, uu, vv)

    # 2. clip [0, DEPTH_CLIP_MAX]
    clipped = np.clip(ray, 0.0, DEPTH_CLIP_MAX)
    # NaN/Inf -> DEPTH_CLIP_MAX
    clipped[~np.isfinite(clipped)] = DEPTH_CLIP_MAX

    # 3. crop: height[:-2], width[4:-4]
    cropped = clipped[:-2, 4:-4]  # (58, 98)

    # 4. bicubic resize -> (58, 87)
    resized = resize(cropped, (DEPTH_W, DEPTH_H), interpolation=INTER_CUBIC)

    # 5. normalize: depth / max_distance - 0.5
    processed = np.clip(resized, 0.0, DEPTH_CLIP_MAX) / DEPTH_CLIP_MAX - 0.5

    return {
        "raw_z": raw_z.astype(np.float32),
        "ray": ray.astype(np.float32),
        "clipped": clipped.astype(np.float32),
        "cropped": cropped.astype(np.float32),
        "resized": resized.astype(np.float32),
        "processed": processed.astype(np.float32),
    }


def save_pgm(path, data, vmin=0.0, vmax=DEPTH_CLIP_MAX):
    """Save as 16-bit PGM."""
    h, w = data.shape
    norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    px = (norm * 65535).astype(np.uint16)
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n65535\n".encode())
        f.write(px.tobytes())


def save_bin(path, data):
    """Save as raw float32 binary."""
    data.astype(np.float32).tofile(path)


class DepthSaver(Node):
    def __init__(self):
        super().__init__("depth_saver")
        save_dir = os.environ.get("RL_DEPTH_SAVE_DIR", "/tmp/depth_output")
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.counter = 0
        self.save_every = int(os.environ.get("DEPTH_SAVE_INTERVAL", str(SAVE_INTERVAL)))

        self.sub = self.create_subscription(
            Image, "/depth_camera/depth/image_raw", self.callback, rclpy.qos.qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Depth saver ready. Saving to {os.path.abspath(save_dir)} every {self.save_every} frames"
        )

    def callback(self, msg: Image):
        self.counter += 1
        if self.counter % self.save_every != 0:
            return

        # parse depth image
        if msg.encoding == "32FC1":
            raw = np.frombuffer(msg.data, dtype=np.float32).reshape(DEPTH_RAW_H, DEPTH_RAW_W).copy()
        elif msg.encoding == "16UC1":
            raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(DEPTH_RAW_H, DEPTH_RAW_W).astype(np.float32) * 0.001
        else:
            self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
            return

        # process
        result = process_depth(raw)

        # save
        step = f"step_{self.counter:06d}"
        d = os.path.join(self.save_dir, step)
        os.makedirs(d, exist_ok=True)

        # 1. raw z-depth (60x106) - exactly what Gazebo publishes
        save_pgm(os.path.join(d, "raw_depth.pgm"), result["raw_z"])
        save_bin(os.path.join(d, "raw_depth.bin"), result["raw_z"])

        # 2. ray depth after conversion (60x106)
        save_pgm(os.path.join(d, "ray_depth.pgm"), result["ray"])
        save_bin(os.path.join(d, "ray_depth.bin"), result["ray"])

        # 3. cropped (58x98)
        save_pgm(os.path.join(d, "cropped_depth.pgm"), result["cropped"])
        save_bin(os.path.join(d, "cropped_depth.bin"), result["cropped"])

        # 4. resized (58x87)
        save_pgm(os.path.join(d, "resized_depth.pgm"), result["resized"])
        save_bin(os.path.join(d, "resized_depth.bin"), result["resized"])

        # 5. processed (58x87, normalized [-0.5, 0.5])
        save_pgm(os.path.join(d, "processed_depth.pgm"), result["processed"], vmin=-0.5, vmax=0.5)
        save_bin(os.path.join(d, "processed_depth.bin"), result["processed"])

        self.get_logger().info(f"Saved {step} ({self.counter} frames received)")


def main():
    rclpy.init()
    node = DepthSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
