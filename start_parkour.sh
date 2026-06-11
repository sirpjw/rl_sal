#!/bin/bash
# rl_sar Parkour Gazebo 一键启动
set -e

# --- 环境 ---
export PATH="/home/jiawei/miniconda3/envs/ros_humble/bin:/home/jiawei/rl_sar/install/bin:/opt/ros/humble/bin:$HOME/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:$PATH"
export LD_LIBRARY_PATH="/home/jiawei/rl_sar/install/lib:/opt/ros/humble/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/jiawei/rl_sar/install/lib/python3.10/site-packages:/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages"
export ROS_VERSION=2 ROS_DISTRO=humble
export AMENT_PREFIX_PATH="/home/jiawei/rl_sar/install:/opt/ros/humble"

# --- 杀旧进程 ---
for p in gzserver gzclient rl_sim robot_state_publisher spawn_entity; do
    killall -9 $p 2>/dev/null || true
done
sleep 2

# --- 清理 Gazebo 锁 ---
rm -f /tmp/gazebo-*/gazebo_master_* 2>/dev/null || true

echo "========================================="
echo "  rl_sar Parkour Gazebo 一键启动"
echo "========================================="

# --- 1. Gazebo ---
WORLD=/home/jiawei/rl_sar/install/share/rl_sar/worlds/stairs.world
echo "[1/4] 启动 Gazebo + 场景..."
gzserver $WORLD -slibgazebo_ros_init.so -slibgazebo_ros_factory.so -slibgazebo_ros_force_system.so &
sleep 3
gzclient &
sleep 2

# --- 2. robot_state_publisher ---
echo "[2/4] 加载机器人描述..."
ROBOT_DESC=$(xacro /home/jiawei/rl_sar/install/share/go2_description/xacro/robot.xacro 2>&1)
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$ROBOT_DESC" &
sleep 2

# --- 3. Spawn 狗 ---
echo "[3/4] Spawn Go2 机器狗..."
ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "{name: go2}" 2>/dev/null || true
sleep 1
ros2 run gazebo_ros spawn_entity.py -topic /robot_description -entity go2 -z 0.5
sleep 2

# --- 4. rl_sim ---
echo "[4/4] 启动控制器..."
echo ""
echo "  按 0 站立 | 按 1 Parkour 模式"
echo "  W/S 前进后退 | A/D 转向 | 空格 停止"
echo ""

ros2 run rl_sar rl_sim

# 清理
killall -9 gzserver gzclient robot_state_publisher 2>/dev/null || true
echo "已退出"
