"""Full live demo (plan sections 23 Phase 11 / 24):

Gazebo (GUI) -> fault injector -> preprocessing -> Shadow EKF ->
Reliability AI -> Adaptive EKF (+ Fixed EKF comparison) -> dashboard.

    ros2 launch asr_bringup asr_demo.launch.py scenario:=slip
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory("asr_bringup")
    sim_time = {"use_sim_time": True}

    def node(pkg, exe, name=None, **params):
        return Node(package=pkg, executable=exe, name=name,
                    parameters=[{**sim_time, **params}], output="screen")

    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="slip"),
        DeclareLaunchArgument("mode", default_value="ai"),   # ai | rule
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("drive", default_value="true"),
        DeclareLaunchArgument("dashboard", default_value="true"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup, "launch", "sim.launch.py")),
            launch_arguments={"world": "asr_world.world",
                              "gui": LaunchConfiguration("gui")}.items()),
        node("asr_sensing", "ground_truth"),
        Node(package="asr_sensing", executable="driver", output="screen",
             condition=IfCondition(LaunchConfiguration("drive")),
             parameters=[{**sim_time, "seed": 42, "duration": 1.0e9}]),
        Node(package="asr_sensing", executable="fault_injector", output="screen",
             parameters=[{**sim_time,
                          "scenario": LaunchConfiguration("scenario"),
                          "t_end": 90.0}]),
        node("asr_sensing", "encoder_odom", input_topic="/asr/joint_states"),
        node("asr_sensing", "lidar_odom", input_topic="/asr/scan"),
        node("asr_shadow_ekf", "shadow_ekf"),
        Node(package="asr_reliability_estimator", executable="reliability",
             output="screen",
             parameters=[{**sim_time, "mode": LaunchConfiguration("mode")}]),
        node("asr_reliability_estimator", "adaptive_ekf", name="adaptive_ekf",
             mode="adaptive", output_topic="/asr/odom_adaptive"),
        node("asr_reliability_estimator", "adaptive_ekf", name="fixed_ekf",
             mode="fixed", output_topic="/asr/odom_fixed"),
        Node(package="asr_bringup", executable="dashboard", output="screen",
             condition=IfCondition(LaunchConfiguration("dashboard")),
             parameters=[sim_time]),
    ])
