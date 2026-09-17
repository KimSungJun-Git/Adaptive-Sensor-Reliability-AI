"""Headless fast-physics recording of one driving run -> rosbag2.

Shuts the whole launch down when the driver finishes its run.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            IncludeLaunchDescription, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory("asr_bringup")
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, "launch", "sim.launch.py")),
        launch_arguments={"world": LaunchConfiguration("world"), "gui": "false",
                          "x_pose": LaunchConfiguration("x_pose"),
                          "y_pose": LaunchConfiguration("y_pose")}.items())

    driver = Node(package="asr_sensing", executable="driver", output="screen",
                  parameters=[{"use_sim_time": True,
                               "seed": LaunchConfiguration("drive_seed"),
                               "duration": LaunchConfiguration("drive_duration")}])
    gt = Node(package="asr_sensing", executable="ground_truth",
              parameters=[{"use_sim_time": True}])
    bag = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", LaunchConfiguration("out"),
             "/joint_states", "/imu", "/scan", "/asr/ground_truth",
             "/cmd_vel", "/clock"],
        output="log")

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="tb3_world_fast.world"),
        DeclareLaunchArgument("x_pose", default_value="-2.0"),
        DeclareLaunchArgument("y_pose", default_value="-0.5"),
        DeclareLaunchArgument("drive_seed", default_value="0"),
        DeclareLaunchArgument("drive_duration", default_value="90.0"),
        DeclareLaunchArgument("out", default_value="/tmp/asr_run"),
        sim, driver, gt, bag,
        RegisterEventHandler(OnProcessExit(
            target_action=driver,
            on_exit=[EmitEvent(event=Shutdown(reason="run finished"))])),
    ])
