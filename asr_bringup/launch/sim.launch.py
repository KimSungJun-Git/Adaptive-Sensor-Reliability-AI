"""Gazebo Classic + TurtleBot3 waffle_pi in the ASR world."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory("asr_bringup")
    tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    gazebo_ros = get_package_share_directory("gazebo_ros")

    world = LaunchConfiguration("world")
    gui = LaunchConfiguration("gui")

    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="asr_world.world"),
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("x_pose", default_value="-2.0"),
        DeclareLaunchArgument("y_pose", default_value="-0.5"),
        SetEnvironmentVariable("TURTLEBOT3_MODEL", "waffle_pi"),
        AppendEnvironmentVariable("GAZEBO_MODEL_PATH",
                                  os.path.join(tb3_gazebo, "models")),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros, "launch", "gzserver.launch.py")),
            launch_arguments={
                "world": PythonExpression(
                    ["'", os.path.join(bringup, "worlds"), "/' + '", world, "'"]),
                "params_file": os.path.join(bringup, "config",
                                            "gazebo_params.yaml"),
            }.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_ros, "launch", "gzclient.launch.py")),
            condition=IfCondition(gui)),
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=["-entity", "waffle_pi",
                       "-file", os.path.join(tb3_gazebo, "models",
                                             "turtlebot3_waffle_pi", "model.sdf"),
                       "-x", LaunchConfiguration("x_pose"),
                       "-y", LaunchConfiguration("y_pose"),
                       "-z", "0.01"],
            output="screen"),
    ])
