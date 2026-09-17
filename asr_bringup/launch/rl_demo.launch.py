"""robot_localization deployment demo (plan section 31.1):

The Covariance Injector republishes the wheel odom / IMU / lidar twist with
reliability-scaled covariance; a stock robot_localization ekf_filter_node
fuses the *_adaptive topics. Runs on top of asr_demo.launch.py.

    ros2 launch asr_bringup rl_demo.launch.py scenario:=slip
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory("asr_bringup")
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="slip"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup, "launch", "asr_demo.launch.py")),
            launch_arguments={"scenario": LaunchConfiguration("scenario"),
                              "dashboard": "false"}.items()),
        Node(package="asr_reliability_estimator", executable="covariance_injector",
             output="screen",
             parameters=[{"use_sim_time": True,
                          "wheel_odom_in": "/odom",
                          "imu_in": "/asr/imu",
                          "lidar_twist_in": "/asr/lidar_twist"}]),
        Node(package="robot_localization", executable="ekf_node",
             name="ekf_filter_node", output="screen",
             parameters=[os.path.join(bringup, "config",
                                      "robot_localization.yaml"),
                         {"use_sim_time": True}]),
    ])
