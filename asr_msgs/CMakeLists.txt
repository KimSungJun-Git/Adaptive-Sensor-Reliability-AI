cmake_minimum_required(VERSION 3.8)
project(asr_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SensorReliability.msg"
  "msg/ReliabilityArray.msg"
  "msg/Innovation.msg"
  "msg/FaultStatus.msg"
  "msg/LidarQuality.msg"
  DEPENDENCIES std_msgs
)

ament_package()
