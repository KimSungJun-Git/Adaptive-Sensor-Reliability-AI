# Per-sensor reliability, one DOF each for v / omega (plan section 31.2).
# A DOF the sensor does not provide is set to -1 (e.g. imu s_v).
std_msgs/Header header
string sensor_name
float64 s_v
float64 s_omega
