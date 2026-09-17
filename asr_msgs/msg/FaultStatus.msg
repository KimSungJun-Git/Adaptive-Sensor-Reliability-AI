# Ground-truth fault activity from the demo fault injector (evaluation only).
std_msgs/Header header
string sensor_name
string channel          # 'v' | 'omega' | 'scan'
string fault_type
bool active
float64 magnitude
