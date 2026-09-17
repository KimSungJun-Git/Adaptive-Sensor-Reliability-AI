# Shadow EKF innovation / NIS for one sensor update.
std_msgs/Header header
string sensor_name
string[] dof            # e.g. ['v', 'omega']
float64[] innovation
float64[] innovation_std  # sqrt(diag(S))
float64 nis
