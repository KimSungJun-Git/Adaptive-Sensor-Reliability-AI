# Scan-matching quality of the latest lidar odometry step.
std_msgs/Header header
float64 rms             # mean point-to-point residual [m]
float64 match_ratio     # fraction of matched points [0..1]
float64 degeneracy      # lambda_min/lambda_max of translation Hessian [0..1]
bool valid
