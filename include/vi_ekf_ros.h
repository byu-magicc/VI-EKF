#include "vi_ekf.h"


class VIEKF_ROS
{
public:

  VIEKF_ROS();
  ~VIEKF_ROS();
  void image_callback(const sensor_msgs::ImageConstPtr& msg);
  void imu_callback(const sensor_msgs::ImuConstPtr& msg);
private:

};
