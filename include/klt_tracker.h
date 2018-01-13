#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <vector>

#include <eigen3/Eigen/Core>

using namespace cv;
using namespace std;

class KLT_Tracker
{
public:
  KLT_Tracker(int _num_features, bool _show_image, int _radius);

  void load_image(Mat img, double t, std::vector<Point2f>& features, std::vector<int>& ids);

private:
  Mat prev_image_;
  bool initialized_ = false;
  bool plot_matches_ = false;
  int num_features_ = 0;
  int feature_nearby_radius_ = 30;

  vector<int> ids_;

  int next_feature_id_ = ;
  VideoWriter video_;
};
