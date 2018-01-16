#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <vector>

#include <eigen3/Eigen/Core>

using namespace cv;
using namespace std;

class KLT_Tracker
{
public:
  KLT_Tracker();
  KLT_Tracker(int _num_features, bool _show_image, int _radius);

  void set_num_features(int _num_features) {num_features_ = _num_features;}
  void set_radius(int _radius) {feature_nearby_radius_ = _radius;}
  void set_show_image(bool _show_image) {plot_matches_= _show_image;}

  void load_image(Mat img, double t, std::vector<Point2f>& features, std::vector<int>& ids);

private:
  Mat prev_image_;
  bool initialized_;
  bool plot_matches_;
  int num_features_;
  int feature_nearby_radius_;

  vector<int> ids_;
  vector<Point2f> prev_features_;
  vector<Point2f> new_features_;
  vector<Scalar> colors_;

  int next_feature_id_;
};