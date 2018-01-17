#include "klt_tracker.h"

KLT_Tracker::KLT_Tracker()
{
  init(25, true, 30);
}

void KLT_Tracker::init(int _num_features, bool _show_image, int _radius)
{
  initialized_ = false;
  num_features_ = _num_features;

  plot_matches_ = _show_image;
  feature_nearby_radius_ = _radius;

  next_feature_id_ = 0;
  prev_features_.clear();
  new_features_.clear();
  ids_.clear();

  colors_.clear();
  for (int i = 0; i < num_features_; i++)
  {
    colors_.push_back(Scalar(std::rand()/(RAND_MAX/255), std::rand()/(RAND_MAX/255), std::rand()/(RAND_MAX/255)));
  }
}


void KLT_Tracker::load_image(Mat img, double t, std::vector<Point2f> &features, std::vector<int> &ids)
{
  Mat grey_img;
  cvtColor(img, grey_img, COLOR_BGR2GRAY);
  if (!initialized_)
  {
    double quality_level = 0.3;
    int block_size = 7;
    goodFeaturesToTrack(grey_img, new_features_, num_features_, quality_level, feature_nearby_radius_, noArray(), block_size);
    for (int i = 0; i < new_features_.size(); i++)
    {
      ids_.push_back(next_feature_id_++);
    }
    prev_image_ = grey_img;
    initialized_ = true;
    prev_features_.resize(new_features_.size());
  }

  else
  {
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(prev_image_, grey_img, prev_features_, new_features_, status, err);

    // Keep only good points
    for (int i = prev_features_.size()-1; i > 0; i--)
    {
      if (status[i] == 0)
      {
        new_features_.erase(new_features_.begin() + i);
        ids_.erase(ids_.begin() + i);
      }
    }

    // If we are missing points, collect new ones
    if (new_features_.size() < num_features_)
    {
      // create a mask around current points
      Mat point_mask(grey_img.size(), CV_8UC1);
      point_mask = 255;
      for (int i = 0; i < new_features_.size(); i++)
      {
        circle(point_mask, new_features_[i], feature_nearby_radius_, 0, -1, 0);
      }

      // Now find a bunch of points, not in the mask
      int num_new_features = num_features_ - new_features_.size();
      vector<Point2f> new_corners;
      goodFeaturesToTrack(grey_img, new_corners, num_new_features, 0.3, feature_nearby_radius_, point_mask, 7);

      for (int i = 0; i < new_corners.size(); i++)
      {
        new_features_.push_back(new_corners[i]);
        ids_.push_back(next_feature_id_++);
      }
    }
  }

  if (plot_matches_)
  {
    Mat color_img;
    cvtColor(grey_img, color_img, COLOR_GRAY2BGR);
    // draw features and ids
    for (int i = 0; i < new_features_.size(); i++)
    {
      Scalar color = colors_[ids_[i] % num_features_];
      circle(color_img, new_features_[i], 5, color, -1);
      putText(color_img, to_string(ids_[i]), new_features_[i], FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 0));
    }
    imshow("Tracker Output", color_img);
    waitKey(1);
  }

  // Save off measurements for output
  features = new_features_;
  ids = ids_;

  // get ready for next iteration
  prev_image_ = grey_img;
  std::swap(new_features_, prev_features_);
  new_features_.resize(prev_features_.size());
}
