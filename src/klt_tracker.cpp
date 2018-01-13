#include "klt_tracker.h"

KLT_Tracker::KLT_Tracker()
{
  KLT_Tracker(25, true, 30);
}

KLT_Tracker::KLT_Tracker(int _num_features, bool _show_image, int _radius)
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
  for (int i = 0; i < 100; i++)
  {
    colors_.push_back(Scalar(std::rand()/(RAND_MAX/255), std::rand()/(RAND_MAX/255), std::rand()/(RAND_MAX/255)));
  }
}


void KLT_Tracker::load_image(Mat img, double t, std::vector<Point2f> &features, std::vector<int> &ids)
{
  if (!initialized_)
  {
    double quality_level = 0.3;
    int block_size = 7;
    goodFeaturesToTrack(img, new_features_, num_features_, quality_level, feature_nearby_radius_);
    for (int i = 0; i < new_features_.size(); i++)
    {
      ids_.push_back(next_feature_id_++);
    }
    prev_image_ = img;
    initialized_ = true;
    prev_features_.resize(new_features_.size());
  }

  else
  {
    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(prev_image_, img, prev_features_, new_features_, status, err);
    prev_image_ = img;

    // Keep only good points
    for (int i = prev_features_.size(); i > 0; i--)
    {
      if (status[i] == 1)
      {
        new_features_.erase(new_features_.begin() + i);
        ids_.erase(ids_.begin() + i);
      }
    }

    // If we are missing points, collect new ones
    if (new_features_.size() < num_features_)
    {
      // create a mask around current points
      Mat point_mask = Mat::ones(img.rows, img.cols, CV_8UC1);
      for (int i = 0; i < new_features_.size(); i++)
      {
        circle(point_mask, new_features_[i], feature_nearby_radius_, 0 -1, 0);
      }

      // Now find a bunch of points, not in the mask
      int num_new_features = num_features_ - new_features_.size();
      vector<Point2f> new_corners;
      goodFeaturesToTrack(img, new_corners, num_new_features, 0.3, feature_nearby_radius_, point_mask );

      for (int i = 0; i < new_corners.size(); i++)
      {
        new_features_.push_back(new_corners[i]);
        ids_.push_back(next_feature_id_++);
      }
    }

    if (plot_matches_)
    {
      // Convert image to color
      Mat color_img;
      cvtColor(img, color_img, COLOR_GRAY2BGR);

      // draw features and ids
      for (int i = 0; i < new_features_.size(); i++)
      {
        circle(img, new_features_[i], 5, colors_[ids_[i] % new_features_.size()], -1);
        putText(img, to_string(ids_[i]), new_features_[i], FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 0));
      }
      imshow("Tracker Output", img);
      waitKey(1);
    }
  }
}
