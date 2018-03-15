#include "klt_tracker.h"

KLT_Tracker::KLT_Tracker()
{
  init(12, true, 30);
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

bool KLT_Tracker::drop_feature(int feature_id)
{
  // get the local index of this feature_id
  int local_id = std::distance(ids_.begin(), std::find(ids_.begin(), ids_.end(), feature_id));
  if (local_id < ids_.size())
  {
    ids_.erase(ids_.begin() + local_id);
    new_features_.erase(new_features_.begin() + local_id);
    return true;
  }
  else
  {
    return false;
  }
}


void KLT_Tracker::load_image(Mat img, double t, std::vector<Point2f> &features, std::vector<int> &ids, OutputArray& output)
{
  Mat grey_img;
  if (img.channels() > 1)
  {
    cvtColor(img, grey_img, COLOR_BGR2GRAY);
  }
  else
  {
    grey_img = img;
  }
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
    deque<int> good_ids;
    int offscreen_points_removed = 0;
    int tooclose_points_removed = 0;
    for (int i = prev_features_.size()-1; i >= 0; i--)
    {
      // If we found a match and the match is in the image
      double new_x = new_features_[i].x;
      double new_y = new_features_[i].y;
      if (status[i] == 0 || new_x <= 1.0 || new_y <= 1.0 || new_x >= img.cols-1.0 || new_y >= img.rows-1.0)
      {
        offscreen_points_removed++;
        new_features_.erase(new_features_.begin() + i);
        ids_.erase(ids_.begin() + i);
        continue;
      }
      
      // Make sure that it's not too close to other points
      bool good_point = true;
      for (auto it = good_ids.begin(); it != good_ids.end(); it++)
      {
        double dx = new_features_[*it].x - new_x;
        double dy = new_features_[*it].y - new_y;
        if (std::pow(dx*dx + dy*dy, 0.5) < feature_nearby_radius_)
        {
          tooclose_points_removed++;
          good_point = false;
          new_features_.erase(new_features_.begin() + i);
          ids_.erase(ids_.begin() + i);
          break;
        }
      }
      if (good_point)
        good_ids.push_back(i);
    }
    if (offscreen_points_removed + tooclose_points_removed > 0)
      std::cout << "removed points:  offscreen: " << offscreen_points_removed << " tooclose: " << tooclose_points_removed << " total: " << offscreen_points_removed + tooclose_points_removed << "\n";
    
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
      putText(color_img, to_string(ids_[i]), new_features_[i], FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
    }
    //    imshow("Tracker Output", color_img);
    color_img.copyTo(output);
    //    waitKey(1);
  }
  
  // Save off measurements for output
  features = new_features_;
  ids = ids_;
  
  // Make sure to saturate the measurement (sometimes the klt tracker returns measurements that are out-of-bounds)
  for (int i = 0; i < features.size(); i++)
  {
    features[i].x = features[i].x > img.cols ? img.cols : features[i].x < 0 ? 0 : features[i].x;
    features[i].y = features[i].y > img.rows ? img.rows : features[i].y < 0 ? 0 : features[i].y;
  }
  
  // get ready for next iteration
  prev_image_ = grey_img;
  std::swap(new_features_, prev_features_);
  new_features_.resize(prev_features_.size());
}
