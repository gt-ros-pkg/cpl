#include <sstream>
#include <iostream>
#include <fstream>
#include <tabletop_pushing/shape_features.h>
#include <ros/ros.h>
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <pcl16/io/pcd_io.h>
#include <pcl16_ros/transforms.h>
#include <pcl16/ros/conversions.h>
#include <pcl16/common/pca.h>
#include <tabletop_pushing/VisFeedbackPushTrackingAction.h>
#include <tabletop_pushing/point_cloud_segmentation.h>
#include <cpl_visual_features/helpers.h>
#include <cpl_visual_features/features/kernels.h>
#include <time.h> // for srand(time(NULL))

using namespace cpl_visual_features;
using namespace tabletop_pushing;

#define XY_RES 0.001

inline int objLocToIdx(double val, double min_val, double max_val)
{
  return round((val-min_val)/XY_RES);
}


int main(int argc, char** argv)
{
  double max_y = 0.5;
  double min_y = -0.5;
  double max_x = 0.5;
  double min_x = -0.5;


  int max_dist = 0.5;
  int max_radius = objLocToIdx(max_dist, min_x, max_x);
  int radius_bins = 6; // actually number of radius separators (including 0)
  int theta_bins = 12;

  int xy_pad = 15;
  cv::Mat hist_img(max_radius*2+xy_pad*2, max_radius*2+xy_pad*2, CV_8UC3, cv::Scalar(255, 255, 255));

  cv::Point img_center(hist_img.rows/2, hist_img.cols/2);
  // Draw log spaced circles
  for (int r = 0; r < radius_bins; ++r)
  {
    float cur_radius = max_radius-log(r+1)/log(float(radius_bins))*max_radius;
    cv::circle(hist_img, img_center, cur_radius, cv::Scalar(255, 0 ,0), 3);
  }

  // Draw polar dividing lines
  for (int t = 0; t < theta_bins; ++t)
  {
    float cur_angle = 2.0*M_PI*(float(t)/theta_bins)-M_PI;
    cv::Point end_point(cos(cur_angle)*max_radius, sin(cur_angle)*max_radius);
    cv::line(hist_img, img_center, end_point+img_center, cv::Scalar(255, 0, 0), 3);
  }
  cv::imshow("log-polar hist", hist_img);
  cv::imwrite("/home/thermans/Desktop/log_polar_hist.png", hist_img);
  cv::waitKey();
  return 0;
}
