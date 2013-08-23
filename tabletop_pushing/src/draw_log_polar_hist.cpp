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

cv::Mat makeHistogramImage(ShapeDescriptor histogram, int n_x_bins, int n_y_bins, int bin_width_pixels)
{
  cv::Mat hist_img(n_y_bins*bin_width_pixels+1, n_x_bins*bin_width_pixels+1, CV_8UC1, cv::Scalar(255));
  int max_count = 10;
  for (int i = 0; i < histogram.size(); ++i)
  {
    if (histogram[i] == 0) continue;
    int pix_val = (1 - histogram[i]/max_count)*255;
    ROS_INFO_STREAM("Pix val: " << pix_val);
    int start_x = (i%n_x_bins)*bin_width_pixels;
    int start_y = (i/n_x_bins)*bin_width_pixels;;
    ROS_INFO_STREAM("(start_x, start_y): (" << start_x << ", " << start_y << ")");
    for (int x = start_x; x < start_x+bin_width_pixels; ++x)
    {
      for (int y = start_y; y < start_y+bin_width_pixels; ++y)
      {
        hist_img.at<uchar>(y, x) = pix_val;
      }
    }
  }
  ROS_INFO_STREAM("Drawing columns");
  // Draw hist bin columns
  for (int i = 0; i <= n_x_bins; ++i)
  {
    int x = i*bin_width_pixels;
    for (int y = 0; y < hist_img.rows; ++y)
    {
      hist_img.at<uchar>(y,x) = 0;
    }
  }
  ROS_INFO_STREAM("Drawing rows");
  // Draw hist bin rows
  for (int i = 0; i <= n_y_bins; ++i)
  {
    int y = i*bin_width_pixels;
    for (int x = 0; x < hist_img.cols; ++x)
    {
      hist_img.at<uchar>(y,x) = 0;
    }
  }
  // cv::imshow("local_hist_img", hist_img);
  // cv::waitKey();
  ROS_INFO_STREAM("Returning");
  return hist_img;
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
  ShapeDescriptor sd;
  for (int r = 0; r < radius_bins; ++r)
  {
    for (int t = 0; t < theta_bins; ++t)
    {
      int s = 0;
      if (t == 0)
      {
        if (r == 4)
        {
          s = 2;
        }
        else if (r == 3)
        {
          s = 10;
        }
      }
      else if (t == 1)
      {
        if (r == 3)
        {
          s = 6;
        }
        if (r == 2)
        {
          s = 2;
        }
      }
      else if (t == 2)
      {
        if (r == 2)
        {
          s = 5;
        }
        else if (r == 1)
        {
          s = 4;
        }
      }
      else if (t == 3)
      {
        if (r == 1)
        {
          s = 1;
        }
        if (r == 0)
        {
          s = 1;
        }
      }
      else if (t == 4)
      {
        if (r == 0)
        {
          s = 1;
        }
      }
      else if (t == 9)
      {
        if (r == 0)
        {
          s = 2;
        }
      }
      else if (t == 10)
      {
        if (r == 0)
        {
          s = 2;
        }
        else if (r == 1)
        {
          s = 3;
        }
        else if (r == 2)
        {
          s = 4;
        }
        else if (r == 3)
        {
          s = 6;
        }
      }
      else if (t == 11)
      {
        if (r == 3)
        {
          s = 10;
        }
        if (r == 4)
        {
          s = 2;
        }
      }
      sd.push_back(s);
      // ROS_INFO_STREAM("(" << r << ", " << t << ") : " << s);
    }
  }
  cv::Mat rect_hist_img = makeHistogramImage(sd, theta_bins, radius_bins, 10);
  cv::imshow("rect_hist", rect_hist_img);
  cv::imwrite("/home/thermans/Desktop/rect_log_polar_hist.png", rect_hist_img);
  cv::waitKey();
  return 0;
}
