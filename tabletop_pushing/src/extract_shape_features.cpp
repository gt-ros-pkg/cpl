#include <sstream>
#include <iostream>
#include <fstream>
#include <cpl_visual_features/features/shape_context.h>
#include <ros/ros.h>
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <pcl16/io/pcd_io.h>
#include <pcl16_ros/transforms.h>
#include <pcl16/ros/conversions.h>

using namespace cpl_visual_features;

#define XY_RES 0.00075

cv::Mat getObjectSillhoute(std::string image_path)
{
  cv::Mat kernel(5,5,CV_8UC1, 255);
  cv::Mat image = cv::imread(image_path,0);
  // Perform close
  cv::dilate(image, image, kernel);
  cv::erode(image, image, kernel);
  return image;
}
inline int worldLocToIdx(double val, double min_val, double max_val)
{
  return round((val-min_val)/XY_RES);
}

cv::Mat getObjectFootprint(std::string image_path, std::string cloud_path)
{
  cv::Mat kernel(5,5,CV_8UC1, 255);
  cv::Mat obj_mask = cv::imread(image_path,0);
  // Perform close
  cv::dilate(obj_mask, obj_mask, kernel);
  cv::erode(obj_mask, obj_mask, kernel);

  pcl16::PointCloud<pcl16::PointXYZ>::Ptr cloud(new pcl16::PointCloud<pcl16::PointXYZ>);

  if (pcl16::io::loadPCDFile<pcl16::PointXYZ> (cloud_path, *cloud) == -1) //* load the file
  {
    PCL16_ERROR ("Couldn't read file test_pcd.pcd \n");
    return cv::Mat();
  }
  double min_x = 300., max_x = -300.;
  double min_y = 300., max_y = -300.;

  // cv::Mat footprint;
  for (int r = 0; r < obj_mask.rows; ++r)
  {
    for (int c = 0; c < obj_mask.cols; ++c)
    {
      if (obj_mask.at<uchar>(r,c) > 0)
      {
        double x = cloud->at(c,r).x;
        double y = cloud->at(c,r).y;
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
      }
    }
  }
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);
  cv::Mat footprint(rows, cols, CV_8UC1, cv::Scalar(0));
  for (int r = 0; r < obj_mask.rows; ++r)
  {
    for (int c = 0; c < obj_mask.cols; ++c)
    {
      if (obj_mask.at<uchar>(r,c) > 0)
      {
        // TODO: Allow different max z in image.
        double x = cloud->at(c,r).x;
        double y = cloud->at(c,r).y;
        if (isnan(x) || isnan(y))
          continue;
        int img_x = worldLocToIdx(x, min_x, max_x);
        int img_y = worldLocToIdx(y, min_y, max_y);
        if (img_y > rows ||img_x > cols || img_y < 0 || img_x < 0)
        {
          ROS_WARN_STREAM("Image index out of bounds! at (" << img_y << ", " << img_x <<
                          ", " << y << ", " << x << ")");
          ROS_WARN_STREAM("X extremes: (" << min_x << ", " << max_x << ")");
          ROS_WARN_STREAM("Y extremes: (" << min_y << ", " << max_y << ")");
          ROS_WARN_STREAM("image size: (" << rows << ", " << cols << ")");
        }
        footprint.at<uchar>(img_y, img_x) = 255;
      }
    }
  }
  // Perform close
  cv::dilate(footprint, footprint, kernel);
  cv::erode(footprint, footprint, kernel);
  // TODO: Low pass filter this response to smooth edges?
  return footprint;
}

int main(int argc, char** argv)
{
  int max_displacement = 600;
  if (argc > 1)
  {
    max_displacement = atoi(argv[1]);
  }
  int idx_a = 0;
  int idx_b = 1;
  if (argc > 2)
  {
    idx_a = atoi(argv[2]);
    idx_b = atoi(argv[3]);
  }
  bool write_images = true;
  double epsilon_cost = 9e3;
  int num_images = 57;
  std::string base_path("/home/thermans/Dropbox/Data/push_learning_object_visual_data/set0/");
  std::string base_out_path("/home/thermans/Dropbox/Data/push_learning_object_visual_data/set0_footprint_comps/");
  std::stringstream out_file_path;
  out_file_path << base_out_path << "shape_scores.txt";
  std::ofstream out_file(out_file_path.str().c_str());
  // for (int i = idx_a; i < idx_a+1; ++i)
  for (int i = 0; i < num_images-1; ++i)
  {
    std::stringstream imageA_path;
    imageA_path << base_path << "object_img" << i << ".png";
    // cv::Mat imageA1 = getObjectSillhoute(imageA_path.str());
    // cv::imshow("imageA1", imageA1*255);
    std::stringstream cloudA_path;
    cloudA_path << base_path << "cloud" << i << ".pcd";
    cv::Mat base_image_a = getObjectFootprint(imageA_path.str(), cloudA_path.str());
    cv::imshow("base_a", base_image_a);

    // for (int j = idx_b; j < idx_b+1; ++j)
    for (int j = i+1; j < num_images; ++j)
    {
      std::stringstream imageB_path;
      imageB_path << base_path << "object_img"  << j << ".png";
      // cv::Mat imageB1 = getObjectSillhoute(imageB_path.str());
      // cv::imshow("imageB1", imageB1*255);
      std::stringstream cloudB_path;
      cloudB_path << base_path << "cloud" << j << ".pcd";
      cv::Mat base_image_b = getObjectFootprint(imageB_path.str(), cloudB_path.str());
      cv::imshow("base_b", base_image_b);

      std::stringstream post_fix;
      post_fix << "_" << i << "_" << j;
      double score = compareShapes(base_image_a, base_image_b, epsilon_cost, write_images,
                                   base_out_path, max_displacement, post_fix.str());

      ROS_INFO_STREAM("Objects " << i << " and " << j << " match with score: " << score);
      out_file << i << ", " << j << " : " << score << "\n";
      out_file.flush();
    }
  }
  out_file.close();
  return 0;
}
