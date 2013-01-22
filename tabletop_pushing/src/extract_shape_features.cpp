#include <sstream>
#include <iostream>
#include <fstream>
#include <cpl_visual_features/features/shape_context.h>
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/transforms.h>
#include <pcl/ros/conversions.h>

using namespace cpl_visual_features;

int main(int argc, char** argv)
{
  int max_displacement = 600;
  if (argc > 1)
  {
    max_displacement = atoi(argv[1]);
  }
  bool write_images = true;
  double epsilon_cost = 9e3;
  int num_images = 57;
  cv::Mat kernel(5,5,CV_8UC1, 255);
  std::string base_path("/home/thermans/Dropbox/Data/push_learning_object_visual_data/set0/object_img");
  std::ofstream out_file("/home/thermans/Desktop/shape_scores.txt");
  for (int i = 0; i < num_images-1; ++i)
  {
    // load images from disk
    std::stringstream imageA_path;
    imageA_path << base_path << i << ".png";
    cv::Mat imageA = cv::imread(imageA_path.str(),0);
    cv::Mat imageA1(imageA.size(), imageA.type());
    cv::dilate(imageA, imageA1, kernel);
    cv::imshow("imageA", imageA*255);
    cv::imshow("imageA1", imageA1*255);
    // for (int j = idx_b; j < idx_b+1; ++j)
    for (int j = i+1; j < num_images; ++j)
    {
      if (i == 8 && j == 21)
        continue;
      std::stringstream imageB_path;
      imageB_path << base_path << j << ".png";
      cv::Mat imageB = cv::imread(imageB_path.str(), 0);
      cv::Mat imageB1(imageB.size(), imageB.type());
      cv::dilate(imageB, imageB1, kernel);
      cv::imshow("imageB", imageB*255);
      cv::imshow("imageB1", imageB1*255);
      double score = compareShapes(imageA1, imageB1, epsilon_cost, write_images, "/home/thermans/Desktop", max_displacement);
      ROS_INFO_STREAM("Objects " << i << " and " << j << " match with score: " << score);
      out_file << i << ", " << j << " : " << score << "\n";
      out_file.flush();
    }
  }
  out_file.close();
  return 0;
}
