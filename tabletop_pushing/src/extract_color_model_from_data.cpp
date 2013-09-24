#include <sstream>
#include <iostream>
#include <fstream>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tabletop_pushing/extern/gmm/gmm.h>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

int main(int argc, char** argv)
{
  std::string data_directory_path(argv[1]);
  int num_image_pairs = atoi(argv[2]);
  std::string out_file_path(argv[3]);
  int num_clusters = 3;
  if (argc > 4)
  {
    num_clusters = atoi(argv[4]);
  }

  std::vector<cv::Vec3f> pnts;
  for (int i = 0; i < num_image_pairs; ++i)
  {
    std::cout << "Reading in img " << i << " of " << num_image_pairs << std::endl;
    std::stringstream arm_img_path;
    arm_img_path << data_directory_path << i << ".png";
    std::stringstream arm_mask_path;
    arm_mask_path << data_directory_path << i << "_mask.png";
    cv::Mat img = cv::imread(arm_img_path.str(), CV_LOAD_IMAGE_COLOR);
    cv::Mat mask = cv::imread(arm_mask_path.str(), CV_LOAD_IMAGE_GRAYSCALE);
    // cv::imshow("img", img);
    // cv::imshow("mask", mask);
    // cv::waitKey();
    cv::Mat img_lab_uchar(img.size(), img.type());
    cv::Mat img_lab(img.size(), CV_32FC3);
    cv::cvtColor(img, img_lab_uchar, CV_BGR2HSV);
    img_lab_uchar.convertTo(img_lab, CV_32FC3, 1.0/255);
    for (int r = 0; r < img_lab.rows; ++r)
    {
      for (int c = 0; c < img_lab.cols; ++c)
      {
        if (mask.at<uchar>(r,c) > 0)
        {
          pnts.push_back(img_lab.at<cv::Vec3f>(r,c));
        }
      }
    }
  }

  std::cout << "Have " << pnts.size() << " points to cluster on." << std::endl;

  if (pnts.size() < 1)
  {
    std::cerr << "No foreground points exist to build color model." << std::endl;
    return -1;
  }

  GMM color_model(0.0001);
  color_model.alloc(num_clusters);
  color_model.kmeansInit(pnts, 0.05);
  color_model.GmmEm(pnts);
  color_model.dispparams();
  // Save GMM
  color_model.savegmm(fs::path(out_file_path));
  // Read in and verify GMM is correct
  GMM new_color_model;
  new_color_model.loadgmm(fs::path(out_file_path));
  new_color_model.dispparams();
  // TODO: Visualize probabilties for sample images?

  return 0;
}
