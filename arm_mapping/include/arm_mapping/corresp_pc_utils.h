#include <stdio.h>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

using namespace cv;
using namespace std;

typedef pcl::PointXYZRGB PRGB;
typedef pcl::PointCloud<PRGB> PCRGB;

void makePCColor(const PCRGB &pc, uint32_t rgb);

void visClouds(const PCRGB &pc_1, const PCRGB &pc_2);

void getMatches(const Mat &img_1, const Mat &img_2,
                vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2,
                vector<DMatch> &matches) ;

void extractKeypoints(const PCRGB &pc, const vector<KeyPoint> &kps, PCRGB &pc_out);

void pruneMatches(const PCRGB &kps_1, const PCRGB &kps_2, const vector<DMatch> &matches_in,
                  vector<DMatch> &matches_out);

void extractMatches(const PCRGB &kps_1, const PCRGB &kps_2, const vector<DMatch> &matches, 
                    PCRGB &matches1, PCRGB &matches2);

Eigen::Matrix4f umeyamaRegistration(const PCRGB& pc1, const PCRGB& pc2);

void transPoints(const PCRGB &pc_in, const Eigen::Matrix4f &trans, PCRGB &pc_out);

int numInliers(const PCRGB &pc_1, const PCRGB &pc_2, double dist_thresh);

Eigen::Matrix4f findTransRansac(const PCRGB &kps_1, const PCRGB &kps_2, const vector<DMatch> &matches);

Eigen::Matrix4f registerPCs(const Mat &img_1, const PCRGB &pc_1, const Mat &img_2, const PCRGB &pc_2);

void readBag(char* filename, vector<Mat> &imgs, vector<PCRGB::Ptr> &pcs, size_t num=-1);

void writeBag(char* filename, PCRGB &pc);
