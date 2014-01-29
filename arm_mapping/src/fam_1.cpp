// ROS includes
#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/image_encodings.h>

// PCL includes
#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp> // BruteForceMatcher
#include <cv_bridge/cv_bridge.h>


// GTSAM includes
#include <gtsam/base/types.h>

// Project Includes
#include "arm_mapping/factor_arm_mapping.h"

//using namespace arm_mapping;

struct Correspondence
{
	int Image1Idx;
	int Feature1Idx;
	int Image2Idx;
	int Feature2Idx;
	float Confidence;
	int GlobalFeatureIdx;
};

// Consts, flags, and counters
const int NUM_POSES = 10;
ros::Publisher pub;
volatile int corrFlag = 0;
volatile int numClouds = 0;
volatile int numImages = 0;

// Global variables
std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
std::vector<std::vector<cv::KeyPoint> > allKeypoints;
std::vector<cv::Mat> features;
std::vector<std::vector<gtsam::Point3> > featureCoordinates;
std::vector<Correspondence> featureMatches;
std::vector<gtsam::Pose3> eePoses;

// Function Prototypes
void rgbCameraImage_cb (const sensor_msgs::ImageConstPtr& inImg);
void pointCloud_cb (const sensor_msgs::PointCloud2ConstPtr& inCloud);
void tf_cb (const tf::tfMessage tfMsg);
void computeCorrespondences();

int main(int argc, char* argv[])
{
	// Reserve space in global vectors
	clouds.reserve(NUM_POSES);
	features.reserve(NUM_POSES);
	featureCoordinates.reserve(NUM_POSES);
	allKeypoints.reserve(NUM_POSES);
	eePoses.reserve(NUM_POSES);
	
	// Initialize ROS
	ros::init (argc, argv, "factor_arm_mapping");
	ros::NodeHandle nh;
	
	// Create a ROS subscribers for the input point cloud
	ros::Subscriber sub = nh.subscribe ("/camera/rgb/points", 2, pointCloud_cb);
	ros::Subscriber subRGBCam = nh.subscribe("/camera/rgb/image_color", 2,rgbCameraImage_cb);
	tf::TransformListener listener;
	listener.addTransformsChangedListener(tf_cb);
	
	std::cout << "Arm Mapping Initialized.\n";

	// Spin
	ros::spin ();
	return 0;
}

void rgbCameraImage_cb (const sensor_msgs::ImageConstPtr& inImg)
{
	if (numImages >= NUM_POSES) {return;}
	++numImages;
	
	std::cout << "Image Callback:\n";
	cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(inImg, sensor_msgs::image_encodings::BGR8);
	std::cout << "\tImage Size: " << cv_ptr->image.size[1] << "x" << cv_ptr->image.size[0] << "\n";
	cv::SurfFeatureDetector detector(400);
	
	std::vector<cv::KeyPoint> keypoints;
	detector.detect(cv_ptr->image, keypoints);
	
	std::cout << "\tFeatures Extracted: " << keypoints.size() << "\n";
	std::cout << "\tnumImages: " << numImages << "\n";
	
#ifndef USE_FPFH_FEATURES
	cv::SurfDescriptorExtractor extractor;
	cv::Mat descriptors;

	extractor.compute( cv_ptr->image, keypoints, descriptors);
	allKeypoints.push_back(keypoints);
	features.push_back(descriptors);
#endif
	
	// Signal all images processed
	if (numImages >= NUM_POSES)
	{
		corrFlag |= 0x1;
		std::cout << "All Images Processed\n";
	}
	
	if (corrFlag == 0x3)
	{
		computeCorrespondences();
	}
}

void pointCloud_cb (const sensor_msgs::PointCloud2ConstPtr& inCloud)
{
	if (numClouds >= 10) {return;}
	++numClouds;
	
	std::cout << "Cloud Callback:\n";
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromROSMsg(*inCloud, *cloud);
	
	// Estimate Normals
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.05f);//0.02f
	ne.setNormalSmoothingSize(0.05f);//0.02f
	ne.setInputCloud(cloud);
	ne.compute(*normals);
	
	
#ifdef USE_FPFH_FEATURES	

	// Create the FPFH estimation class, and pass the input dataset+normals to it
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud (cloud);
	fpfh.setInputNormals (normals);
	// Create an empty kdtree representation, and pass it to the FPFH estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	
	tree->setInputCloud(cloud);

	fpfh.setSearchMethod(tree);

	// Output dataset
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

	// Use all neighbors in a sphere of radius 20cm
	// IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
	fpfh.setRadiusSearch (0.20);
	
	std::cout << "\tEstimator initialized. Beginning FPFHS computation.\n";
	
	// Compute the features
	fpfh.compute (*fpfhs);
	// TODO: Store FPFH Signatures and Indices
	features.push_back(fpfhs);
#endif
	
	clouds.push_back(cloud);
	
	std::cout << "\tnumClouds: " << numClouds << "\n";
	
	// Signal all clouds processed
	if (numClouds >= NUM_POSES)
	{
		corrFlag |= 0x2;
		std::cout << "All Clouds Processed\n";
	}
	
	if (corrFlag == 0x3)
	{
		computeCorrespondences();
	}
}

void tf_cb (const tf::tfMessage tfMsg)
{
	// Not sure what to do here...
	//eePoses.push_back(gtsam::Pose3())
}

void computeCorrespondences()
{
	
	// Initialize and sort feature descriptors
#ifdef USE_FPFH_FEATURES
	std::deque<pcl::KdTreeFLANN<pcl::FPFHSignature33>::Ptr> featureTrees;
	featureTrees.resize(numClouds);
	for (int i = 0; i < numClouds; i++)
	{
		featureTrees[i]->setInputCloud(features[i]);
	}
#else
	std::vector<cv::DMatch> matches;
#endif
	
	// Look up feature coordinates
	for (int i = 0; i < NUM_POSES; i++)
	{
		std::vector<cv::KeyPoint>imageKeypoints = allKeypoints[i];
		for (int j = 0; j < (int)(imageKeypoints.size()); j++)
		{
			pcl::PointXYZ pxyz;
			pxyz = clouds[i]->points[ imageKeypoints[j].pt.x + imageKeypoints[j].pt.y * clouds[i]->width];
			gtsam::Point3 p3(pxyz.x, pxyz.y, pxyz.z);
			
			featureCoordinates[i].push_back(p3);
		}
	}
	
	// Find Putative Matches (PCL version?)
	for (int i = 0; i < NUM_POSES; i++)
	{
		for (int j = 0; j < NUM_POSES; j++)
		{
			if (i == j){continue;}
#ifdef USE_FPFH_FEATURES
			for (int k = 0; k < (int)(featureTrees[j]->getInputCloud()->points.size()); k++)
			{
				std::vector<int> indices(1);
				std::vector<float> distances(1);
				featureTrees[i]->nearestKSearch(featureTrees[j]->getInputCloud()->points[k], 1, indices, distances);
				if (distances[0] > MAX_FEATURESPACE_DISTANCE) {continue;}
				FeatureMatch fm;
				fm.Image1Idx = i;
				fm.Image2Idx = j;
				fm.Feature1Idx = indices[0];
				fm.Feature2Idx = k;
				fm.Confidence = distances[0];
				featureMatches.push_back(fm);
			}
#else
			cv::BruteForceMatcher< cv::L2<float> > matcher;
			matcher.match( features[i], features[j], matches );
			
			for (int k = 0; k < (int)(matches.size()); k++)
			{
				//if (matches[k].distance > MAX_FEATURESPACE_DISTANCE) {std::cout << "Discarded Match...\n"; continue;}
				Correspondence fm;
				fm.Image1Idx = i;
				fm.Image2Idx = j;
				fm.Feature1Idx = matches[k].queryIdx;
				fm.Feature2Idx = matches[k].trainIdx;
				fm.Confidence = matches[k].distance;
				featureMatches.push_back(fm);
			}
#endif
		}
	}
	std::cout << featureMatches.size() << " Putative Correspondences.\n";
	
	// TODO: Use RANSAC to thin putative matches?
	
	// Add feature correspondences to problem statement
	arm_mapping::FAMProblem prob;
	prob.correspondences(featureMatches);
	prob.points_cameras = featureCoordinates;
	prob.base_Ts_ee = eePoses;
	
	gtsam::Pose3 solution = solveProblemOffsetPose(prob);
	
	std::cout << "Computed Matches\n";
	std::cout << "Solution:\n" << solution << "\n";
	corrFlag = 0;
}
