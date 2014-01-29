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
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/console/print.h>
//#include <pcl/visualization/cloud_viewer.h>
//#include <pcl/visualization/pcl_visualizer.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp> // BruteForceMatcher
#include <cv_bridge/cv_bridge.h>

// BOOST includes
#include <boost/thread.hpp>
#include <algorithm>


// GTSAM includes
#include <gtsam/base/types.h>

// Project Includes
#include "arm_mapping/factor_arm_mapping.h"

//using namespace arm_mapping;

// Consts, flags, and counters
const int NUM_POSES = 20;
ros::Publisher pub;
tf::TransformListener* listener;
tf::Transform* eeOffset;
//pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
volatile int corrFlag = 0;
volatile int numClouds = 0;
volatile int numImages = 0;

// Global variables
std::vector<pcl::PointCloud<pcl::PointXYZ> > clouds;
std::vector<std::vector<cv::KeyPoint> > allKeypoints;
std::vector<cv::Mat> features;
std::vector<std::vector<gtsam::Point3> > cameraPoints;
std::vector<arm_mapping::Correspondence> featureMatches;
std::vector<gtsam::Pose3> eePoses;
std::vector<cv_bridge::CvImagePtr> imagePtrs;

// Function Prototypes
void rgbCameraImage_cb (const sensor_msgs::ImageConstPtr& inImg);
void pointCloud_cb (const sensor_msgs::PointCloud2ConstPtr& inCloud);
void computeCorrespondences();
bool compareMatches(cv::DMatch a, cv::DMatch b);
std::vector<int> RANSACInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudA,
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudB);


int main(int argc, char* argv[])
{
	// Initialize ROS
	ros::init (argc, argv, "factor_arm_mapping");
	ros::NodeHandle nh;
	
	// Initialize OpenCV threads
	static boost::once_flag cv_thread_flag = BOOST_ONCE_INIT;
	boost::call_once(cv_thread_flag, &cv::startWindowThread);
	cv::namedWindow("Image Display");
	
	// Reserve space in global vectors
	clouds.reserve(NUM_POSES);
	features.reserve(NUM_POSES);
	allKeypoints.reserve(NUM_POSES);
	eePoses.reserve(NUM_POSES);
	cameraPoints.resize(NUM_POSES);
	for (int i = 0; i < NUM_POSES; i++)
	{
		cameraPoints[i].resize(0);
	}
	imagePtrs.reserve(NUM_POSES);
	
	// Initialize offset transform to solve for
	tf::Transform tForm;
	eeOffset = &tForm;
	eeOffset->setOrigin(tf::Vector3(0.3,0.3,0.3));
	//eeOffset->setRotation(tf::Quaternion(0,1,0,1));
	
	tf::TransformListener l;
	listener = &l;
	
	// Create a ROS subscribers for the input point cloud
	ros::Subscriber sub = nh.subscribe ("/camera/rgb/points", 2, pointCloud_cb);
	ros::Subscriber subRGBCam = nh.subscribe("/camera/rgb/image_color", 2,rgbCameraImage_cb);
	
	std::cout << "Arm Mapping Initialized.\n";
	
	// Load and play bagfile
	int pid = fork();
	if (pid == 0)
	{
		system("rosbag play ~/Downloads/desk2.bag -q");
	}
	else
	{
		// Spin
		ros::spin ();
	}
	
	return 0;
}

void rgbCameraImage_cb (const sensor_msgs::ImageConstPtr& inImg)
{
	if (numImages >= NUM_POSES) {return;}
	++numImages;
	
	std::cout << "Image Callback:\n";
	cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(inImg, sensor_msgs::image_encodings::BGR8);
	imagePtrs.push_back(cv_ptr);
	cv::imshow("Image Display", cv_ptr->image);
	
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
	if (numClouds >= NUM_POSES) {return;}
	++numClouds;
	
	std::cout << "Cloud Callback:\n";
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromROSMsg(*inCloud, *cloud);
	
	// Get End Effector Pose
	try
	{
	//ros::Time now = ros::Time::now();
	tf::StampedTransform transform;
	listener->lookupTransform("/world","/kinect", ros::Time(0), transform);
	transform.inverseTimes(*eeOffset);
	eePoses.push_back(gtsam::Pose3(gtsam::Quaternion(
			transform.getRotation().getX(),transform.getRotation().getY(),
			transform.getRotation().getZ(),transform.getRotation().getW()), 
			gtsam::Point3(transform.getOrigin().getX(),
					transform.getOrigin().getY(),transform.getOrigin().getZ())));
	}
	catch (tf::TransformException* ex)
	{
		ROS_ERROR("%s",ex->what());
	}
#ifdef USE_FPFH_FEATURES		
	// Estimate Normals
	pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
	pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
	ne.setMaxDepthChangeFactor(0.05f);//0.02f
	ne.setNormalSmoothingSize(0.05f);//0.02f
	ne.setInputCloud(cloud);
	ne.compute(*normals);

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
	
	clouds.push_back(*cloud);
	//viewer.showCloud((boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> >)cloud);
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

void computeCorrespondences()
{
	const float MAX_FEATURESPACE_DISTANCE = 0.16;
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
	
	// Find Putative Matches
	for (int i = 0; i < NUM_POSES; i++)
	{
		//for (int j = 0; j < NUM_POSES; j++)
		for (int j = max(0, i-1); j < min(NUM_POSES, i+1); j++)
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
			matcher.match( features[i], features[j], matches);
			std::sort(matches.begin(),matches.end(), compareMatches);
			
			// mask matches
			vector<char> matchMask( matches.size(), 0 );
			int numGoodMatches;
			for (numGoodMatches= 0; numGoodMatches < (int)(matches.size()); numGoodMatches++)
			{
				// matches are sorted, so we can break once they get too big
				if (matches[numGoodMatches].distance > MAX_FEATURESPACE_DISTANCE) {break;}
				matchMask[numGoodMatches] = 1;
			}
			if (numGoodMatches < 4) {break;}
			//std::cout << numGoodMatches << "\n";
			
			// inspect matches
			cv::Mat img;
			cv::drawMatches(imagePtrs[i]->image, allKeypoints[i], imagePtrs[j]->image, allKeypoints[j], matches, img, cv::Scalar::all(-1), cv::Scalar::all(-1), matchMask);
			cv::imshow("Image Display", img);
			cv::waitKey(5000);
			
			pcl::PointCloud<pcl::PointXYZ>::Ptr keysXYZi, keysXYZj;
			pcl::PointCloud<pcl::PointXYZ> cloud; 
			keysXYZi = cloud.makeShared();
			keysXYZj = cloud.makeShared();
			
			for (int k = 0; k < (int)(numGoodMatches); k++)
			{
				cv::DMatch match = matches[k];
				
				// Look up 3D coordinates for both matching points;
				pcl::PointXYZ pi, pj;
				
				pi = clouds[i].points[allKeypoints[i][match.queryIdx].pt.x + clouds[i].width * allKeypoints[i][match.queryIdx].pt.y];
				pj = clouds[j].points[allKeypoints[j][match.trainIdx].pt.x + clouds[j].width * allKeypoints[j][match.trainIdx].pt.y];
				
				if (pi.z != pi.z || pj.z != pj.z) {continue;}
				
				keysXYZi->points.push_back(pi);
				keysXYZj->points.push_back(pj);
			}
			
			// RANSAC check points
			std::vector<int> inliers = RANSACInliers(keysXYZi, keysXYZj);
			std::cout << "Dropped outliers: " << numGoodMatches << "->" << inliers.size() << "\n";
			for (int k = 0; k < (int)(inliers.size()); k++)
			{
				//if (inliers[k])
				pcl::PointXYZ pi, pj;
								
				pi = keysXYZi->points[inliers[k]];
				pj = keysXYZj->points[inliers[k]];
				gtsam::Point3 gtspi(pi.x, pi.y, pi.z);
				gtsam::Point3 gtspj(pj.x, pj.y, pj.z);
				gtspi = eePoses[i].transform_to(gtspi);
				gtspj = eePoses[j].transform_to(gtspj);
				
				cameraPoints[i].push_back(gtspi);
				cameraPoints[j].push_back(gtspj);
				
				arm_mapping::Correspondence fm(i, j, cameraPoints[i].size()-1, cameraPoints[j].size()-1);
				featureMatches.push_back(fm);
			}
			
#endif
		}
	}
	std::cout << featureMatches.size() << " Putative Correspondences.\n";
	
	
	// Add feature correspondences to problem statement
	arm_mapping::FAMProblem prob;
	prob.correspondences = featureMatches;
	prob.points_cameras = cameraPoints;
	prob.base_Ts_ee = eePoses;
	
	std::cout << "Computing Matches...";
	gtsam::Pose3 solution = solveProblemOffsetPose(prob);
	
	std::cout << "Computed Matches\n";
	std::cout << "Solution: \n";
	solution.print();
	corrFlag = 0;
	
	ros::shutdown();
}

bool compareMatches(const cv::DMatch a, const cv::DMatch b)
{
	return a.distance < b.distance;
}

std::vector<int> RANSACInliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudA,
            const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudB)//,
            //Eigen::Matrix4f& Tresult)
{
    pcl::SampleConsensusModelRegistration<pcl::PointXYZ>::Ptr sac_model(new pcl::SampleConsensusModelRegistration<pcl::PointXYZ>(cloudA));
    sac_model->setInputTarget(cloudB);
 
    pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(sac_model);
    //pcl::LeastMedianSquares<pcl::PointNormal> ransac(sac_model); //might as well try these out too!
    //pcl::ProgressiveSampleConsensus<pcl::PointNormal> ransac(sac_model);
    ransac.setDistanceThreshold(0.1);
 
    //upping the verbosity level to see some info
    //pcl::console::VERBOSITY_LEVEL vblvl = pcl::console::getVerbosityLevel();
    //pcl::console::setVerbosityLevel(pcl::console::VERBOSITY_LEVEL::L_DEBUG);
    ransac.computeModel(1);
    //pcl::console::setVerbosityLevel(vblvl);
 
    Eigen::VectorXf coeffs;
    ransac.getModelCoefficients(coeffs);
    //assert(coeffs.size() == 16);
    //Tresult = Eigen::Map<Eigen::Matrix4f>(coeffs.data(),4,4);
 
    vector<int> inliers; ransac.getInliers(inliers);
    return inliers;
}
