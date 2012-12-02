#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
// PCL specific includes
#include <pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/integral_image_normal.h>
//#include <pcl/kdtree/kdtree.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>

// GTSAM
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BearingRangeFactor.h>

ros::Publisher pub;
volatile int numFrames = 0;
std::deque<pcl::PointCloud<pcl::FPFHSignature33>::Ptr> features;

void
cloud_cb (const sensor_msgs::PointCloud2ConstPtr& inCloud)
{
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
	
	// Republish normals to check
	sensor_msgs::PointCloud2Ptr oC(new sensor_msgs::PointCloud2);
	sensor_msgs::PointCloud2Ptr& outCloud = oC;
	pcl::toROSMsg(*normals, *outCloud);
	pub.publish(outCloud);
	std::cout << "Sent to " << pub.getNumSubscribers() << " Subscribers.\n";

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
	
	// Compute the features
	fpfh.compute (*fpfhs);
	
	// TODO: Store FPFH Signatures and Indices
	features.push_back(fpfhs);
	numFrames++;
}

void
computeCorrespondences()
{
	const float MAX_FEATURESPACE_DISTANCE = 1.0;
	gtsam::NonlinearFactorGraph graph;
	
	std::deque<pcl::KdTreeFLANN<pcl::FPFHSignature33>::Ptr> featureTrees;
	std::map<int, int> Correspondences; // TODO: look up PCL way to do this...
	featureTrees.resize(numFrames);
	
	// Initialize and sort features
	for (int i = 0; i < numFrames; i++)
	{
		featureTrees[i]->setInputCloud(features[i]);
	}
	
	// Find Putative Matches (PCL version?)
	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < numFrames; j++)
		{
			for (int k = 0; k < (int)(featureTrees[j]->getInputCloud()->points.size()); k++)
			{
				std::vector<int> indices(1);
				std::vector<float> distances(1);
				featureTrees[i]->nearestKSearch(featureTrees[j]->getInputCloud()->points[k], 1, indices, distances);
				if (distances[0] > MAX_FEATURESPACE_DISTANCE) {continue;}
				// TODO: Store the nearest neighbor indices for each feature for each frame
			}
		}
	}
	
	// TODO: Use RANSAC to thin putative matches
	
	
	// TODO: Add feature correspondences to graph
	for (int i = 0; i < numFrames; i++)
	{
		for (int j = 0; j < (int)(features[i]->points.size()); j++)
		{
			// TODO: lookup Key of globally observed feature
			int featureKey = 1; // graph key for feature
			int idx = 0; // point index of feature
			int u = 0,v = 0; //Image coordinates
			pcl::PointXYZ p_c; //Camera Frame point coordinates
			
			
			gtsam::noiseModel::Diagonal::shared_ptr noise = gtsam::noiseModel::Diagonal::Sigmas(Eigen::Vector4f(0.3, 0.3, 0.3, 0.1));
			
			graph.add(gtsam::BearingRangeFactor(i,featureKey,,,noise));
		}
	}
}


int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "my_pcl_tutorial");
  ros::NodeHandle nh;
  
  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("/camera/rgb/points", 1, cloud_cb);
  printf("Completed Setup.\n");
  // Create a ROS publisher for the output point cloud
  pub = nh.advertise<sensor_msgs::PointCloud2> ("output", 1);

  // Spin
  ros::spin ();
}
