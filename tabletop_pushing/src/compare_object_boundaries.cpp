#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <ros/ros.h>
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <pcl16/io/pcd_io.h>
#include <pcl16/surface/concave_hull.h>
#include <tabletop_pushing/shape_features.h>
#include <tabletop_pushing/point_cloud_segmentation.h>
#include <tabletop_pushing/VisFeedbackPushTrackingAction.h>
// PCL
#include <pcl16/common/pca.h>

// cpl_visual_features
#include <cpl_visual_features/helpers.h>

typedef tabletop_pushing::VisFeedbackPushTrackingFeedback PushTrackerState;
using cpl_visual_features::subPIAngle;
using namespace tabletop_pushing;


double getThetaFromEllipse(cv::RotatedRect& obj_ellipse)
{
  return subPIAngle(DEG2RAD(obj_ellipse.angle)+0.5*M_PI);
}

void fitHullEllipse(XYZPointCloud& hull_cloud, cv::RotatedRect& obj_ellipse)
{
  Eigen::Vector3f eigen_values;
  Eigen::Matrix3f eigen_vectors;
  Eigen::Vector4f centroid;

  // HACK: Copied/adapted from PCA in PCL because PCL was seg faulting after an update on the robot
  // Compute mean
  centroid = Eigen::Vector4f::Zero();
  // ROS_INFO_STREAM("Getting centroid");
  pcl16::compute3DCentroid(hull_cloud, centroid);
  // Compute demeanished cloud
  Eigen::MatrixXf cloud_demean;
  // ROS_INFO_STREAM("Demenaing point cloud");
  pcl16::demeanPointCloud(hull_cloud, centroid, cloud_demean);

  // Compute the product cloud_demean * cloud_demean^T
  // ROS_INFO_STREAM("Getting alpha");
  Eigen::Matrix3f alpha = static_cast<Eigen::Matrix3f> (cloud_demean.topRows<3> () * cloud_demean.topRows<3> ().transpose ());

  // Compute eigen vectors and values
  // ROS_INFO_STREAM("Getting eigenvectors");
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> evd (alpha);
  // Organize eigenvectors and eigenvalues in ascendent order
  for (int i = 0; i < 3; ++i)
  {
    eigen_values[i] = evd.eigenvalues()[2-i];
    eigen_vectors.col(i) = evd.eigenvectors().col(2-i);
  }

  // try{
  //   pca.setInputCloud(cloud_no_z.makeShared());
  //   ROS_INFO_STREAM("Getting mean");
  //   centroid = pca.getMean();p
  //   ROS_INFO_STREAM("Getting eiven values");
  //   eigen_values = pca.getEigenValues();
  //   ROS_INFO_STREAM("Getting eiven vectors");
  //   eigen_vectors = pca.getEigenVectors();
  // } catch(pcl16::InitFailedException ife)
  // {
  //   ROS_WARN_STREAM("Failed to compute PCA");
  //   ROS_WARN_STREAM("ife: " << ife.what());
  // }

  obj_ellipse.center.x = centroid[0];
  obj_ellipse.center.y = centroid[1];
  obj_ellipse.angle = RAD2DEG(atan2(eigen_vectors(1,0), eigen_vectors(0,0))-0.5*M_PI);
  // NOTE: major axis is defined by height
  obj_ellipse.size.height = std::max(eigen_values(0)*0.1, 0.07);
  obj_ellipse.size.width = std::max(eigen_values(1)*0.1, 0.03);

}

XYZPointCloud getHullFromPCDFile(std::string cloud_path)
{
  XYZPointCloud cloud;
  if (pcl16::io::loadPCDFile<pcl16::PointXYZ>(cloud_path, cloud) == -1) //* load the file
  {
    ROS_ERROR_STREAM("Couldn't read file " << cloud_path);
  }
  // ROS_INFO_STREAM("Cloud has " << cloud.size() << " points.");
  XYZPointCloud hull_cloud;
  float hull_alpha = 0.01;
  pcl16::ConcaveHull<pcl16::PointXYZ> hull;
  hull.setDimension(2);  // NOTE: Get 2D projection of object
  hull.setInputCloud(cloud.makeShared());
  hull.setAlpha(hull_alpha);
  hull.reconstruct(hull_cloud);
  // ROS_INFO_STREAM("Hull has " << hull_cloud.size() << " points.");
  return hull_cloud;
}

float compareObjectHullShapes(XYZPointCloud& hull_cloud_a,XYZPointCloud& hull_cloud_b)
{
  // Extract shape features from hull_a & hull_b and perform alignment
  double match_cost;
  cpl_visual_features::Path matches = compareBoundaryShapes(hull_cloud_a, hull_cloud_b, match_cost);
  // Visualize the matches & report the score
  // ROS_INFO_STREAM("Found minimum cost match of: " << match_cost);
  PushTrackerState state;
  cv::RotatedRect obj_ellipse;
  fitHullEllipse(hull_cloud_a, obj_ellipse);
  state.x.theta = getThetaFromEllipse(obj_ellipse);
  // Get (x,y) centroid of boundary
  state.x.x = obj_ellipse.center.x;
  state.x.y = obj_ellipse.center.y;
  cv::Mat match_img = visualizeObjectBoundaryMatches(hull_cloud_a, hull_cloud_b, state, matches);
  cv::imshow("Object boundary matches", match_img);
  cv::waitKey(3);
  return match_cost;
}

int mainComputeHeatKernelSignature(int argc, char** argv)
{
  if (argc < 2 || argc > 3)
  {
    ROS_INFO_STREAM("usage: " << argv[0] << " cloud_path [m]");
    return -1;
  }
  std::string cloud_path(argv[1]);
  int m = (argc == 3) ? atoi(argv[2]) : 1;
  std::vector<std::vector<float> > match_scores;
  // Read in point clouds for a & b and extract the hulls
  XYZPointCloud hull_cloud = getHullFromPCDFile(cloud_path);

  PushTrackerState state;
  cv::RotatedRect obj_ellipse;
  fitHullEllipse(hull_cloud, obj_ellipse);
  state.x.theta = getThetaFromEllipse(obj_ellipse);
  // Get (x,y) centroid of boundary
  state.x.x = obj_ellipse.center.x;
  state.x.y = obj_ellipse.center.y;
  cv::Mat hull_img = visualizeObjectBoundarySamples(hull_cloud, state);
  cv::imshow("Input hull", hull_img);

  // Run laplacian smoothing
  // XYZPointCloud smoothed_hull_cloud = laplacianSmoothBoundary(hull_cloud, m);
  XYZPointCloud smoothed_hull_cloud = laplacianBoundaryCompression(hull_cloud, m);
  cv::Mat smoothed_hull_img = visualizeObjectBoundarySamples(smoothed_hull_cloud, state);
  std::stringstream smoothed_hull_name;
  smoothed_hull_name << "smoothed_hull_" << m;
  cv::imshow(smoothed_hull_name.str(), smoothed_hull_img);
  cv::waitKey();
  return 0;
}

int mainCompareShapeContext(int argc, char** argv)
{
  // Parse file names for a and b
  if (argc != 5)
  {
    ROS_INFO_STREAM("usage: " << argv[0] << " class_a_cloud_path class_b_cloud_path num_a num_b");
    return -1;
  }
  // TODO: Cycle through multiple directories comparing values and computing statistics
  std::string cloud_a_base_path(argv[1]);
  std::string cloud_b_base_path(argv[2]);
  int num_a = atoi(argv[3]);
  int num_b = atoi(argv[4]);
  std::vector<std::vector<float> > match_scores;
  float score_sum = 0;
  for (int a = 0; a < num_a; ++a)
  {
    std::stringstream cloud_a_path;
    cloud_a_path << cloud_a_base_path << a << ".pcd";
    // Read in point clouds for a & b and extract the hulls
    XYZPointCloud hull_cloud_a = getHullFromPCDFile(cloud_a_path.str());
    std::vector<float> scores_b;
    float score_sum_b = 0;
    for (int b = 0; b < num_b; ++b)
    {
      std::stringstream cloud_b_path;
      cloud_b_path << cloud_b_base_path << b << ".pcd";
      XYZPointCloud hull_cloud_b = getHullFromPCDFile(cloud_b_path.str());
      float match_score = compareObjectHullShapes(hull_cloud_a, hull_cloud_b);
      scores_b.push_back(match_score);
      score_sum += match_score;
      score_sum_b += match_score;
    }
    match_scores.push_back(scores_b);
    ROS_INFO_STREAM("Mean score: " << score_sum_b/num_b);
  }
  ROS_INFO_STREAM("Overall mean score: " << score_sum/(num_b*num_a));
  // TODO: Examine interclass and intraclass distributions of score matches
  return 0;
}

int main(int argc, char** argv)
{
  // return mainCompareShapeContext(argc, argv);
  return mainComputeHeatKernelSignature(argc, argv);
}
