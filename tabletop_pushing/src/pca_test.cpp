#include <ros/ros.h>
// PCL
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <pcl16/common/pca.h>
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
typedef pcl16::PointCloud<pcl16::PointXYZ> XYZPointCloud;

cv::RotatedRect fit2DMassEllipse(XYZPointCloud& obj_cloud)
{
  pcl16::PCA<pcl16::PointXYZ> pca;
  XYZPointCloud cloud_no_z;
  cloud_no_z.header = obj_cloud.header;
  cloud_no_z.width = obj_cloud.size();
  cloud_no_z.height = 1;
  cloud_no_z.is_dense = false;
  cloud_no_z.resize(obj_cloud.size());
  if (obj_cloud.size() < 3)
  {
    ROS_WARN_STREAM("Too few points to find ellipse");
    cv::RotatedRect obj_ellipse;
    obj_ellipse.center.x = 0.0;
    obj_ellipse.center.y = 0.0;
    obj_ellipse.angle = 0;
    obj_ellipse.size.width = 0;
    obj_ellipse.size.height = 0;
    return obj_ellipse;
  }
  for (unsigned int i = 0; i < obj_cloud.size(); ++i)
  {
    cloud_no_z[i] = obj_cloud[i];
    cloud_no_z[i].z = 0.0f;
  }

  pca.setInputCloud(cloud_no_z.makeShared());
  Eigen::Vector3f eigen_values;
  Eigen::Matrix3f eigen_vectors;
  Eigen::Vector4f centroid;
  try{
    ROS_INFO_STREAM("Getting mean");
    centroid = pca.getMean();
    ROS_INFO_STREAM("Getting eiven values");
    eigen_values = pca.getEigenValues();
    ROS_INFO_STREAM("Getting eiven vectors");
    eigen_vectors = pca.getEigenVectors();
  } catch(pcl16::InitFailedException ife)
  {
    ROS_WARN_STREAM("Failed to compute PCA");
    ROS_WARN_STREAM("ife: " << ife.what());
  }

  cv::RotatedRect obj_ellipse;
  obj_ellipse.center.x = centroid[0];
  obj_ellipse.center.y = centroid[1];
  obj_ellipse.angle = RAD2DEG(atan2(eigen_vectors(1,0), eigen_vectors(0,0))-0.5*M_PI);
  // NOTE: major axis is defined by height
  obj_ellipse.size.height = std::max(eigen_values(0)*0.1, 0.07);
  obj_ellipse.size.width = std::max(eigen_values(1)*0.1, 0.03);
  ROS_INFO_STREAM("Center: (" << obj_ellipse.center.x << ", " << obj_ellipse.center.y << ")");
  ROS_INFO_STREAM("angle: " << obj_ellipse.angle);
  ROS_INFO_STREAM("size: (" << obj_ellipse.size.height << ", " << obj_ellipse.size.width << ")");
  return obj_ellipse;
}

int main()
{
  XYZPointCloud input_cloud;
  input_cloud.width = 2000;
  input_cloud.height = 1;
  input_cloud.is_dense = false;
  input_cloud.resize(input_cloud.width*input_cloud.height);
  for (int i = 0; i < input_cloud.points.size(); ++i)
  {
    input_cloud[i].x = 2.0 * rand() / (RAND_MAX) - 1.0;
    input_cloud[i].y = 2.0 * rand() / (RAND_MAX) - 1.0;
    input_cloud[i].z = 1.0 * rand() / (RAND_MAX);
  }
  // TODO: Populate input_cloud
  cv::RotatedRect ellipse = fit2DMassEllipse(input_cloud);
  return 0;
}
