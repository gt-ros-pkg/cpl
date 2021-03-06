/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014, Georgia Institute of Technology
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Georgia Institute of Technology nor the names of
 *     its contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/
#ifndef point_cloud_segmentation_h_DEFINED
#define point_cloud_segmentation_h_DEFINED

// ROS Message Types
#include <std_msgs/Header.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose2D.h>

// TF
#include <tf/transform_listener.h>

// PCL
#include <pcl16/point_cloud.h>
#include <pcl16/point_types.h>
#include <pcl16/common/eigen.h>
#include <pcl16/common/io.h>
#include <pcl16/ModelCoefficients.h>

// OpenCV
#include <opencv2/core/core.hpp>

// STL
#include <vector>
#include <deque>
#include <string>
#include <math.h>

// Boost
#include <boost/shared_ptr.hpp>

namespace tabletop_pushing
{

typedef pcl16::PointCloud<pcl16::PointXYZ> XYZPointCloud;
typedef pcl16::PointCloud<pcl16::Normal> NormalCloud;
class ProtoObject
{
 public:
  XYZPointCloud cloud;
  NormalCloud normals;
  Eigen::Vector4f centroid;
  int id;
  bool moved;
  Eigen::Matrix4f transform;
  std::vector<int> boundary_angle_dist;
  std::vector<int> push_history;
  bool singulated;
  double icp_score;
};
typedef std::deque<ProtoObject> ProtoObjects;

class PointCloudSegmentation
{
 public:
  PointCloudSegmentation(boost::shared_ptr<tf::TransformListener> tf);

  /**
   * Function to find the table plane in the image by plane RANSAC
   *
   * @param cloud           The input cloud
   * @param objs_cloud      [Returned] the cloud not containing the table plane
   * @param plane_cloud     [Returned] the cloud containing the plane
   * @param center          [Returned] the center of the table plane
   * @param init_table_find set true to have a greater search space for finding table plane
   * @param find_hull       set true to estimate the extent of the plane
   * @param find_centroid   set true to estimate the extent of the plane
   */
  void getTablePlane(XYZPointCloud& cloud, XYZPointCloud& objs_cloud,
                     XYZPointCloud& plane_cloud, Eigen::Vector4f& center,
                     bool init_table_find=false,
                     bool find_hull=false, bool find_centroid=false);

  /**
   * Function to find the table plane in the image by using multi-plane segmentation (requires structured input cloud)
   *
   * @param cloud       The input cloud
   * @param objs_cloud  [Returned] the cloud not containing the table plane
   * @param plane_cloud [Returned] the cloud containing the plane
   * @param center      [Returned] the center of the table plane
   * @param find_hull   set true to estimate the extent of the plane
   */
  void getTablePlaneMPS(XYZPointCloud& cloud, XYZPointCloud& objs_cloud,
                        XYZPointCloud& plane_cloud, Eigen::Vector4f& center,
                        bool find_hull=false, bool find_centroid=false);

  /**
   * Function to segment independent spatial regions from a supporting plane
   *
   * @param input_cloud   The point cloud to operate on.
   * @param objs          [Returned] The object clusters.
   * @param use_mps       If true then mps is used instead of RANSAC to get the table plane
   */
  void findTabletopObjects(XYZPointCloud& input_cloud, ProtoObjects& objs, bool use_mps=false);

  /**
   * Function to segment independent spatial regions from a supporting plane
   *
   * @param input_cloud   The point cloud to operate on.
   * @param objs          [Returned] The object clusters.
   * @param objs_cloud    The point cloud containing the object points.
   * @param use_mps       If true then mps is used instead of RANSAC to get the table plane
   */
  void findTabletopObjects(XYZPointCloud& input_cloud, ProtoObjects& objs,
                           XYZPointCloud& objs_cloud, bool use_mps=false);

  /**
   * Function to segment independent spatial regions from a supporting plane
   *
   * @param input_cloud   The point cloud to operate on.
   * @param objs          [Returned] The object clusters.
   * @param objs_cloud    [Returned] The point cloud containing the object points.
   * @param plane_cloud   [Returned] The point cloud containing the table plane points.
   * @param use_mps       If true then mps is used instead of RANSAC to get the table plane
   */
  void findTabletopObjects(XYZPointCloud& input_cloud, ProtoObjects& objs,
                           XYZPointCloud& objs_cloud,
                           XYZPointCloud& plane_cloud, bool use_mps=false);

  /**
   * Segment the spactial regions from the plan, but first reduce the search space based on previous object pose
   *
   * @param input_cloud The point cloud to operate on.
   * @param objs        [Returned] The object clusters.
   * @param prev_state  The previous object state estimate to guide the segmentation
   * @param search_radius How far to look in x and y from center of prev_state
   */
  void findTabletopObjectsRestricted(XYZPointCloud& input_cloud, ProtoObjects& objs,
                                     geometry_msgs::Pose2D& prev_state,
                                     double search_radius);


  /**
   * Function to segment point cloud regions using euclidean clustering
   *
   * @param objects_cloud The cloud of objects to cluster
   * @param objs          [Returned] The independent clusters
   */
  void clusterProtoObjects(XYZPointCloud& objects_cloud, ProtoObjects& objs);

  /**
   * Perform Iterated Closest Point between two proto objects.
   *
   * @param a The first object
   * @param b The second object
   *
   * @return The ICP fitness score of the match
   */
  double ICPProtoObjects(ProtoObject& a, ProtoObject& b, Eigen::Matrix4f& transform);

  double ICPBoundarySamples(XYZPointCloud& hull_t_0, XYZPointCloud& hull_t_1,
                            Eigen::Matrix4f& init_transform, Eigen::Matrix4f& transform,
                            XYZPointCloud& aligned);

  /**
   * Find the regions that have moved between two point clouds
   *
   * @param prev_cloud    The first cloud to use in differencing
   * @param cur_cloud     The second cloud to use
   * @param moved_regions [Returned] The new set of objects that have moved in the second cloud
   * @param suf           [Optional]
   */
  void getMovedRegions(XYZPointCloud& prev_cloud, XYZPointCloud& cur_cloud, ProtoObjects& moved_regions,
                       std::string suf="");

  /**
   * Match moved regions to previously extracted protoobjects
   *
   * @param objs The previously extracted objects
   * @param moved_regions The regions that have been detected as having moved
   *
   */
  void matchMovedRegions(ProtoObjects& objs, ProtoObjects& moved_regions);

  void fitCylinderRANSAC(ProtoObject& obj, XYZPointCloud& cylinder_cloud, pcl16::ModelCoefficients& cylinder);
  void fitSphereRANSAC(ProtoObject& obj, XYZPointCloud& sphere_cloud, pcl16::ModelCoefficients& sphere);

  static inline double dist(pcl16::PointXYZ a, pcl16::PointXYZ b)
  {
    const double dx = a.x-b.x;
    const double dy = a.y-b.y;
    const double dz = a.z-b.z;
    return std::sqrt(dx*dx+dy*dy+dz*dz);
  }

  static inline double dist(geometry_msgs::Point b, pcl16::PointXYZ a)
  {
    return dist(a,b);
  }

  static inline double dist(pcl16::PointXYZ a, geometry_msgs::Point b)
  {
    const double dx = a.x-b.x;
    const double dy = a.y-b.y;
    const double dz = a.z-b.z;
    return std::sqrt(dx*dx+dy*dy+dz*dz);
  }

  static inline double sqrDist(Eigen::Vector3f& a, pcl16::PointXYZ& b)
  {
    const double dx = a[0]-b.x;
    const double dy = a[1]-b.y;
    const double dz = a[2]-b.z;
    return dx*dx+dy*dy+dz*dz;
  }

  static inline double sqrDist(Eigen::Vector4f& a, Eigen::Vector4f& b)
  {
    const double dx = a[0]-b[0];
    const double dy = a[1]-b[1];
    const double dz = a[2]-b[2];
    return dx*dx+dy*dy+dz*dz;
  }

  static inline double sqrDist(pcl16::PointXYZ a, pcl16::PointXYZ b)
  {
    const double dx = a.x-b.x;
    const double dy = a.y-b.y;
    const double dz = a.z-b.z;
    return dx*dx+dy*dy+dz*dz;
  }

  static inline double sqrDist(pcl16::PointXYZ a, Eigen::Vector4f b)
  {
    const double dx = a.x-b[0];
    const double dy = a.y-b[1];
    const double dz = a.z-b[2];
    return dx*dx+dy*dy+dz*dz;
  }

  static inline double sqrDistXY(pcl16::PointXYZ a, pcl16::PointXYZ b)
  {
    const double dx = a.x-b.x;
    const double dy = a.y-b.y;
    return dx*dx+dy*dy;
  }

  /**
   * Naively determine if two point clouds intersect based on distance threshold
   * between points.
   *
   * @param cloud0 First cloud for interesection test
   * @param cloud1 Second cloud for interesection test
   *
   * @return true if any points from cloud0 and cloud1 have distance less than
   * voxel_down_res_
   */
  bool cloudsIntersect(XYZPointCloud cloud0, XYZPointCloud cloud1);

  bool cloudsIntersect(XYZPointCloud cloud0, XYZPointCloud cloud1,
                       double thresh);

  bool pointIntersectsCloud(XYZPointCloud cloud, geometry_msgs::Point pt,
                            double thresh);

  float pointLineXYDist(pcl16::PointXYZ p,Eigen::Vector3f vec,Eigen::Vector4f base);

  void lineCloudIntersection(XYZPointCloud& cloud, Eigen::Vector3f vec,
                             Eigen::Vector4f base, XYZPointCloud& line_cloud);
  void lineCloudIntersectionEndPoints(XYZPointCloud& cloud, Eigen::Vector3f vec, Eigen::Vector4f base,
                                      std::vector<pcl16::PointXYZ>& end_points);

  /**
   * Filter a point cloud to only be above the estimated table and within the
   * workspace in x, then downsample the voxels.
   *
   * @param cloud_in   The cloud to filter and downsample
   * @param cloud_down [Returned] The downsampled cloud
   */
  void downsampleCloud(XYZPointCloud& cloud_in, XYZPointCloud& cloud_down);

  /**    * Filter the cloud then downsample
   *
   * @param cloud_in   The cloud to filter and downsample
   * @param cloud_down [Returned] The downsampled cloud
   * @param min_x      Min x for filter
   * @param max_x      Max x for filter
   * @param min_y      Min y for filter
   * @param max_y      Max y for filter
   * @param min_z      Min z for filter
   * @param max_z      Max z for filter
   * @param filter_y   If true filter in y direction, default to false
   */
  void downsampleCloud(XYZPointCloud& cloud_in, XYZPointCloud& cloud_down,
                       double min_x, double max_x,
                       double min_y, double max_y,
                       double min_z, double max_z, bool filter_y = false);

  /**
   * Method to project the current proto objects into an image
   *
   * @param objs The set of objects
   * @param img_in An image of correct size for the projection
   * @param target_frame The frame of the associated image
   *
   * @return Image containing the projected objects
   */
  cv::Mat projectProtoObjectsIntoImage(ProtoObjects& objs, cv::Size img_size, std::string target_frame);

  /**
   * Method to project the current proto object into an image
   *
   * @param obj The objects
   * @param img_in An image of correct size for the projection
   * @param target_frame The frame of the associated image
   *
   * @return Image containing the projected object
   */
  cv::Mat projectProtoObjectIntoImage(ProtoObject& obj, cv::Size img_size, std::string target_frame);

  /**
   * Visualization function of proto objects projected into an image
   *
   * @param obj_img The projected objects image
   * @param objs The set of proto objects
   */
  cv::Mat displayObjectImage(cv::Mat& obj_img,
                             std::string win_name="projected objects",
                             bool use_display=true);

  void projectPointCloudIntoImage(XYZPointCloud& cloud, cv::Mat& lbl_img,
                                  std::string target_frame, unsigned int id=1);

  cv::Point projectPointIntoImage(pcl16::PointXYZ cur_point_pcl,
                                  std::string point_frame,
                                  std::string target_frame);

  void projectPointCloudIntoImage(XYZPointCloud& cloud, cv::Mat& lbl_img);

  cv::Point projectPointIntoImage(geometry_msgs::PointStamped cur_point);

  cv::Point projectPointIntoImage(geometry_msgs::PointStamped cur_point,
                                  std::string target_frame);
  cv::Point projectPointIntoImage(Eigen::Vector3f cur_point_eig,
                                  std::string point_frame, std::string target_frame);

  Eigen::Vector4f getTableCentroid() const
  {
    return table_centroid_;
  }

 protected:
  boost::shared_ptr<tf::TransformListener> tf_;
  Eigen::Vector4f table_centroid_;

 public:
  std::vector<cv::Vec3f> colors_;
  double min_table_z_;
  double max_table_z_;
  double min_workspace_x_;
  double max_workspace_x_;
  double min_workspace_y_;
  double max_workspace_y_;
  double min_workspace_z_;
  double max_workspace_z_;
  double table_z_;
  double table_ransac_thresh_;
  double table_ransac_angle_thresh_;
  double cluster_tolerance_;
  double cloud_diff_thresh_;
  int min_cluster_size_;
  int max_cluster_size_;
  double voxel_down_res_;
  double cloud_intersect_thresh_;
  double hull_alpha_;
  bool use_voxel_down_;
  int num_downsamples_;
  sensor_msgs::CameraInfo cam_info_;
  std_msgs::Header cur_camera_header_;
  int moved_count_thresh_;
  int icp_max_iters_;
  double icp_transform_eps_;
  double icp_max_cor_dist_;
  double icp_ransac_thresh_;
  int mps_min_inliers_;
  double mps_min_angle_thresh_;
  double mps_min_dist_thresh_;
  double cylinder_ransac_thresh_;
  double cylinder_ransac_angle_thresh_;
  double sphere_ransac_thresh_;
  bool optimize_cylinder_coefficients_;
};

void getMaskedPointCloud(XYZPointCloud& input_cloud, cv::Mat& mask, XYZPointCloud& masked_cloud)
{
  // Select points from point cloud that are in the mask:
  pcl16::PointIndices mask_indices;
  mask_indices.header = input_cloud.header;
  for (int y = 0; y < mask.rows; ++y)
  {
    uchar* mask_row = mask.ptr<uchar>(y);
    for (int x = 0; x < mask.cols; ++x)
    {
      if (mask_row[x] != 0)
      {
        mask_indices.indices.push_back(y*input_cloud.width + x);
      }
    }
  }

  pcl16::copyPointCloud(input_cloud, mask_indices, masked_cloud);
}
};
#endif // point_cloud_segmentation_h_DEFINED
