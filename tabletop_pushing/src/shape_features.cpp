#include <sstream>
#include <iostream>
#include <fstream>
#include <tabletop_pushing/shape_features.h>
#include <pcl16_ros/transforms.h>
#include <pcl16/ros/conversions.h>
#include <pcl16/surface/concave_hull.h>
#include <iostream>

#define XY_RES 0.00075
using namespace cpl_visual_features;
using tabletop_pushing::ProtoObject;
namespace tabletop_pushing
{

cv::Point lineLineIntersection(cv::Point a1, cv::Point a2, cv::Point b1, cv::Point b2)
{
  float denom = (a1.x-a2.x)*(b1.y-b2.y)-(a1.y-a2.y)*(b1.x-b2.x);
  if (denom == 0) // Parrallel lines, return somethign else
  {
  }
  cv::Point intersection( ((a1.x*a2.y - a1.y*a2.x)*(b1.x-b2.x) -
                           (a1.x - a2.x)*(b1.x*b2.y - b1.y*b2.x))/denom,
                          ((a1.x*a2.y - a1.y*a2.x)*(b1.y-b2.y) -
                           (a1.y - a2.y)*(b1.x*b2.y - b1.y*b2.x))/denom);
  return intersection;
}

inline int worldLocToIdx(double val, double min_val, double max_val)
{
  return round((val-min_val)/XY_RES);
}

cv::Point worldPtToImgPt(pcl16::PointXYZ world_pt, double min_x, double max_x,
                         double min_y, double max_y)
{
  cv::Point img_pt(worldLocToIdx(world_pt.x, min_x, max_x),
                   worldLocToIdx(world_pt.y, min_y, max_y));
  return img_pt;
}

XYZPointCloud getObjectBoundarySamples(ProtoObject& cur_obj)
{
  // Get 2D projection of object
  // TODO: Remove the z, then add it back after finding the hull... how do we do this?
  XYZPointCloud footprint_cloud(cur_obj.cloud);
  for (int i = 0; i < footprint_cloud.size(); ++i)
  {
    // HACK: This is a complete hack, based on the current table height used in pushing.
    footprint_cloud.at(i).z = -0.3;
  }

  // TODO: Examine sensitivity of hull_alpha...
  double hull_alpha = 0.01;
  XYZPointCloud hull_cloud;
  pcl16::ConcaveHull<pcl16::PointXYZ> hull;
  hull.setDimension(2);  // NOTE: Get 2D projection of object
  hull.setInputCloud(footprint_cloud.makeShared());
  hull.setAlpha(hull_alpha);
  hull.reconstruct(hull_cloud);

  return hull_cloud;
}

cv::Mat getObjectFootprint(cv::Mat obj_mask, pcl16::PointCloud<pcl16::PointXYZ>& cloud)
{
  cv::Mat kernel(5,5,CV_8UC1, 255);
  cv::Mat obj_mask_target;
  obj_mask.copyTo(obj_mask_target);
  // Perform close
  cv::dilate(obj_mask, obj_mask_target, kernel);
  cv::erode(obj_mask_target, obj_mask, kernel);
  // cv::erode(obj_mask, obj_mask, kernel);
  double min_x = 300., max_x = -300.;
  double min_y = 300., max_y = -300.;

  for (int r = 0; r < obj_mask.rows; ++r)
  {
    for (int c = 0; c < obj_mask.cols; ++c)
    {
      if (obj_mask.at<uchar>(r,c) > 0)
      {
        double x = cloud.at(c,r).x;
        double y = cloud.at(c,r).y;
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
        double x = cloud.at(c,r).x;
        double y = cloud.at(c,r).y;
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
  // Perform open
  cv::erode(footprint, footprint, kernel);
  cv::dilate(footprint, footprint, kernel);
  return footprint;
}

ShapeLocations extractObjectShapeContext(ProtoObject& cur_obj, bool use_center)
{
  XYZPointCloud samples_pcl = getObjectBoundarySamples(cur_obj);
  return extractShapeContextFromSamples(samples_pcl, cur_obj, use_center);
}

ShapeLocations extractShapeContextFromSamples(XYZPointCloud& samples_pcl, ProtoObject& cur_obj, bool use_center)
{
  Samples2f samples;
  for (unsigned int i = 0; i < samples_pcl.size(); ++i)
  {
    cv::Point2f pt(samples_pcl[i].x, samples_pcl[i].y);
    samples.push_back(pt);
  }
  int radius_bins = 5;
  int theta_bins = 12;
  double max_radius = 0.5;
  cv::Point2f center(cur_obj.centroid[0], cur_obj.centroid[1]);
  ShapeDescriptors descriptors = constructDescriptors(samples, center, use_center, radius_bins, theta_bins, max_radius);
  ShapeLocations locs;
  for (unsigned int i = 0; i < descriptors.size(); ++i)
  {
    ShapeLocation loc(samples_pcl[i], descriptors[i]);
    locs.push_back(loc);
  }
  computeShapeFeatureAffinityMatrix(locs, use_center);
  return locs;
}

bool pointIsBetweenOthers(pcl16::PointXYZ& pt, pcl16::PointXYZ& x1, pcl16::PointXYZ& x2)
{
  // Project the vector pt->x2 onto the vector x1->x2
  const float a_x = x2.x - pt.x;
  const float a_y = x2.y - pt.y;
  const float b_x = x2.x - x1.x;
  const float b_y = x2.y - x1.y;
  const float a_dot_b = a_x*b_x + a_y*b_y;
  const float b_dot_b = b_x*b_x + b_y*b_y;
  const float a_onto_b = a_dot_b/b_dot_b;

  // If the (squared) distance of the projection is less than the vector from x1->x2 then it is between them
  const float d_1_x = a_onto_b*b_x;
  const float d_1_y = a_onto_b*b_y;
  const float d_1 = std::sqrt(d_1_x*d_1_x + d_1_y*d_1_y);
  const float d_2 = std::sqrt(b_x*b_x + b_y*b_y);

  return d_1 < d_2;
}

void drawSamplePoints(XYZPointCloud& hull, XYZPointCloud& samples, pcl16::PointXYZ& center_pt,
                      pcl16::PointXYZ& sample_pt, pcl16::PointXYZ& approach_pt,
                      pcl16::PointXYZ e_left, pcl16::PointXYZ e_right,
                      pcl16::PointXYZ c_left, pcl16::PointXYZ c_right)
{
  double max_y = 0.5;
  double min_y = -0.5;
  double max_x = 1.0;
  double min_x = 0.0;
  // TODO: Make function to get cv::Size from (max_x, min_x, max_y, min_y, XY_RES)
  int rows = ceil((max_y-min_y)/XY_RES);
  int cols = ceil((max_x-min_x)/XY_RES);
  cv::Mat footprint(rows, cols, CV_8UC3, cv::Scalar(0.0,0.0,0.0));

  for (int i = 0; i < hull.size(); ++i)
  {
    pcl16::PointXYZ obj_pt = hull[i];
    cv::Point img_pt = worldPtToImgPt(obj_pt, min_x, max_x, min_y, max_y);
    cv::Scalar color(0, 0, 128);
    cv::circle(footprint, img_pt, 1, color);
  }
  for (int i = 0; i < samples.size(); ++i)
  {
    cv::Point img_pt = worldPtToImgPt(samples[i], min_x, max_x, min_y, max_y);
    cv::Scalar color(0, 255, 0);
    cv::circle(footprint, img_pt, 3, color);
  }
  cv::Point img_center = worldPtToImgPt(center_pt, min_x, max_x, min_y, max_y);
  cv::Point img_approach_pt = worldPtToImgPt(approach_pt, min_x, max_x, min_y, max_y);
  cv::Point img_sample_pt = worldPtToImgPt(sample_pt, min_x, max_x, min_y, max_y);
  cv::Point e_left_img = worldPtToImgPt(e_left, min_x, max_x, min_y, max_y);
  cv::Point e_right_img = worldPtToImgPt(e_right, min_x, max_x, min_y, max_y);
  cv::Point c_left_img = worldPtToImgPt(c_left, min_x, max_x, min_y, max_y);
  cv::Point c_right_img = worldPtToImgPt(c_right, min_x, max_x, min_y, max_y);

  cv::circle(footprint, img_center, 3, cv::Scalar(0,255,255));
  cv::circle(footprint, img_approach_pt, 3, cv::Scalar(0,255,255));
  cv::circle(footprint, e_left_img, 3, cv::Scalar(0,255,255));
  cv::circle(footprint, e_right_img, 3, cv::Scalar(0,255,255));
  cv::line(footprint, img_approach_pt, img_center, cv::Scalar(0,255,255));
  cv::line(footprint, e_left_img, e_right_img, cv::Scalar(0,255,255));
  cv::line(footprint, e_left_img, c_left_img, cv::Scalar(0,255,255));
  cv::line(footprint, e_right_img, c_right_img, cv::Scalar(0,255,255));
  
  // Draw sample point last
  cv::line(footprint, img_sample_pt, img_center, cv::Scalar(255,255,255));
  cv::circle(footprint, img_sample_pt, 3, cv::Scalar(255,255,255));

  cv::imshow("local samples", footprint);
  cv::waitKey();
}

XYZPointCloud getLocalSamples(XYZPointCloud& hull, ProtoObject& cur_obj, pcl16::PointXYZ sample_pt,
                              float sample_spread)
{
  // TODO: This is going to get all points in front of the gripper, not only those on the close boundary
  float radius = sample_spread / 2.0;
  pcl16::PointXYZ center_pt(cur_obj.centroid[0], cur_obj.centroid[1], cur_obj.centroid[2]);
  float center_angle = std::atan2(center_pt.y - sample_pt.y, center_pt.x - sample_pt.x);
  float approach_dist = 0.15;
  pcl16::PointXYZ approach_pt(sample_pt.x - std::cos(center_angle)*approach_dist,
                              sample_pt.y - std::sin(center_angle)*approach_dist, 0.0);
  pcl16::PointXYZ e_vect(std::cos(center_angle+M_PI/2.0)*radius,
                         std::sin(center_angle+M_PI/2.0)*radius, 0.0);
  pcl16::PointXYZ e_left(approach_pt.x + e_vect.x, approach_pt.y + e_vect.y, 0.0);
  pcl16::PointXYZ e_right(approach_pt.x - e_vect.x, approach_pt.y - e_vect.y, 0.0);
  pcl16::PointXYZ c_left(center_pt.x + e_vect.x, center_pt.y + e_vect.y, 0.0);
  pcl16::PointXYZ c_right(center_pt.x - e_vect.x, center_pt.y - e_vect.y, 0.0);
  ROS_INFO_STREAM("center_pt: " << center_pt);
  ROS_INFO_STREAM("sample_pt: " << sample_pt);
  ROS_INFO_STREAM("approach_pt: " << approach_pt);
  ROS_INFO_STREAM("e_vect: " << e_vect);
  ROS_INFO_STREAM("e_left: " << e_left);
  ROS_INFO_STREAM("e_right: " << e_right);

  std::vector<pcl16::PointXYZ> left_segments;
  std::vector<pcl16::PointXYZ> right_segments;
  // TODO: Test intersection of gripper end point rays and all line segments on the object boundary
  // TODO: Determine which intersection is closest

  // TODO: Walk from left intersection to right intersection
  bool sample_in_walk = false;
  std::vector<int> indices;
  for (int i = 0; i < hull.size(); ++i)
  {
    // TODO: Test if sample_pt is in the outside
  }

  // Copy to new cloud and return
  XYZPointCloud local_samples;
  pcl16::copyPointCloud(hull, indices, local_samples);
  drawSamplePoints(hull, local_samples, center_pt, sample_pt, approach_pt, e_left, e_right,
                   c_left, c_right);
  return local_samples;
}

ShapeDescriptor extractLocalShapeFeatures(XYZPointCloud& hull, ProtoObject& cur_obj,
                                          pcl16::PointXYZ sample_pt, float sample_spread)
{
  XYZPointCloud local_samples = getLocalSamples(hull, cur_obj, sample_pt, sample_spread);
  ShapeDescriptor sd;
  return sd;
}

// ShapeDescriptor extractGlobalShapeFeatures(XYZPointCloud& samples_pcl, ProtoObject& cur_obj,
//                                            pcl16::PointXYZ sample_loc, float sample_spread)
// {
// }


/**
 * Create an affinity matrix for a set of ShapeLocations
 *
 * @param locs The vector of ShapeLocation descriptors to compare
 *
 * @return An upper-triangular matrix of all pairwise distances between descriptors
 */
cv::Mat computeShapeFeatureAffinityMatrix(ShapeLocations& locs, bool use_center)
{
  cv::Mat affinity(locs.size(), locs.size(), CV_64FC1, cv::Scalar(0.0));
  double max_affinity = 0.0;
  ShapeDescriptors normalized;
  for (int i = 0; i < locs.size(); ++i)
  {
    ShapeDescriptor desc(locs[i].descriptor_);
    double feature_sum = 0;
    for (int j = 0; j < desc.size(); ++j)
    {
      feature_sum += desc[j];
    }
    if (feature_sum == 0) continue;
    for (int j = 0; j < desc.size(); ++j)
    {
      desc[j] = sqrt(desc[j]/feature_sum);
    }
    normalized.push_back(desc);
  }

  for (int r = 0; r < affinity.rows; ++r)
  {
    for (int c = r; c < affinity.cols; ++c)
    {
      if (r == c)
      {
        affinity.at<double>(r,c) = 1.0;
        continue;
      }
      double sim_score = 1.0 - sqrt(shapeFeatureSquaredEuclideanDist(normalized[r],normalized[c]));
      affinity.at<double>(r,c) = sim_score;
      affinity.at<double>(c,r) = sim_score;
      if (affinity.at<double>(r,c) > max_affinity)
      {
        max_affinity = affinity.at<double>(r,c);
      }
    }
  }
  if (use_center)
  {
    cv::imshow("affinity-with-centers", affinity);
  }
  else
  {
    cv::imshow("affinity", affinity);
  }
  return affinity;
}

/**
 * Compute the (chi-squared) distance between two ShapeLocation descriptors
 *
 * @param a The first descriptor
 * @param b The second descriptor
 *
 * @return The distance between a and b
 */
double shapeFeatureChiSquareDist(ShapeDescriptor& a, ShapeDescriptor& b)
{
  // compute affinity between shape context i and j
  // using chi-squared test statistic
  double d_affinity = 0;

  double a_sum = 0.0;
  double b_sum = 0.0;
  for (unsigned int k=0; k < a.size(); k++)
  {
    a_sum += a[k];
    b_sum += b[k];
  }
  if (a_sum == 0.0)
  {
    a_sum = 1.0;
  }
  if (b_sum == 0.0)
  {
    b_sum = 1.0;
  }
  for (unsigned int k=0; k < a.size(); k++)
  {
    // NOTE: Normalizing to have L1 of 1 for comparison
    const double a_k = a[k]/a_sum;
    const double b_k = b[k]/b_sum;
    const double a_plus_b = a_k + b_k;
    if (a_plus_b > 0)
    {
      d_affinity += pow(a_k - b_k, 2) / (a_plus_b);
    }
  }
  d_affinity = d_affinity/2;
  return d_affinity;
}

/**
 * Compute the squared euclidean distance between two ShapeLocation descriptors
 *
 * @param a The first descriptor
 * @param b The second descriptor
 *
 * @return The distance between a and b
 */
double shapeFeatureSquaredEuclideanDist(ShapeDescriptor& a, ShapeDescriptor& b)
{
  double dist = 0;
  for(int k = 0; k < a.size(); ++k)
  {
    double k_dist = a[k] - b[k];
    dist += k_dist*k_dist;
  }
  return dist;
}

void clusterShapeFeatures(ShapeLocations& locs, int num_clusters, std::vector<int>& cluster_ids, ShapeDescriptors& centers,
                          double min_err_change, int max_iter, int num_retries)
{
  cv::Mat samples(locs.size(), locs[0].descriptor_.size(), CV_32FC1);
  for (int r = 0; r < samples.rows; ++r)
  {
    // NOTE: Normalize features here
    float feature_sum = 0;
    for (int c = 0; c < samples.cols; ++c)
    {
      samples.at<float>(r,c) = locs[r].descriptor_[c];
      feature_sum += samples.at<float>(r,c);
    }
    if (feature_sum == 0)
    {
      continue;
    }
    for (int c = 0; c < samples.cols; ++c)
    {
      samples.at<float>(r,c) /= feature_sum;
      // NOTE: Use Hellinger distance for comparison
      samples.at<float>(r,c) = sqrt(samples.at<float>(r,c));
    }
  }
  cv::TermCriteria term_crit(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, max_iter, min_err_change);
  // cv::Mat labels;
  cv::Mat centers_cv;
  double slack = cv::kmeans(samples, num_clusters, cluster_ids, term_crit, num_retries, cv::KMEANS_PP_CENTERS,
                            centers_cv);
  for (int r = 0; r < centers_cv.rows; ++r)
  {
    ShapeDescriptor s(centers_cv.cols, 0);
    for (int c = 0; c < centers_cv.cols; ++c)
    {
      s[c] = centers_cv.at<float>(r,c);
    }
    centers.push_back(s);
  }
}

/**
 * Find the nearest cluster center to the given descriptor
 *
 * @param descriptor The query descriptor
 * @param centers The set of cluster centers
 *
 * @return The cluster id (index) of the nearest center
 */
int closestShapeFeatureCluster(ShapeDescriptor& descriptor, ShapeDescriptors& centers, double& min_dist)
{
  int min_idx = -1;
  min_dist = FLT_MAX;
  ShapeDescriptor normalized(descriptor);
  float feature_sum = 0;
  for (int i = 0; i < normalized.size(); ++i)
  {
    feature_sum += normalized[i];
  }
  for (int i = 0; i < normalized.size(); ++i)
  {
    normalized[i] = sqrt(normalized[i]/feature_sum);
  }
  for (int c = 0; c < centers.size(); ++c)
  {
    double c_dist = (shapeFeatureSquaredEuclideanDist(normalized, centers[c]));
    if (c_dist < min_dist)
    {
      min_dist = c_dist;
      min_idx = c;
    }
  }
  return min_idx;
}

};
