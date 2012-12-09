#include "arm_mapping/corresp_pc_utils.h"
#include "arm_mapping/factor_arm_mapping.h"
#include "geometry_msgs/PoseStamped.h"

#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/extract_indices.h>

using namespace arm_mapping;

void readImgPCPoseBag(char* filename, vector<Mat> &imgs, vector<PCRGB::Ptr> &pcs, 
                      vector<gtsam::Pose3> &poses, int num)
{
  printf("Reading bag...");
  rosbag::Bag bag_in(filename, rosbag::bagmode::Read);
  vector<string> topics;
  string pc_topic = "/pc";
  string color_img_topic = "/image";
  string pose_topic = "/pose";
  topics.push_back(pc_topic);
  topics.push_back(color_img_topic);
  topics.push_back(pose_topic);
  rosbag::View view(bag_in, rosbag::TopicQuery(topics));
  num *= 3;
  BOOST_FOREACH(rosbag::MessageInstance const m, view) {
    if(m.getTopic() == pc_topic) {
      PCRGB::Ptr new_pc(new PCRGB());
      pcl::fromROSMsg<PRGB>(*m.instantiate<sensor_msgs::PointCloud2>(),*new_pc);
      pcs.push_back(new_pc);
    }
    if(m.getTopic() == color_img_topic) {
      sensor_msgs::Image::ConstPtr img = m.instantiate<sensor_msgs::Image>();
      cv_bridge::CvImagePtr cv_ptr;
      cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
      imgs.push_back(cv_ptr->image);
    }
    if(m.getTopic() == pose_topic) {
      geometry_msgs::PoseStamped::ConstPtr pose = m.instantiate<geometry_msgs::PoseStamped>();
      poses.push_back(geomPoseToGtsamPose3(pose->pose));
    }
    if(num >= 0)
      if(--num == 0)
        break;
  }
  bag_in.close();
  printf("done.\n");
}

void getCleanMatches(const Mat &img_1, const PCRGB &pc_1, const Mat &img_2, const PCRGB &pc_2,
                PCRGB &kp_all1, PCRGB &kp_all2)
{
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector< DMatch > matches, matches_pruned;
    PCRGB kps_1, kps_2;
    getMatches(img_1, img_2, keypoints_1, keypoints_2, matches);
    extractKeypoints(pc_1, keypoints_1, kps_1);
    extractKeypoints(pc_2, keypoints_2, kps_2);
    //visClouds(kps_1, kps_2);
    pruneMatches(kps_1, kps_2, matches, matches_pruned);
    extractMatches(kps_1, kps_2, matches_pruned, kp_all1, kp_all2);
}

pcl::PointIndices::Ptr RANSACInliers(const PCRGB& cloudA, const PCRGB& cloudB, double dist_thresh)
{
    pcl::SampleConsensusModelRegistration<PRGB>::Ptr sac_model(
        new pcl::SampleConsensusModelRegistration<PRGB>(cloudA.makeShared()));
    sac_model->setInputTarget(cloudB.makeShared());
    pcl::RandomSampleConsensus<PRGB> ransac(sac_model);
    ransac.setDistanceThreshold(dist_thresh);
    ransac.computeModel(1);
    Eigen::VectorXf coeffs;
    ransac.getModelCoefficients(coeffs);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
    Eigen::Matrix4f T = Eigen::Map<Eigen::Matrix4f>(coeffs.data(),4,4).transpose();
    cout<< "RESULT RANSAC " << endl <<  T;
    ransac.getInliers(inliers->indices);
    if(T(0,3)*T(0,3) + T(1,3)*T(1,3) + T(2,3)*T(2,3) > 0.6*0.6)
      inliers->indices.clear();
    return inliers;
}

void extractFromIndices(const PCRGB& in, pcl::PointIndices::Ptr& inds, PCRGB& out)
{
  pcl::ExtractIndices<PRGB> extract;
  extract.setInputCloud(in.makeShared());
  extract.setIndices(inds);
  extract.setNegative(false);
  extract.filter(out);
}

/*
void bagProblem(char* filename, KAMProblem& fam)
{
  rosbag::Bag bag_out(filename, rosbag::bagmode::Write);
  pcl::toROSMsg<PRGB>(pc, pc_msg);
  bag_out.write("pc", ros::Time::now(), pc_msg);
  bag_out.close();
}
*/

void visualizeSolution(vector<PCRGB::Ptr> &pcs, 
                       vector<gtsam::Pose3> &poses,
                       gtsam::Pose3& solution)
{
  PCRGB total(*pcs[0]);
  total.clear();
  for(size_t i=0;i<poses.size();i++) 
    transPoints(*pcs[i],(poses[i].matrix() * solution.inverse().matrix()).cast<float>(),total);
  writeBag("output_pc.bag", total);
  total.header.frame_id = "/camera_rgb_optical_frame";
    pcl::visualization::CloudViewer viewer("Registrations");
    viewer.showCloud(total.makeShared());
    while(!viewer.wasStopped()) {}
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "fam_mapping");
  ros::NodeHandle nh_priv("~");
  double dist_thresh;
  nh_priv.param<double>("dist_thresh", dist_thresh, 0.06);
  int num_poses;
  nh_priv.param<int>("num_poses", num_poses, -1);

  KAMProblem prob;
  vector<Mat> imgs;
  vector<PCRGB::Ptr> pcs;
  readImgPCPoseBag(argv[1], imgs, pcs, prob.base_Ts_ee, num_poses);
  printf("Num: %d, %d\n", imgs.size(), num_poses);

  prob.points_cameras.resize(imgs.size());
  for(size_t i=0;i<imgs.size();i++) {
    prob.base_Ts_ee[i].print("Pose");
    for(size_t j=i+1;j<imgs.size();j++) {
      printf("comp: i %d, j %d\n", i, j);
      PCRGB kps_1, kps_2, kps_1_reduced, kps_2_reduced;
      //cv::imshow("Image Display", imgs[i]);
      //cv::waitKey(5000);
      getCleanMatches(imgs[i], *pcs[i], imgs[j], *pcs[j], kps_1, kps_2);
      //visClouds(kps_1, kps_2);
      pcl::PointIndices::Ptr inliers = RANSACInliers(kps_1, kps_2, dist_thresh);
      printf("%d to %d\n", kps_1.size(), inliers->indices.size());
      extractFromIndices(kps_1, inliers, kps_1_reduced); 
      extractFromIndices(kps_2, inliers, kps_2_reduced); 
      for(size_t k=0;k<kps_1_reduced.size();k++) {
        PRGB pt1, pt2;
        pt1 = kps_1_reduced.points[k];
        pt2 = kps_2_reduced.points[k];
        gtsam::Point3 gtspt1(pt1.x, pt1.y, pt1.z);
        gtsam::Point3 gtspt2(pt2.x, pt2.y, pt2.z);
        if(k%10 == 0) {
          gtspt1.print("\n\npt1: ");
          gtspt2.print("pt2: ");
        }
        prob.points_cameras[i].push_back(gtspt1);
        prob.points_cameras[j].push_back(gtspt2);
        prob.correspondences.push_back(
            Correspondence(i, j, prob.points_cameras[i].size()-1, prob.points_cameras[j].size()-1));
      }
    }
  }

	std::cout << "Computing Matches...";
	gtsam::Pose3 solution = solveKAMProblem(prob);
	
	std::cout << "Computed Matches\n";
	std::cout << "Solution: \n";
	solution.print();

  visualizeSolution(pcs, prob.base_Ts_ee, solution);
  
  return 0;
}
