#include "arm_mapping/factor_arm_mapping.h"

using namespace std;
using namespace gtsam;

namespace arm_mapping {

class OffsetFrameFactor : public NoiseModelFactor1<Pose3>
{
  public:
    Point3 p1_local, p2_local;
    Pose3 middle2_T_middle1;
    OffsetFrameFactor(const gtsam::SharedNoiseModel& noiseModel, 
                     Key j1, Point3 _p1_local, Point3 _p2_local, Pose3 _middle2_T_middle1) :
      NoiseModelFactor1<Pose3>(noiseModel, j1), 
        p1_local(_p1_local), p2_local(_p2_local), middle2_T_middle1(_middle2_T_middle1) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& local_T_middle,
                                boost::optional<Matrix&> H = boost::none) const
    {
      if(H) {
        (*H) = gtsam::Matrix_(3, 6);
        Point3 pt1 = local_T_middle.inverse() * p1_local;
        Point3 pt2 = local_T_middle.inverse() * p2_local;
        Matrix cross1 = Matrix_(3,3,
                               0.0,-pt1.z(),pt1.y(),
                               pt1.z(),0.0,-pt1.x(),
                               -pt1.y(),pt1.x(),0.0);
        Matrix cross2 = Matrix_(3,3,
                               0.0,-pt2.z(),pt2.y(),
                               pt2.z(),0.0,-pt2.x(),
                               -pt2.y(),pt2.x(),0.0);
        (*H).block<3,3>(0,0) = (middle2_T_middle1.rotation().matrix() * cross1 - cross2);
        (*H).block<3,3>(0,3) = (gtsam::Rot3::yaw(0).matrix() - middle2_T_middle1.rotation().matrix());
        /*
        cout << "H " << (*H) << endl;
        cout << "pt1 " << pt1.vector() << endl;
        cout << "pt2 " << pt2.vector() << endl;
        cout << "cross1 " << cross1 << endl;
        cout << "cross2 " << cross2 << endl;
        printf("pt1 xyz: %f %f %f\n", -pt1.x(), -pt1.y(), -pt1.z());
        printf("pt2 xyz: %f %f %f\n", -pt2.x(), -pt2.y(), -pt2.z());
        */
      }
      return (middle2_T_middle1 * local_T_middle.inverse() * p1_local -
              local_T_middle.inverse() * p2_local).vector(); 
    }
};

void randomPose(int num_poses, vector<Pose3>& poses)
{
  boost::mt19937 rng;
  boost::uniform_real<> pt_range(-1, 1);
  boost::uniform_real<> angle_range(0, M_PI);
  for(int i=0;i<num_poses;i++) {
    poses.push_back(Pose3(Rot3::ypr(angle_range(rng),angle_range(rng),angle_range(rng)),
                          Point3(pt_range(rng),pt_range(rng),pt_range(rng))));
  }
}

void generateRandomProblem(int num_cameras, int num_points, int cameras_per_point, 
                           FAMProblem& prob, FAMSolution& sol)
{
  boost::mt19937 rng;
  boost::uniform_real<> pt_dist(-0.7, 0.7);
  boost::uniform_int<> uni_dist;
  boost::variate_generator<boost::mt19937, boost::uniform_int<> > rand_int(rng, uni_dist);
  boost::uniform_real<> ee_dist(-0.1, 0.1);
  boost::uniform_real<> angle_range(0, M_PI);

  sol.ee_T_cam = Pose3(gtsam::Rot3::ypr(angle_range(rng), angle_range(rng), angle_range(rng)), 
                   gtsam::Point3(ee_dist(rng), ee_dist(rng), ee_dist(rng)));
  randomPose(num_cameras, sol.base_Ts_cam);
  prob.points_cameras.resize(num_cameras);
  for(int i=0;i<num_cameras;i++)
    prob.base_Ts_ee.push_back(sol.base_Ts_cam[i] * sol.ee_T_cam.inverse());
  for(int i=0;i<num_points;i++) {
    Point3 pt_base = Point3(pt_dist(rng), pt_dist(rng), pt_dist(rng));
    sol.points_base.push_back(pt_base);
    vector<int> inds;
    for(int j=0;j<num_cameras;j++)
      inds.push_back(j);
    random_shuffle(inds.begin(), inds.end(), rand_int);
    // pick cameras_per_point base_Ts_cam to link this point to
    for(int j=0;j<cameras_per_point;j++) 
      prob.points_cameras[inds[j]].push_back(sol.base_Ts_cam[inds[j]].inverse() * pt_base);
    for(int j=0;j<cameras_per_point;j++) 
      for(int k=j+1;k<cameras_per_point;k++)
        prob.correspondences.push_back(Correspondence(inds[j],inds[k],
              prob.points_cameras[inds[j]].size()-1,prob.points_cameras[inds[k]].size()-1));
  }

  // GENERATE SOLUTION
  sol.solution.insert(Symbol('k',0), sol.ee_T_cam.inverse());
}

Pose3 solveProblemOffsetPose(const FAMProblem& prob)
{
  // noise terms
  noiseModel::Isotropic::shared_ptr ee_fact_noise = noiseModel::Isotropic::Sigma(3, 0.02);

  gtsam::NonlinearFactorGraph graph;
  Values init_estimate;
  init_estimate.insert(Symbol('k',0), Pose3());

  for(size_t i=0;i<prob.correspondences.size();i++) {
    gtsam::Point3 pt1_cam = prob.points_cameras[prob.correspondences[i].cam1]
                                               [prob.correspondences[i].ind1];
    gtsam::Point3 pt2_cam = prob.points_cameras[prob.correspondences[i].cam2]
                                               [prob.correspondences[i].ind2];
    gtsam::Pose3 ee2_T_ee1 = prob.base_Ts_ee[prob.correspondences[i].cam2].inverse() * 
                             prob.base_Ts_ee[prob.correspondences[i].cam1];
    graph.add(boost::make_shared<OffsetFrameFactor>(ee_fact_noise, 
                                                    Symbol('k',0), pt1_cam, pt2_cam, ee2_T_ee1));
  }
  Values result = DoglegOptimizer(graph, init_estimate).optimize();
  return result.at<Pose3>(Symbol('k',0));
}

geometry_msgs::Pose gtsamPose3ToGeomPose(const gtsam::Pose3& pose_in)
{
  geometry_msgs::Pose pose_out;
  pose_out.position.x = pose_in.x();
  pose_out.position.y = pose_in.y();
  pose_out.position.z = pose_in.z();
  Quaternion q = pose_in.rotation().toQuaternion();
  pose_out.orientation.w = q.w();
  pose_out.orientation.x = q.x();
  pose_out.orientation.y = q.y();
  pose_out.orientation.z = q.z();
  return pose_out;
}

gtsam::Pose3 geomPoseToGtsamPose3(const geometry_msgs::Pose& pose_in)
{
  return gtsam::Pose3(Rot3(Quaternion(pose_in.orientation.w, pose_in.orientation.x, 
                                      pose_in.orientation.y, pose_in.orientation.z)), 
                      Point3(pose_in.position.x, pose_in.position.y, pose_in.position.z));
}

}
