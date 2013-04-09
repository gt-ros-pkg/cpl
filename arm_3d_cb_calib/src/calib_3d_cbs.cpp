#include "ros/param.h"
#include "arm_3d_cb_calib/calib_3d_cbs.h"

using namespace std;
using namespace gtsam;

namespace arm_3d_cb_calib {

void randomPoses(int num_poses, vector<Pose3>& poses, double radius=1.0)
{
  boost::mt19937 rng;
  boost::uniform_real<> pt_range(-radius, radius);
  boost::uniform_real<> angle_range(0, M_PI);
  for(int i=0;i<num_poses;i++) {
    poses.push_back(Pose3(Rot3::ypr(angle_range(rng),angle_range(rng),angle_range(rng)),
                          Point3(pt_range(rng),pt_range(rng),pt_range(rng))));
  }
}

/////////////// Multi Kinect Arm Checkerboard Calib Problem //////////////////

void generateCBCalibProblem(CBCalibProblem& prob, CBCalibSolution& sol, int num_ees)
{
  size_t num_kinects = prob.kinect_p_points.size();
  size_t num_cb_points = prob.cb_p_points.size();
  randomPoses(num_kinects, sol.kinect_T_base_poses, 10);
  vector<Pose3> offset_pose_;
  randomPoses(1, offset_pose_, 0.4);
  sol.cb_T_ee_pose = offset_pose_[0];
  randomPoses(num_ees, prob.base_T_ee_poses, 1.5);

  for(size_t j=0;j<num_kinects;j++)
    for(size_t i=0;i<num_ees;i++) {
      prob.kinect_p_points[j].push_back(vector<Point3>());
      cout << (sol.kinect_T_base_poses[j] * prob.base_T_ee_poses[i] * 
            sol.cb_T_ee_pose.inverse()).inverse().matrix() << endl;
      for(size_t l=0;l<num_cb_points;l++)
        prob.kinect_p_points[j][i].push_back(
            sol.kinect_T_base_poses[j] * prob.base_T_ee_poses[i] * 
            sol.cb_T_ee_pose.inverse() * prob.cb_p_points[l]);
    }
}

class CheckerboardArmFactor : public NoiseModelFactor2<Pose3, Pose3>
{
  public:
    Point3 cb_p_point, kinect_p_point;
    Pose3 base_T_ee;
    CheckerboardArmFactor(const gtsam::SharedNoiseModel& noiseModel, 
                          Key j1, Key j2, Point3 _cb_p_point, Point3 _kinect_p_point, 
                          Pose3 _base_T_ee) :
      NoiseModelFactor2<Pose3, Pose3>(noiseModel, j1, j2), 
        cb_p_point(_cb_p_point), kinect_p_point(_kinect_p_point), base_T_ee(_base_T_ee) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& cb_T_ee,
                                const gtsam::Pose3& kinect_T_base,
                                boost::optional<Matrix&> H_1 = boost::none,
                                boost::optional<Matrix&> H_2 = boost::none) const
    {
      if(H_1) {
        (*H_1) = gtsam::Matrix_(3, 6);
        Point3 pt = cb_T_ee.inverse() * cb_p_point;
        Matrix cross = Matrix_(3,3,
                               0.0,-pt.z(),pt.y(),
                               pt.z(),0.0,-pt.x(),
                               -pt.y(),pt.x(),0.0);
        (*H_1).block<3,3>(0,0) = base_T_ee.rotation().matrix() * cross;
        (*H_1).block<3,3>(0,3) = -base_T_ee.rotation().matrix();
      }
      if(H_2) {
        (*H_2) = gtsam::Matrix_(3, 6);
        Point3 pt = kinect_T_base.inverse() * kinect_p_point;
        Matrix cross = Matrix_(3,3,
                               0.0,-pt.z(),pt.y(),
                               pt.z(),0.0,-pt.x(),
                               -pt.y(),pt.x(),0.0);
        (*H_2).block<3,3>(0,0) = -cross;
        (*H_2).block<3,3>(0,3) = gtsam::Rot3::yaw(0).matrix();
      }
      return (base_T_ee * cb_T_ee.inverse() * cb_p_point - 
              kinect_T_base.inverse() * kinect_p_point).vector();
    }
};

class PointTransFactor : public NoiseModelFactor1<Pose3>
{
  public:
    Point3 cb_p_point, kinect_p_point;
    PointTransFactor(const gtsam::SharedNoiseModel& noiseModel, 
                          Key j1, Point3 _cb_p_point, Point3 _kinect_p_point) :
      NoiseModelFactor1<Pose3>(noiseModel, j1), 
        cb_p_point(_cb_p_point), kinect_p_point(_kinect_p_point) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& cb_T_kinect,
                                boost::optional<Matrix&> H_1 = boost::none) const
    {
      if(H_1) {
        (*H_1) = gtsam::Matrix_(3, 6);
        Point3 pt = cb_T_kinect.inverse() * cb_p_point;
        Matrix cross = Matrix_(3,3,
                               0.0,-pt.z(),pt.y(),
                               pt.z(),0.0,-pt.x(),
                               -pt.y(),pt.x(),0.0);
        (*H_1).block<3,3>(0,0) = cross;
        (*H_1).block<3,3>(0,3) = -eye(3);
      }
      return (cb_T_kinect.inverse() * cb_p_point - kinect_p_point).vector();
    }
};

class MultiPoseFactor : public NoiseModelFactor3<Pose3, Pose3, Pose3>
{
  public:
    Pose3 base_T_ee;
    MultiPoseFactor(const gtsam::SharedNoiseModel& noiseModel, 
                          Key j1, Key j2, Key j3, Pose3 _base_T_ee) :
      NoiseModelFactor3<Pose3, Pose3, Pose3>(noiseModel, j1, j2, j3), 
        base_T_ee(_base_T_ee) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& ee_T_cb,
                                const gtsam::Pose3& kinect_T_base,
                                const gtsam::Pose3& cb_T_kinect,
                                boost::optional<Matrix&> H_1 = boost::none,
                                boost::optional<Matrix&> H_2 = boost::none,
                                boost::optional<Matrix&> H_3 = boost::none) const
    {
      if(H_1) {
        (*H_1) = (cb_T_kinect * kinect_T_base * base_T_ee * ee_T_cb).adjointMap();
      }
      if(H_2) {
        (*H_2) = (cb_T_kinect * kinect_T_base).adjointMap();
      }
      if(H_3) {
        (*H_3) = cb_T_kinect.adjointMap();
      }
      return Pose3::Logmap(cb_T_kinect * kinect_T_base * base_T_ee * ee_T_cb);
    }
};

void solveCBCalibProblem(const CBCalibProblem& prob, CBCalibSolution& sol)
{
  size_t num_kinects = prob.kinect_p_points.size();
  size_t num_cb_points = prob.cb_p_points.size();
  // noise terms
  double pt_noise, pose_noise;
  ros::param::param<double>("~pt_noise", pt_noise, 0.02);
  ros::param::param<double>("~pose_noise", pose_noise, 0.02);
  noiseModel::Isotropic::shared_ptr pt_fact_noise = noiseModel::Isotropic::Sigma(3, pt_noise);
  noiseModel::Isotropic::shared_ptr mp_fact_noise = noiseModel::Isotropic::Sigma(6, pose_noise);
  //noiseModel::MEstimator::Base::shared_ptr robust_model = noiseModel::MEstimator::Fair::Create(3.0);
  //noiseModel::Robust::shared_ptr robust_ee_fact_noise = noiseModel::Robust::Create(robust_model, ee_fact_noise);

  gtsam::NonlinearFactorGraph graph;
  Values init_estimate;

  bool init;
  double init_x, init_y, init_z, init_qx, init_qy, init_qz, init_qw;
  ros::param::param<bool>("~init", init, 0);
  ros::param::param<double>("~init_x", init_x, 0.0);
  ros::param::param<double>("~init_y", init_y, 0.0);
  ros::param::param<double>("~init_z", init_z, 0.0);
  ros::param::param<double>("~init_qx", init_qx, 0.0);
  ros::param::param<double>("~init_qy", init_qy, 0.0);
  ros::param::param<double>("~init_qz", init_qz, 0.0);
  ros::param::param<double>("~init_qw", init_qw, 1.0);
  int cb_ind = 0;
  Pose3 init_offset = Pose3(Quaternion(init_qw,init_qx,init_qy,init_qz),
                                      Point3(init_x,init_y,init_z));
  init_estimate.insert(Symbol('o',0), init_offset.inverse());
  if(init) {
    graph.add(boost::make_shared<PriorFactor<Pose3> >(
          Symbol('o',0), init_offset, mp_fact_noise));
    printf("*****************************************************\n%f %f %f %f %f %f %f\n", init_x, init_y, init_z, init_qx, init_qy, init_qz, init_qw);
  }
  for(size_t j=0;j<num_kinects;j++) {
    init_estimate.insert(Symbol('k',j), Pose3());
    for(size_t i=0;i<prob.kinect_p_points[j].size();i++) {
      init_estimate.insert(Symbol('c',cb_ind), Pose3());

      graph.add(boost::make_shared<MultiPoseFactor>(
            mp_fact_noise, 
            Symbol('o',0), Symbol('k',j), Symbol('c',cb_ind), 
            prob.base_T_ee_poses[i]));

      for(size_t l=0;l<num_cb_points;l++) {
        Point3 kinect_p = prob.kinect_p_points[j][i][l];
        if(kinect_p.x() != kinect_p.x())
          continue;
        graph.add(boost::make_shared<PointTransFactor>(
              pt_fact_noise, 
              Symbol('c',cb_ind), 
              prob.cb_p_points[l], kinect_p));
      }
      cb_ind++;
    }
  }
  Values result = DoglegOptimizer(graph, init_estimate).optimize();
  result.print();
  printf("start error: %f\n", graph.error(init_estimate));
  printf("end error: %f\n", graph.error(result));
  sol.cb_T_ee_pose = result.at<Pose3>(Symbol('o',0)).inverse();
  for(size_t j=0;j<num_kinects;j++) 
    sol.kinect_T_base_poses.push_back(result.at<Pose3>(Symbol('k',j)));
}

//////////////////////////////////////////////////////////////////////////////

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
