#include "arm_mapping/factor_arm_mapping.h"
#include "ros/param.h"

using namespace std;
using namespace gtsam;

namespace arm_mapping {

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

//////////////////////// Kinect Arm Mapping Problem //////////////////////////

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

void generateKAMProblem(int num_cameras, int num_points, int cameras_per_point, 
                           KAMProblem& prob, KAMSolution& sol)
{
  boost::mt19937 rng;
  boost::uniform_real<> pt_dist(-0.7, 0.7);
  boost::uniform_int<> uni_dist;
  boost::variate_generator<boost::mt19937, boost::uniform_int<> > rand_int(rng, uni_dist);
  boost::uniform_real<> ee_dist(-0.1, 0.1);
  boost::uniform_real<> angle_range(0, M_PI);

  sol.ee_T_cam = Pose3(gtsam::Rot3::ypr(angle_range(rng), angle_range(rng), angle_range(rng)), 
                   gtsam::Point3(ee_dist(rng), ee_dist(rng), ee_dist(rng)));
  randomPoses(num_cameras, sol.base_Ts_cam);
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

Pose3 solveKAMProblem(const KAMProblem& prob)
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

//////////////////////////////////////////////////////////////////////////////

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

void solveCBCalibProblem(const CBCalibProblem& prob, CBCalibSolution& sol)
{
  size_t num_kinects = prob.kinect_p_points.size();
  size_t num_cb_points = prob.cb_p_points.size();
  // noise terms
  noiseModel::Isotropic::shared_ptr ee_fact_noise = noiseModel::Isotropic::Sigma(3, 0.02);
  //noiseModel::MEstimator::Base::shared_ptr robust_model = noiseModel::MEstimator::Fair::Create(3.0);
  //noiseModel::Robust::shared_ptr robust_ee_fact_noise = noiseModel::Robust::Create(robust_model, ee_fact_noise);

  gtsam::NonlinearFactorGraph graph;
  Values init_estimate;
  init_estimate.insert(Symbol('o',0), Pose3());

  double init_x, init_y, init_z, init_yaw, init_pitch, init_roll;
  ros::param::param<double>("~init_x", init_x, 0.0);
  ros::param::param<double>("~init_y", init_y, 0.0);
  ros::param::param<double>("~init_z", init_z, 0.0);
  ros::param::param<double>("~init_yaw", init_yaw, 0.0);
  ros::param::param<double>("~init_pitch", init_pitch, 0.0);
  ros::param::param<double>("~init_roll", init_roll, 0.0);
  for(size_t j=0;j<num_kinects;j++) {
    init_estimate.insert(Symbol('k',j), Pose3(Rot3::ypr(init_yaw,init_pitch,init_roll),
                                              Point3(init_x,init_y,init_z)));
    for(size_t i=0;i<prob.kinect_p_points[j].size();i++) {
      for(size_t l=0;l<num_cb_points;l++) {
        Point3 kinect_p = prob.kinect_p_points[j][i][l];
        if(kinect_p.x() != kinect_p.x())
          continue;
        graph.add(boost::make_shared<CheckerboardArmFactor>(
              ee_fact_noise, Symbol('o',0), Symbol('k',j), 
              prob.cb_p_points[l], kinect_p, prob.base_T_ee_poses[i]));
      }
    }
  }
  printf("start error: %f\n", graph.error(init_estimate));
  Values result = DoglegOptimizer(graph, init_estimate).optimize();
  result.print();
  printf("end error: %f\n", graph.error(result));
  sol.cb_T_ee_pose = result.at<Pose3>(Symbol('o',0));
  for(size_t j=0;j<num_kinects;j++) 
    sol.kinect_T_base_poses.push_back(result.at<Pose3>(Symbol('k',j)));
}

//////////////////////////////////////////////////////////////////////////////

///////////////// Multi Kinect Checkerboard Calib Problem ////////////////////

void generateKinectCBCalibProblem(KinectCBCalibProblem& prob, KinectCBCalibSolution& sol)
{
  size_t num_cbs = prob.kinect_p_points.size();
  size_t num_kinects = prob.kinect_p_points[0].size();
  size_t num_cb_points = prob.cb_p_points.size();
  randomPoses(num_kinects, sol.kinect_T_world_poses, 10);
  randomPoses(num_cbs, sol.cb_T_world_poses, 10);
  Pose3 world_frame = sol.kinect_T_world_poses[0];
  for(size_t i=0;i<num_kinects;i++)
    sol.kinect_T_world_poses[i] = sol.kinect_T_world_poses[i] * world_frame.inverse();
  for(size_t i=0;i<num_cbs;i++)
    sol.cb_T_world_poses[i] = sol.cb_T_world_poses[i] * world_frame.inverse();

  for(size_t j=0;j<num_cbs;j++)
    for(size_t i=0;i<num_kinects;i++)
      for(size_t l=0;l<num_cb_points;l++)
        prob.kinect_p_points[j][i].push_back(
            sol.kinect_T_world_poses[i] *  
            sol.cb_T_world_poses[j].inverse() * prob.cb_p_points[l]);
}

class CheckerboardKinectFactor : public NoiseModelFactor2<Pose3, Pose3>
{
  public:
    Point3 cb_p_point, kinect_p_point;
    CheckerboardKinectFactor(const gtsam::SharedNoiseModel& noiseModel, 
                          Key j1, Key j2, Point3 _cb_p_point, Point3 _kinect_p_point) :
      NoiseModelFactor2<Pose3, Pose3>(noiseModel, j1, j2), 
        cb_p_point(_cb_p_point), kinect_p_point(_kinect_p_point) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& cb_T_world,
                                const gtsam::Pose3& kinect_T_world,
                                boost::optional<Matrix&> H_1 = boost::none,
                                boost::optional<Matrix&> H_2 = boost::none) const
    {
      if(H_1) {
        (*H_1) = gtsam::Matrix_(3, 6);
        Point3 pt = cb_T_world.inverse() * cb_p_point;
        Matrix cross = Matrix_(3,3,
                               0.0,-pt.z(),pt.y(),
                               pt.z(),0.0,-pt.x(),
                               -pt.y(),pt.x(),0.0);
        (*H_1).block<3,3>(0,0) = -cross;
        (*H_1).block<3,3>(0,3) = gtsam::Rot3::yaw(0).matrix();
      }
      if(H_2) {
        (*H_2) = gtsam::Matrix_(3, 6);
        Point3 pt = kinect_T_world.inverse() * kinect_p_point;
        Matrix cross = Matrix_(3,3,
                               0.0,-pt.z(),pt.y(),
                               pt.z(),0.0,-pt.x(),
                               -pt.y(),pt.x(),0.0);
        (*H_2).block<3,3>(0,0) = cross;
        (*H_2).block<3,3>(0,3) = -gtsam::Rot3::yaw(0).matrix();
      }
      return (kinect_T_world.inverse() * kinect_p_point -
              cb_T_world.inverse() * cb_p_point).vector();
    }
};

void solveKinectCBCalibProblem(const KinectCBCalibProblem& prob, KinectCBCalibSolution& sol)
{
  size_t num_cbs = prob.kinect_p_points.size();
  size_t num_kinects = prob.kinect_p_points[0].size();
  size_t num_cb_points = prob.cb_p_points.size();
  // noise terms
  noiseModel::Isotropic::shared_ptr cb_fact_noise = noiseModel::Isotropic::Sigma(3, 0.02);
  noiseModel::Isotropic::shared_ptr init_kin_fact_noise = noiseModel::Isotropic::Sigma(6, 0.02);

  gtsam::NonlinearFactorGraph graph;
  graph.add(boost::make_shared<PriorFactor<Pose3> >(
            Symbol('k',0), Pose3(), init_kin_fact_noise));
  Values init_estimate;
  for(size_t i=0;i<num_kinects;i++) {
    init_estimate.insert(Symbol('k',i), Pose3());
  }
  for(size_t j=0;j<num_cbs;j++) {
    init_estimate.insert(Symbol('c',j), Pose3());
  }

  for(size_t j=0;j<num_cbs;j++) {
    for(size_t i=0;i<num_kinects;i++) {
      if(prob.kinect_p_points[j][i].size() > 0) {
        for(size_t l=0;l<num_cb_points;l++) {
          Point3 kinect_p = prob.kinect_p_points[j][i][l];
          if(kinect_p.x() != kinect_p.x())
            continue;
          graph.add(boost::make_shared<CheckerboardKinectFactor>(
                cb_fact_noise, Symbol('c',j), Symbol('k',i), 
                prob.cb_p_points[l], kinect_p));
        }
      }
    }
  }
  printf("start error: %f\n", graph.error(init_estimate));
  Values result = DoglegOptimizer(graph, init_estimate).optimize();
  result.print();
  printf("end error: %f\n", graph.error(result));
  Pose3 world_frame = result.at<Pose3>(Symbol('k',0));
  for(size_t j=0;j<num_cbs;j++) 
    sol.cb_T_world_poses.push_back(result.at<Pose3>(Symbol('c',j))*world_frame.inverse());
  for(size_t i=0;i<num_kinects;i++) 
    sol.kinect_T_world_poses.push_back(result.at<Pose3>(Symbol('k',i))*world_frame.inverse());
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
