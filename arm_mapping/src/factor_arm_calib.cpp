#include "arm_mapping/factor_arm_calib.h"

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

void doIt(int num_kinects, int num_ees)
{
  vector<Pose3> kinect_T_base_poses;
  vector<Pose3> offset_pose_;
  vector<Pose3> base_T_ee_poses;
  Pose3 cb_T_ee_pose;
  vector<Point3> cb_p_points;
  vector<vector<vector<Point3> > > kinect_p_points(num_kinects, vector<vector<Point3> >(num_ees));

  randomPoses(num_kinects, kinect_T_base_poses, 10);
  randomPoses(1, offset_pose_, 0.4);
  cb_T_ee_pose = offset_pose_[0];
  randomPoses(num_ees, base_T_ee_poses, 1.5);

  double cb_width = 0.03;
  int num_cb_width = 5, num_cb_height = 6;
  for(int i=0;i<num_cb_width;i++)
    for(int j=0;j<num_cb_height;j++)
      cb_p_points.push_back(Point3(i*cb_width,j*cb_width,0));

  for(int j=0;j<num_kinects;j++)
    for(int i=0;i<num_ees;i++)
      for(int l=0;l<num_cb_width*num_cb_height;l++)
        kinect_p_points[j][i].push_back(
            kinect_T_base_poses[j] * base_T_ee_poses[i] * cb_T_ee_pose.inverse() * cb_p_points[l]);

  
  ////////////////////////////////////////////////////////////////////////


  // noise terms
  noiseModel::Isotropic::shared_ptr ee_fact_noise = noiseModel::Isotropic::Sigma(3, 0.02);

  gtsam::NonlinearFactorGraph graph;
  Values init_estimate;
  init_estimate.insert(Symbol('o',0), Pose3());

  for(int j=0;j<num_kinects;j++) {
    init_estimate.insert(Symbol('k',j), Pose3());
    for(int i=0;i<num_ees;i++) {
      for(int l=0;l<num_cb_width*num_cb_height;l++) {
        graph.add(boost::make_shared<CheckerboardArmFactor>(
              ee_fact_noise, Symbol('o',0), Symbol('k',j), 
              cb_p_points[l], kinect_p_points[j][i][l], base_T_ee_poses[i]));
      }
    }
  }
  printf("start error: %f\n", graph.error(init_estimate));
  Values result = DoglegOptimizer(graph, init_estimate).optimize();
  result.print();
  printf("end error: %f\n", graph.error(result));
}

int main(int argc, char* argv[])
{
  doIt(3, 10);
}
