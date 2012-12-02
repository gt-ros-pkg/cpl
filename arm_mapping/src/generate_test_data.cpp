
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/SimpleCamera.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
//#include <gtsam_unstable/slam/ReferenceFrameFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <vector>
#include <math.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>

using namespace std;
using namespace gtsam;

class ReferenceFrameFactor : public NoiseModelFactor3<Pose3, Point3, Point3>
{
  public:
    ReferenceFrameFactor(const gtsam::SharedNoiseModel& noiseModel, 
                         Key j1, Key j2, Key j3) :
      NoiseModelFactor3<Pose3,Point3,Point3>(noiseModel, j1, j2, j3) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& lTg,
                                const gtsam::Point3& pl,
                                const gtsam::Point3& pg,
                                boost::optional<Matrix&> H_1 = boost::none,
                                boost::optional<Matrix&> H_2 = boost::none,
                                boost::optional<Matrix&> H_3 = boost::none) const
    {
      if(H_1) {
        (*H_1) = gtsam::Matrix_(3, 6);
        gtsam::Point3 pt = lTg.inverse() * pl;
        (*H_1).block<3,3>(0,0) = gtsam::Matrix_(3,3,0,-pt.z(),pt.y(),pt.z(),0,-pt.x(),-pt.y(),pt.x(),0);
        (*H_1).block<3,3>(0,3) = -gtsam::Rot3::yaw(0).matrix();
      }
      if(H_2)
        (*H_2) = lTg.rotation().transpose().matrix();
      if(H_3)
        (*H_3) = gtsam::Rot3::yaw(0).matrix();
       return (lTg.inverse() * pl - pg).vector(); 
    }
};

class RelativePoseFactor : public NoiseModelFactor2<Point3, Point3>
{
  public:
    Pose3 lTg;
    RelativePoseFactor(const gtsam::SharedNoiseModel& noiseModel, 
                         Key j1, Key j2, Pose3 _lTg) :
      NoiseModelFactor2<Point3,Point3>(noiseModel, j1, j2), lTg(_lTg) {}

    gtsam::Vector evaluateError(const gtsam::Point3& pl,
                                const gtsam::Point3& pg,
                                boost::optional<Matrix&> H_1 = boost::none,
                                boost::optional<Matrix&> H_2 = boost::none) const
    {
      if(H_1)
        (*H_1) = lTg.rotation().transpose().matrix();
      if(H_2)
        (*H_2) = gtsam::Rot3::yaw(0).matrix();
       return (lTg.inverse() * pl - pg).vector(); 
    }
};

class RelativePointFactor : public NoiseModelFactor2<Pose3, Point3>
{
  public:
    Point3 pl;
    RelativePointFactor(const gtsam::SharedNoiseModel& noiseModel, 
                         Key j1, Key j2, Point3 _pl) :
      NoiseModelFactor2<Pose3,Point3>(noiseModel, j1, j2), pl(_pl) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& lTg,
                                const gtsam::Point3& pg,
                                boost::optional<Matrix&> H_1 = boost::none,
                                boost::optional<Matrix&> H_2 = boost::none) const
    {
      if(H_1) {
        (*H_1) = gtsam::Matrix_(3, 6);
        gtsam::Point3 pt = lTg.inverse() * pl;
        (*H_1).block<3,3>(0,0) = gtsam::Matrix_(3,3,0,-pt.z(),pt.y(),pt.z(),0,-pt.x(),-pt.y(),pt.x(),0);
        (*H_1).block<3,3>(0,3) = -gtsam::Rot3::yaw(0).matrix();
      }
      if(H_2)
        (*H_2) = gtsam::Rot3::yaw(0).matrix();
       return (lTg.inverse() * pl - pg).vector(); 
    }
};

void circlePose(int num_poses, double radius, gtsam::Values& values)
{
  gtsam::Rot3 gRo(0,1,0,1,0,0,0,0,-1);
  double theta = 0.0;
  for(int i=0;i<num_poses;i++) {
    values.insert(gtsam::Symbol('c',i), 
                  gtsam::Pose3(gRo.compose(gtsam::Rot3::ypr(-theta,0,theta/3)),
                               gtsam::Point3(radius*cos(theta), radius*sin(theta), 0)));
    theta += 2*M_PI/num_poses;
  }
}

void armMappingTestData(int num_cameras, int num_points, int cameras_per_point)
{
  gtsam::Values cameras;
  gtsam::Values ee_frames;
  vector<int> cam_frames;
  vector<int> pt_inds;
  gtsam::Pose3 offset_pose(gtsam::Rot3::ypr(0.3, 0.1, 0.2), gtsam::Point3(0.03,0.1,0.2));
  circlePose(num_cameras, 1, cameras);
  //cameras.print();
  for(int i=0;i<num_cameras;i++) {
    ee_frames.insert(gtsam::Symbol('e',i), 
                     cameras.at<gtsam::Pose3>(gtsam::Symbol('c',i)) * offset_pose.inverse());
  }
  double radius = 1;
  boost::mt19937 rng;
  boost::uniform_real<> pt_dist(-radius*0.7, radius*0.7);
  boost::uniform_int<> uni_dist;
  boost::variate_generator<boost::mt19937, boost::uniform_int<> > rand_int(rng, uni_dist);
  gtsam::Values points_global;
  gtsam::Values points_local;
  int plocal_counter = 0;
  for(int i=0;i<num_points;i++) {
    gtsam::Point3 pt = gtsam::Point3(pt_dist(rng), pt_dist(rng), pt_dist(rng));
    points_global.insert(gtsam::Symbol('p',i), pt);
    vector<int> inds;
    for(int j=0;j<num_cameras;j++) {
      inds.push_back(j);
    }
    random_shuffle(inds.begin(), inds.end(), rand_int);
    // pick cameras_per_point cameras to link this point to
    for(int j=0;j<cameras_per_point;j++) {
      points_local.insert(gtsam::Symbol('p',plocal_counter++), 
                          cameras.at<gtsam::Pose3>(gtsam::Symbol('c',inds[j])) * pt);
      cam_frames.push_back(inds[j]);
      pt_inds.push_back(i);
    }
  }




  // inference
  gtsam::NonlinearFactorGraph graph;
  noiseModel::Isotropic::shared_ptr pt_noise = noiseModel::Isotropic::Sigma(3, 0.02);
  noiseModel::Diagonal::shared_ptr cam_noise = noiseModel::Diagonal::Sigmas(Vector_(6, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03));
  //noiseModel::Diagonal::shared_ptr cam_noise = noiseModel::Diagonal::Sigmas(Vector_(6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  noiseModel::Isotropic::shared_ptr ee_fact_noise = noiseModel::Isotropic::Sigma(3, 0.2);
  noiseModel::Isotropic::shared_ptr cam_fact_noise = noiseModel::Isotropic::Sigma(3, 0.2);
  Values initialEstimate;
  initialEstimate.insert(Symbol('k',0), Pose3());
  //initialEstimate.insert(Symbol('k',0), offset_pose);
  for(size_t i=0;i<cam_frames.size();i++) {
    gtsam::Point3 pt_cam = points_local.at<gtsam::Point3>(gtsam::Symbol('p',i));
    gtsam::Pose3 pose_cam = ee_frames.at<gtsam::Pose3>(gtsam::Symbol('e',cam_frames[i]));
    
    // add measurements
    //graph.add(gtsam::PriorFactor<gtsam::Point3>(gtsam::Symbol('c',i),pt_cam,pt_noise));
    //graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('e',cam_frames[i]),pose_cam,cam_noise));

    // add factors
    graph.add(boost::make_shared<RelativePoseFactor>(ee_fact_noise, 
                                                     Symbol('a',i), 
                                                     Symbol('b',pt_inds[i]),
                                                     pose_cam));
    graph.add(boost::make_shared<RelativePointFactor>(cam_fact_noise, 
                                                      Symbol('k',0), 
                                                      Symbol('a',i),
                                                      pt_cam));

    initialEstimate.insert(Symbol('a',i),pt_cam);
#if 0
    if(!initialEstimate.exists(Symbol('b',pt_inds[i])))
      initialEstimate.insert(Symbol('b',pt_inds[i]),pose_cam.inverse()*pt_cam);
#endif
    //initialEstimate.insert(Symbol('c',i),pt_cam);
  }
#if 0
  for(size_t i=0;i<num_cameras;i++) {
    gtsam::Pose3 pose_cam = ee_frames.at<gtsam::Pose3>(gtsam::Symbol('e',i));
    initialEstimate.insert(Symbol('e',i),pose_cam);
  }
#endif
#if 1
  for(size_t i=0;i<num_points;i++) {
    initialEstimate.insert(Symbol('b',i), points_global.at<Point3>(Symbol('p',i)));
  }
  //for(size_t i=0;i<;i++) {
#endif
  graph.print();
  printf("error %f\n", graph.error(initialEstimate));

  Values result = DoglegOptimizer(graph, initialEstimate).optimize();
  //Values result = LevenbergMarquardtOptimizer(graph, initialEstimate).optimize();
  result.print("done:\n");
  printf("opt error: %f\n", graph.error(initialEstimate));
}

int main(int argc, char* argv[])
{
  int num_cameras = 20;
  int num_points = 400;
  int cameras_per_point = 5;
  armMappingTestData(num_cameras, num_points, cameras_per_point);
}
