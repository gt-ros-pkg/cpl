#ifndef FACTOR_ARM_MAPPING_H
#define FACTOR_ARM_MAPPING_H

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/SimpleCamera.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/linear/linearExceptions.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/range/algorithm/random_shuffle.hpp>

#include <geometry_msgs/Pose.h>

#include <vector>
#include <math.h>

using namespace std;
using namespace gtsam;

namespace arm_mapping {

struct Correspondence { 
  int cam1, cam2, ind1, ind2; 
  Correspondence(int _cam1, int _cam2, int _ind1, int _ind2) :
    cam1(_cam1), cam2(_cam2), ind1(_ind1), ind2(_ind2) {}
};

////////////////////////////////////////////////////////////////////////////
// register point clouds taken from a kinect mounted on the arm from correspondances
struct KAMProblem
{
  vector<Pose3> base_Ts_ee;
  vector<vector<Point3> > points_cameras;
  vector<Correspondence> correspondences;
};

struct KAMSolution
{
  vector<Pose3> base_Ts_cam;
  vector<Point3> points_base;
  Pose3 ee_T_cam;
  Values solution;
};

void generateKAMProblem(int num_cameras, int num_points, int cameras_per_point, 
                           KAMProblem& prob, KAMSolution& sol);
Pose3 solveKAMProblem(const KAMProblem& prob);
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
// Kinect is mounted on the arm's end effector and its pose is unknown
struct CBCalibProblem
{
  vector<Pose3> base_T_ee_poses;
  vector<Point3> cb_p_points;
  vector<vector<vector<Point3> > > kinect_p_points;
  CBCalibProblem(int num_kinects, double cb_width, int num_cb_width, int num_cb_height) :
    kinect_p_points(num_kinects, vector<vector<Point3> >()) {
    for(int i=0;i<num_cb_width;i++)
      for(int j=0;j<num_cb_height;j++)
        cb_p_points.push_back(Point3(i*cb_width,j*cb_width,0));
  }
};

struct CBCalibSolution
{
  Pose3 cb_T_ee_pose;
  vector<Pose3> kinect_T_base_poses;
};

void generateCBCalibProblem(CBCalibProblem& prob, CBCalibSolution& sol, int num_ees);
void solveCBCalibProblem(const CBCalibProblem& prob, CBCalibSolution& sol);
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
// Kinects in freespace seeing mutual checkerboards
struct KinectCBCalibProblem
{
  vector<Point3> cb_p_points;
  vector<vector<vector<Point3> > > kinect_p_points;
  KinectCBCalibProblem(int num_kinects, int num_cbs, double cb_width, 
                       int num_cb_width, int num_cb_height) :
    kinect_p_points(num_cbs, vector<vector<Point3> >(num_kinects)) {
    for(int i=0;i<num_cb_width;i++)
      for(int j=0;j<num_cb_height;j++)
        cb_p_points.push_back(Point3(i*cb_width,j*cb_width,0));
  }
};

struct KinectCBCalibSolution
{
  vector<Pose3> kinect_T_world_poses;
  vector<Pose3> cb_T_world_poses;
};

void generateKinectCBCalibProblem(KinectCBCalibProblem& prob, KinectCBCalibSolution& sol);
void solveKinectCBCalibProblem(const KinectCBCalibProblem& prob, KinectCBCalibSolution& sol);
////////////////////////////////////////////////////////////////////////////

geometry_msgs::Pose gtsamPose3ToGeomPose(const gtsam::Pose3& pose_in);
gtsam::Pose3 geomPoseToGtsamPose3(const geometry_msgs::Pose& pose_in);

}

#endif // FACTOR_ARM_MAPPING_H
