#ifndef CALIB_3D_CBS_H
#define CALIB_3D_CBS_H

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

namespace arm_3d_cb_calib {

////////////////////////////////////////////////////////////////////////////
// Kinect is mounted on the arm's end effector and its pose is unknown
struct CBCalibProblem
{
  vector<Pose3> base_T_ee_poses;
  vector<Point3> cb_p_points;
  vector<vector<vector<Point3> > > kinect_p_points;
  CBCalibProblem(int num_kinects, double cb_width, int num_cb_width, int num_cb_height) :
    kinect_p_points(num_kinects, vector<vector<Point3> >()) {
    for(int j=0;j<num_cb_height;j++)
      for(int i=0;i<num_cb_width;i++)
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

geometry_msgs::Pose gtsamPose3ToGeomPose(const gtsam::Pose3& pose_in);
gtsam::Pose3 geomPoseToGtsamPose3(const geometry_msgs::Pose& pose_in);

}
#endif
