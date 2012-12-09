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

struct FAMProblem
{
  vector<Pose3> base_Ts_ee;
  vector<vector<Point3> > points_cameras;
  vector<Correspondence> correspondences;
};

struct FAMSolution
{
  vector<Pose3> base_Ts_cam;
  vector<Point3> points_base;
  Pose3 ee_T_cam;
  Values solution;
};

void generateRandomProblem(int num_cameras, int num_points, int cameras_per_point, 
                           FAMProblem& prob, FAMSolution& sol);
Pose3 solveProblemOffsetPose(const FAMProblem& prob);

geometry_msgs::Pose gtsamPose3ToGeomPose(const gtsam::Pose3& pose_in);
gtsam::Pose3 geomPoseToGtsamPose3(const geometry_msgs::Pose& pose_in);

}

#endif // FACTOR_ARM_MAPPING_H
