#include "arm_mapping/factor_arm_mapping.h"

using namespace arm_mapping;

int main(int argc, char* argv[])
{
  int num_kinects = 3;
  int num_cbs = 12;
  KinectCBCalibProblem prob(num_kinects, num_cbs, 0.03, 5, 6);
  KinectCBCalibSolution ground_sol, comp_sol;
  generateKinectCBCalibProblem(prob, ground_sol);
  solveKinectCBCalibProblem(prob, comp_sol);

  for(int i=0;i<num_kinects;i++)
    (comp_sol.kinect_T_world_poses[i].inverse() * ground_sol.kinect_T_world_poses[i]).print();
  for(int i=0;i<num_cbs;i++)
    (comp_sol.cb_T_world_poses[i].inverse() * ground_sol.cb_T_world_poses[i]).print();

  return 0;
}
