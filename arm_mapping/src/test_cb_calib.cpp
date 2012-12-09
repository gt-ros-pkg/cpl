#include "arm_mapping/factor_arm_mapping.h"

using namespace arm_mapping;

int main(int argc, char* argv[])
{
  int num_kinects = 3;
  int num_ees = 12;
  CBCalibProblem prob(num_kinects, num_ees, 0.03, 5, 6);
  CBCalibSolution ground_sol, comp_sol;
  generateCBCalibProblem(prob, ground_sol);
  solveCBCalibProblem(prob, comp_sol);
  for(int i=0;i<num_kinects;i++)
    (comp_sol.kinect_T_base_poses[i].inverse() * ground_sol.kinect_T_base_poses[i]).print();
  (comp_sol.cb_T_ee_pose.inverse() * ground_sol.cb_T_ee_pose).print();

  return 0;
}
