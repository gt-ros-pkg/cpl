#include "arm_mapping/factor_arm_mapping.h"

using namespace arm_mapping;

int main(int argc, char* argv[])
{
  int num_cameras = 8;
  int num_points = 200;
  int cameras_per_point = 6;
  FAMProblem prob;
  FAMSolution sol;
  generateRandomProblem(num_cameras, num_points, cameras_per_point, prob, sol);
  sol.ee_T_cam.print("Offset pose:\n");
  (solveProblemOffsetPose(prob) * sol.ee_T_cam).print("Solution error:\n");
  return 0;
}
