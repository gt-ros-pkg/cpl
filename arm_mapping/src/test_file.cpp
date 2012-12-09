
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
#include <gtsam/linear/linearExceptions.h>
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
        (*H_2) = - gtsam::Rot3::yaw(0).matrix();
       return (lTg.inverse() * pl - pg).vector(); 
    }
};

class RelativePointFactor : public NoiseModelFactor2<Pose3, Point3>
{
  public:
    Point3 pg;
    RelativePointFactor(const gtsam::SharedNoiseModel& noiseModel, 
                         Key j1, Key j2, Point3 _pg) :
      NoiseModelFactor2<Pose3,Point3>(noiseModel, j1, j2), pg(_pg) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& lTg,
                                const gtsam::Point3& pl,
                                boost::optional<Matrix&> H_1 = boost::none,
                                boost::optional<Matrix&> H_2 = boost::none) const
    {
      if(H_1) {
        (*H_1) = gtsam::Matrix_(3, 6);
        gtsam::Point3 pt = lTg.inverse() * pl;
        (*H_1).block<3,3>(0,0) = gtsam::Matrix_(3,3,0,-pt.z(),pt.y(),pt.z(),0,-pt.x(),-pt.y(),pt.x(),0);
        (*H_1).block<3,3>(0,3) = -gtsam::Rot3::yaw(0).matrix();
        //cout << "H_1 " << (*H_1) << endl;
      }
      if(H_2) {
        (*H_2) = lTg.rotation().transpose().matrix();
        //cout << "H_2 " << (*H_2) << endl;
      }
       return (lTg.inverse() * pl - pg).vector(); 
    }
};

class MiddlePoseFactor : public NoiseModelFactor2<Pose3, Point3>
{
  public:
    Pose3 middle_T_global;
    Point3 p_local;
    MiddlePoseFactor(const gtsam::SharedNoiseModel& noiseModel, 
                      Key j1, Key j2, Pose3 _middle_T_global, Point3 _p_local) :
      NoiseModelFactor2<Pose3,Point3>(noiseModel, j1, j2), 
      middle_T_global(_middle_T_global), p_local(_p_local) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& middle_T_local,
                                const gtsam::Point3& p_global,
                                boost::optional<Matrix&> H_1 = boost::none,
                                boost::optional<Matrix&> H_2 = boost::none) const
    {
      if(H_1) {
        (*H_1) = gtsam::Matrix_(3, 6);
        gtsam::Point3 pt = middle_T_local.inverse() * middle_T_global * p_global;
        (*H_1).block<3,3>(0,0) = gtsam::Matrix_(3,3,
                                                0,-pt.z(),pt.y(),
                                                pt.z(),0,-pt.x(),
                                                -pt.y(),pt.x(),0);
        (*H_1).block<3,3>(0,3) = -gtsam::Rot3::yaw(0).matrix();
        cout << "H_1 " << H_1 << endl;
      }
      if(H_2) {
        (*H_2) = (middle_T_local.rotation().transpose().matrix() *
                  middle_T_global.rotation().matrix());
        cout << "H_2 " << H_2 << endl;
      }
       return (middle_T_local.inverse() * middle_T_global * p_global - p_local).vector(); 
    }
};

class DoublePoseFactor : public NoiseModelFactor3<Pose3, Pose3, Point3>
{
  public:
    Point3 p_local;
    DoublePoseFactor(const gtsam::SharedNoiseModel& noiseModel, 
                     Key j1, Key j2, Key j3, Point3 _p_local) :
      NoiseModelFactor3<Pose3,Pose3,Point3>(noiseModel, j1, j2, j3), p_local(_p_local) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& local_T_middle,
                                const gtsam::Pose3& global_T_middle,
                                const gtsam::Point3& p_global,
                                boost::optional<Matrix&> H_1 = boost::none,
                                boost::optional<Matrix&> H_2 = boost::none,
                                boost::optional<Matrix&> H_3 = boost::none) const
    {
      if(H_1) {
        (*H_1) = gtsam::Matrix_(3, 6);
        gtsam::Point3 pt = local_T_middle.inverse() * p_local;
        (*H_1).block<3,3>(0,0) = -gtsam::Matrix_(3,3,
                                                0.0,-pt.z(),pt.y(),
                                                pt.z(),0.0,-pt.x(),
                                                -pt.y(),pt.x(),0.0);
        (*H_1).block<3,3>(0,3) = gtsam::Rot3::yaw(0).matrix();
        //cout << "H_1 " << H_1 << endl;
      }
      if(H_2) {
        (*H_2) = gtsam::Matrix_(3, 6);
        gtsam::Point3 pt = global_T_middle.inverse() * p_global;
        (*H_2).block<3,3>(0,0) = gtsam::Matrix_(3,3,
                                                0.0,-pt.z(),pt.y(),
                                                pt.z(),0.0,-pt.x(),
                                                -pt.y(),pt.x(),0.0);
        (*H_2).block<3,3>(0,3) = -gtsam::Rot3::yaw(0).matrix();
        //cout << "H_2 " << H_2 << endl;
      }
      if(H_3) {
        (*H_3) = global_T_middle.rotation().transpose().matrix();
        //cout << "H_3 " << H_3 << endl;
      }
       return (global_T_middle.inverse() * p_global - local_T_middle.inverse() * p_local).vector(); 
    }
};

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

void circlePose(int num_poses, double radius, vector<Pose3> poses)
{
  gtsam::Rot3 gRo(0,1,0,1,0,0,0,0,-1);
  double theta = 0.0;
  for(int i=0;i<num_poses;i++) {
    poses.push_back(gtsam::Pose3(gRo.compose(gtsam::Rot3::ypr(-theta,0,theta/3)),
                                 gtsam::Point3(radius*cos(theta), radius*sin(theta), 0)));
    theta += 2*M_PI/num_poses;
  }
}

void randomPose(int num_poses, vector<Pose3>& poses)
{
  boost::mt19937 rng;
  boost::uniform_real<> pt_range(-1, 1);
  boost::uniform_real<> angle_range(0, M_PI);
  for(int i=0;i<num_poses;i++) {
    poses.push_back(Pose3(Rot3::ypr(angle_range(rng),angle_range(rng),angle_range(rng)),
                          Point3(pt_range(rng),pt_range(rng),pt_range(rng))));
  }
}

struct Correspondence { 
  int cam1, cam2, ind1, ind2; 
  Correspondence(int _cam1, int _cam2, int _ind1, int _ind2) :
    cam1(_cam1), cam2(_cam2), ind1(_ind1), ind2(_ind2) {}
};

void armMappingTestData2(int num_cameras, int num_points, int cameras_per_point)
{
  vector<Pose3> base_Ts_cam;
  vector<Pose3> base_Ts_ee;
  vector<Point3> points_base;
  vector<Point3> points_camera;
  vector<vector<Point3> > points_cameras(num_cameras);
  vector<Correspondence> correspondences;
  vector<int> pt_camera_frame_inds;
  vector<int> pt_camera_base_ind;

  // GENERATE RANDOM DATA
  Pose3 ee_T_cam(gtsam::Rot3::ypr(0.3, 0.1, 0.2), gtsam::Point3(0.03,0.1,0.2));
  //circlePose(num_cameras, 1, base_Ts_cam);
  randomPose(num_cameras, base_Ts_cam);
  //base_Ts_cam.print();
  for(int i=0;i<num_cameras;i++) {
    base_Ts_ee.push_back(base_Ts_cam[i] * ee_T_cam.inverse());
  }
  double radius = 1;
  boost::mt19937 rng;
  boost::uniform_real<> pt_dist(-radius*0.7, radius*0.7);
  boost::uniform_int<> uni_dist;
  boost::variate_generator<boost::mt19937, boost::uniform_int<> > rand_int(rng, uni_dist);
  for(int i=0;i<num_points;i++) {
    gtsam::Point3 pt_base = gtsam::Point3(pt_dist(rng), pt_dist(rng), pt_dist(rng));
    points_base.push_back(pt_base);
    vector<int> inds;
    for(int j=0;j<num_cameras;j++)
      inds.push_back(j);
    random_shuffle(inds.begin(), inds.end(), rand_int);
    // pick cameras_per_point base_Ts_cam to link this point to
    for(int j=0;j<cameras_per_point;j++) 
      points_cameras[inds[j]].push_back(base_Ts_cam[inds[j]].inverse() * pt_base);
    for(int j=0;j<cameras_per_point;j++) 
      for(int k=j+1;k<cameras_per_point;k++)
        correspondences.push_back(Correspondence(inds[j],inds[k],
              points_cameras[inds[j]].size()-1,points_cameras[inds[k]].size()-1));
  }

  // GENERATE SOLUTION
  Values test_solution;
  test_solution.insert(Symbol('k',0), ee_T_cam.inverse());


  // INFERENCE

  // noise terms
  noiseModel::Isotropic::shared_ptr ee_fact_noise = noiseModel::Isotropic::Sigma(3, 0.002);

  gtsam::NonlinearFactorGraph graph;
  Values init_estimate;
  init_estimate.insert(Symbol('k',0), Pose3(gtsam::Rot3::ypr(0.3, 0.1, 0.2), gtsam::Point3(0.00,0.1,0.2)));
  //init_estimate.insert(Symbol('k',0), Pose3());

  for(size_t i=0;i<correspondences.size();i++) {
    gtsam::Point3 pt1_cam = points_cameras[correspondences[i].cam1][correspondences[i].ind1];
    gtsam::Point3 pt2_cam = points_cameras[correspondences[i].cam2][correspondences[i].ind2];
    gtsam::Pose3 ee2_T_ee1 = base_Ts_ee[correspondences[i].cam2].inverse() * 
                             base_Ts_ee[correspondences[i].cam1];
    graph.add(boost::make_shared<OffsetFrameFactor>(ee_fact_noise, 
                                                    Symbol('k',0), pt1_cam, pt2_cam, ee2_T_ee1));
  }
  graph.print();
  printf("opt error: %f\n", graph.error(test_solution));

  Values result = DoglegOptimizer(graph, init_estimate).optimize();
  result.print();
  result.at<Pose3>(Symbol('k',0)).print();
  ee_T_cam.inverse().print();
  (result.at<Pose3>(Symbol('k',0)) * ee_T_cam).print();

  gtsam::LevenbergMarquardtParams lm_params;
  //lm_params.setlambdaInitial(0.001);
  LevenbergMarquardtOptimizer lmo(graph, init_estimate, lm_params);
  for(int i=0;i<1000;i++) {
    lmo.iterate();
    printf("cur error: %f\n", graph.error(lmo.values()));
  }
}

void armMappingTestData(int num_cameras, int num_points, int cameras_per_point)
{
  vector<Pose3> base_Ts_cam;
  vector<Pose3> base_Ts_ee;
  vector<Point3> points_base;
  vector<Point3> points_camera;
  vector<int> pt_camera_frame_inds;
  vector<int> pt_camera_base_ind;

  // GENERATE RANDOM DATA
  Pose3 ee_T_cam(gtsam::Rot3::ypr(0.3, 0.1, 0.2), gtsam::Point3(0.03,0.1,0.2));
  //circlePose(num_cameras, 1, base_Ts_cam);
  randomPose(num_cameras, base_Ts_cam);
  //base_Ts_cam.print();
  for(int i=0;i<num_cameras;i++) {
    base_Ts_ee.push_back(base_Ts_cam[i] * ee_T_cam.inverse());
  }
  double radius = 1;
  boost::mt19937 rng;
  boost::uniform_real<> pt_dist(-radius*0.7, radius*0.7);
  boost::uniform_int<> uni_dist;
  boost::variate_generator<boost::mt19937, boost::uniform_int<> > rand_int(rng, uni_dist);
  for(int i=0;i<num_points;i++) {
    gtsam::Point3 pt_base = gtsam::Point3(pt_dist(rng), pt_dist(rng), pt_dist(rng));
    points_base.push_back(pt_base);
    vector<int> inds;
    for(int j=0;j<num_cameras;j++) {
      inds.push_back(j);
    }
    random_shuffle(inds.begin(), inds.end(), rand_int);
    // pick cameras_per_point base_Ts_cam to link this point to
    for(int j=0;j<cameras_per_point;j++) {
      points_camera.push_back(base_Ts_cam[inds[j]].inverse() * pt_base);
      pt_camera_frame_inds.push_back(inds[j]);
      pt_camera_base_ind.push_back(i);
    }
  }

  // GENERATE SOLUTION
  Values test_solution;
  test_solution.insert(Symbol('k',0), ee_T_cam.inverse());
  for(int i=0;i<num_points;i++)
    test_solution.insert(Symbol('b',i), points_base[i]);
  for(int i=0;i<num_cameras;i++)
    test_solution.insert(Symbol('e',i), base_Ts_ee[i]);
  //for(int i=0;i<num_points*cameras_per_point;i++)
  //  test_solution.insert(Symbol('a',i),
  //      ee_T_cam * base_Ts_cam[pt_camera_frame_inds[i]].inverse() * 
  //      points_base[pt_camera_base_ind[i]]);


#if 1
  // INFERENCE

  // noise terms
  noiseModel::Isotropic::shared_ptr pt_noise = noiseModel::Isotropic::Sigma(3, 0.02);
  noiseModel::Diagonal::shared_ptr cam_noise = noiseModel::Diagonal::Sigmas(Vector_(6, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03));
  //noiseModel::Diagonal::shared_ptr cam_noise = noiseModel::Diagonal::Sigmas(Vector_(6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  noiseModel::Isotropic::shared_ptr ee_fact_noise = noiseModel::Isotropic::Sigma(3, 0.02);
  noiseModel::Isotropic::shared_ptr cam_fact_noise = noiseModel::Isotropic::Sigma(3, 0.02);
  noiseModel::Diagonal::shared_ptr pose_noise = noiseModel::Diagonal::Sigmas(
      Vector_(6, 0.1, 0.1, 0.1, M_PI, M_PI, M_PI));
  noiseModel::Diagonal::shared_ptr pose_acc_noise = noiseModel::Diagonal::Sigmas(
      Vector_(6, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02));

  gtsam::NonlinearFactorGraph graph;
  Values init_estimate;
  init_estimate.insert(Symbol('k',0), Pose3());
  //initialEstimate.insert(Symbol('k',0), ee_T_cam);
  //graph.add(gtsam::PriorFactor<Pose3>(Symbol('k',0), Pose3(), pose_noise));



  /*
  graph.add(gtsam::PriorFactor<Pose3>(Symbol('k',0), ee_T_cam.inverse(), pose_noise));
  for(size_t i=0;i<points_base.size();i++)
    graph.add(gtsam::PriorFactor<Point3>(Symbol('b',i), points_base[i], pt_noise));
  for(int i=0;i<num_cameras;i++)
    graph.add(gtsam::PriorFactor<Pose3>(Symbol('e',i), base_Ts_ee[i], pose_acc_noise));
  */


  printf("here %d\n", pt_camera_frame_inds.size());
  for(size_t i=0;i<pt_camera_frame_inds.size();i++) {
    gtsam::Point3 pt_cam = points_camera[i];
    gtsam::Pose3 base_T_ee = base_Ts_ee[pt_camera_frame_inds[i]];

    // add factors
    graph.add(boost::make_shared<MiddlePoseFactor>(ee_fact_noise, 
                                                   Symbol('k',0), 
                                                   Symbol('b',pt_camera_base_ind[i]),
                                                   base_T_ee.inverse(),
                                                   pt_cam));

    /*
    graph.add(boost::make_shared<RelativePoseFactor>(ee_fact_noise, 
                                                     Symbol('a',i), 
                                                     Symbol('b',pt_camera_base_ind[i]),
                                                     base_T_ee.inverse()));
    graph.add(boost::make_shared<RelativePointFactor>(cam_fact_noise, 
                                                      Symbol('k',0), 
                                                      Symbol('a',i),
                                                      pt_cam));
    */
    /*
    graph.add(boost::make_shared<DoublePoseFactor>(ee_fact_noise, 
                                                   Symbol('k',0), 
                                                   Symbol('e',pt_camera_frame_inds[i]), 
                                                   Symbol('b',pt_camera_base_ind[i]),
                                                   pt_cam));
    */

    init_estimate.insert(Symbol('a',i),pt_cam);
    if(!init_estimate.exists(Symbol('b',pt_camera_base_ind[i])))
      init_estimate.insert(Symbol('b',pt_camera_base_ind[i]),
                           base_T_ee * pt_cam);
  }
  //for(int i=0;i<num_cameras;i++)
  //  init_estimate.insert(Symbol('e',i), base_Ts_ee[i]);

  graph.print();
  //printf("opt error: %f\n", graph.error(test_solution));

  gtsam::LevenbergMarquardtParams lm_params;
  lm_params.setlambdaInitial(100.0);
  //Values result = LevenbergMarquardtOptimizer(graph, init_estimate).optimizeSafely();
  Values result = DoglegOptimizer(graph, init_estimate).optimize();
  result.print();
  return;
  try {
    //Values result = LevenbergMarquardtOptimizer(graph, test_solution, lm_params).optimize();
    //Values result = DoglegOptimizer(graph, test_solution).optimize();
  }
  catch (gtsam::IndeterminantLinearSystemException& e) {
    printf("nearby: %d\n", e.nearbyVariable());
    //test_solution.at<Point3>(e.nearbyVariable()).print();
  }
  //Values result = LevenbergMarquardtOptimizer(graph, init_estimate).optimize();
  //Values result = LevenbergMarquardtOptimizer(graph, initialEstimate).optimize();
  LevenbergMarquardtOptimizer lmo(graph, init_estimate, lm_params);
  for(int i=0;i<1000;i++) {
    lmo.iterate();
    printf("cur error: %f\n", graph.error(lmo.values()));
  }
  //result.print("done:\n");
  //printf("est error: %f\n", graph.error(result));

#endif


#if 0  
  // INFERENCE
  gtsam::NonlinearFactorGraph graph;
  noiseModel::Isotropic::shared_ptr pt_noise = noiseModel::Isotropic::Sigma(3, 0.02);
  noiseModel::Diagonal::shared_ptr cam_noise = noiseModel::Diagonal::Sigmas(Vector_(6, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03));
  //noiseModel::Diagonal::shared_ptr cam_noise = noiseModel::Diagonal::Sigmas(Vector_(6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
  noiseModel::Isotropic::shared_ptr ee_fact_noise = noiseModel::Isotropic::Sigma(3, 0.2);
  noiseModel::Isotropic::shared_ptr cam_fact_noise = noiseModel::Isotropic::Sigma(3, 0.2);
  Values initialEstimate;
  initialEstimate.insert(Symbol('k',0), Pose3());
  //initialEstimate.insert(Symbol('k',0), ee_T_cam);
  for(size_t i=0;i<pt_camera_frame_inds.size();i++) {
    gtsam::Point3 pt_cam = points_camera[i];
    gtsam::Pose3 pose_cam = base_Ts_ee[pt_camera_frame_inds[i]];
    
    // add measurements
    //graph.add(gtsam::PriorFactor<gtsam::Point3>(gtsam::Symbol('c',i),pt_cam,pt_noise));
    //graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('e',pt_camera_frame_inds[i]),pose_cam,cam_noise));

    // add factors
    graph.add(boost::make_shared<RelativePoseFactor>(ee_fact_noise, 
                                                     Symbol('a',i), 
                                                     Symbol('b',pt_camera_base_ind[i]),
                                                     pose_cam));
    graph.add(boost::make_shared<RelativePointFactor>(cam_fact_noise, 
                                                      Symbol('k',0), 
                                                      Symbol('a',i),
                                                      pt_cam));

    initialEstimate.insert(Symbol('a',i),pt_cam);
#if 0
    if(!initialEstimate.exists(Symbol('b',pt_camera_base_ind[i])))
      initialEstimate.insert(Symbol('b',pt_camera_base_ind[i]),pose_cam.inverse()*pt_cam);
#endif
    //initialEstimate.insert(Symbol('c',i),pt_cam);
  }
#if 0
  for(size_t i=0;i<num_cameras;i++) {
    gtsam::Pose3 pose_cam = base_Ts_ee[i];
    initialEstimate.insert(Symbol('e',i),pose_cam);
  }
#endif
#if 1
  for(size_t i=0;i<num_points;i++) {
    initialEstimate.insert(Symbol('b',i), points_base[i]);
  }
  //for(size_t i=0;i<;i++) {
#endif
  graph.print();
  printf("error %f\n", graph.error(initialEstimate));

  Values result = DoglegOptimizer(graph, initialEstimate).optimize();
  //Values result = LevenbergMarquardtOptimizer(graph, initialEstimate).optimize();
  result.print("done:\n");
  printf("opt error: %f\n", graph.error(initialEstimate));
#endif
}

int main(int argc, char* argv[])
{
  int num_cameras = 6;
  int num_points = 20;
  int cameras_per_point = 3;
  armMappingTestData2(num_cameras, num_points, cameras_per_point);
}
