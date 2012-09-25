/*********************************************************************
 *
 *  Copyright (c) 2012, Georgia Institute of Technology
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Georgia Institute of Technology nor the names of
 *     its contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

// ROS
#include <ros/ros.h>
#include <ros/package.h>

#include <std_msgs/Header.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// TF
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>

// Boost
#include <boost/shared_ptr.hpp>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// STL
#include <vector>
#include <deque>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <utility>
#include <stdexcept>
#include <float.h>
#include <math.h>
#include <time.h> // for srand(time(NULL))
#include <cstdlib> // for MAX_RAND
#include <sstream>

// PR2_GRIPPER_SENSOR_ACTION
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <visual_servo/VisualServoAction.h>

// Others
#include <visual_servo/VisualServoTwist.h>
#include <visual_servo/VisualServoPose.h>
#include <std_srvs/Empty.h>
#include "visual_servo.cpp"

// statemachine constants
#define INIT          0
#define SETTLE        1
#define INIT_OBJS     2
#define INIT_HAND     3
#define INIT_DESIRED  4
#define POSE_CONTR    5
#define POSE_CONTR_2  6
#define VS_CONTR_1    7
#define VS_CONTR_2    8
#define GRAB          9
#define RELOCATE      10
#define DESCEND_INIT  11
#define DESCEND       12
#define RELEASE       13
#define FINISH        14
#define TERM          15

// floating point mod
#define fmod(a,b) a - (float)((int)(a/b)*b)

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,sensor_msgs::PointCloud2> MySyncPolicy;
typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;
typedef actionlib::SimpleActionClient<visual_servo::VisualServoAction> VisualServoClient; 

using boost::shared_ptr;
using geometry_msgs::TwistStamped;
using geometry_msgs::PoseStamped;
using geometry_msgs::QuaternionStamped;
using geometry_msgs::Point;
using visual_servo::VisualServoTwist;
using visual_servo::VisualServoPose;

class VisualServoNode
{
  public:
    VisualServoNode(ros::NodeHandle &n) :
      n_(n), n_private_("~"),
      image_sub_(n, "color_image_topic", 1),
      depth_sub_(n, "depth_image_topic", 1),
      cloud_sub_(n, "point_cloud_topic", 1),
      sync_(MySyncPolicy(15), image_sub_, depth_sub_, cloud_sub_),
      it_(n), tf_(), have_depth_data_(false), camera_initialized_(false),
      PHASE(INIT)
  {
    vs_ = shared_ptr<VisualServo>(new VisualServo(JACOBIAN_TYPE_PSEUDO));
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());

    n_private_.param("display_wait_ms", display_wait_ms_, 3);

    std::string default_optical_frame = "/head_mount_kinect_rgb_optical_frame";
    n_private_.param("optical_frame", optical_frame_, default_optical_frame);
    std::string default_workspace_frame = "/torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_, default_workspace_frame);
    std::string cam_info_topic_def = "/kinect_head/rgb/camera_info";
    n_private_.param("cam_info_topic", cam_info_topic_, cam_info_topic_def);

    // color segmentation parameters
    n_private_.param("target_hue_value", target_hue_value_, 10);
    n_private_.param("target_hue_threshold", target_hue_threshold_, 20);
    n_private_.param("gripper_tape_hue_value", gripper_tape_hue_value_, 180);
    n_private_.param("gripper_tape_hue_threshold", gripper_tape_hue_threshold_, 50);
    n_private_.param("default_sat_bot_value", default_sat_bot_value_, 40);
    n_private_.param("default_sat_top_value", default_sat_top_value_, 40);
    n_private_.param("default_val_value", default_val_value_, 200);
    n_private_.param("min_contour_size", min_contour_size_, 10.0);

    // others
    n_private_.param("jacobian_type", jacobian_type_, JACOBIAN_TYPE_INV);
    n_private_.param("vs_err_term_thres", vs_err_term_threshold_, 0.001);
    n_private_.param("pose_servo_z_offset", pose_servo_z_offset_, 0.045);
    n_private_.param("place_z_velocity", place_z_velocity_, -0.025);
    n_private_.param("gripper_tape1_offset_x", tape1_offset_x_, 0.02);
    n_private_.param("gripper_tape1_offset_y", tape1_offset_y_, -0.025);
    n_private_.param("gripper_tape1_offset_z", tape1_offset_z_, 0.07);

    // Setup ros node connections
    sync_.registerCallback(&VisualServoNode::sensorCallback, this);
    initializeService();
    ROS_INFO("Initialization 0: Node init & Register Callback Done");
  }

    // destructor
    ~VisualServoNode()
    {
    }

    /**
     * Called when Kinect information is avaiable. Refresh rate of about 30Hz 
     */
    void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg, 
        const sensor_msgs::ImageConstPtr& depth_msg, const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {

      ros::Time start = ros::Time::now();

      // Store camera information only once
      if (!camera_initialized_)
      {
        cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic_, n_, ros::Duration(3.0));
        camera_initialized_ = true;
        vs_->setCamInfo(cam_info_);
        ROS_INFO("Initialization: Camera Info Done");
      }

      cv::Mat color_frame, depth_frame;
      cv_bridge::CvImagePtr color_cv_ptr = cv_bridge::toCvCopy(img_msg);
      cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg);

      color_frame = color_cv_ptr->image;
      depth_frame = depth_cv_ptr->image;

      XYZPointCloud cloud; 
      pcl::fromROSMsg(*cloud_msg, cloud);
      tf_->waitForTransform(workspace_frame_, cloud.header.frame_id,
          cloud.header.stamp, ros::Duration(0.9));
      pcl_ros::transformPointCloud(workspace_frame_, cloud, cloud, *tf_);
      cur_camera_header_ = img_msg->header;
      cur_color_frame_ = color_frame;
      cur_orig_color_frame_ = color_frame.clone();
      cur_depth_frame_ = depth_frame.clone();
      cur_point_cloud_ = cloud;

      // need to profile this
      executeStatemachine();

#define DISPLAY 1
#ifdef DISPLAY
      // show a pretty imshow
      // setDisplay();
#endif
    }

    void executeStatemachine()
    {
      temp_draw_.clear();
      switch(PHASE)
      {
        case INIT:
          {
            ROS_INFO("Initializing Services and Robot Configuration");
            std_srvs::Empty e;
            i_client_.call(e);
            PHASE = SETTLE;
          }
          break;

        case SETTLE:
          {
            ROS_INFO("Phase Settle: Move Gripper to Init Position");
            // Move Hand to Some Preset Pose
            visual_servo::VisualServoPose p_srv = formPoseService(0.62, 0.05, -0.1);
            p_srv.request.arm = "l";
            if (p_client_.call(p_srv))
            {
              PHASE = VS_CONTR_1;
            }
            else
            {
              ROS_WARN("Failed to put the hand in initial configuration");
              ROS_WARN("For debug purpose, just skipping to the next step");
              PHASE = VS_CONTR_1;
            }
          }
          break;

        case VS_CONTR_1:
          {
            // send a goal to the action
            visual_servo::VisualServoGoal goal;
            PoseStamped p;
            p.pose.position.x = 0.55;
            p.pose.position.y = 0.2;
            p.pose.position.z = 0.05;
            goal.pose = p;

            double timeout = 30.0;
            ROS_INFO("Sending Goal [%.3f %.3f %.3f] (TO = %.1f)", p.pose.position.x,
            p.pose.position.y, p.pose.position.z, timeout);
            l_vs_client_->sendGoal(goal);

            bool finished_before_timeout = l_vs_client_->waitForResult(ros::Duration(timeout));

            if (finished_before_timeout)
            {
              actionlib::SimpleClientGoalState state = l_vs_client_->getState();
              ROS_INFO("Action finished: %s",state.toString().c_str());
            }
          }
          break;

        default:
          {
            // make the list shorter
            ROS_INFO("Routine Ended.");
            std::cout << "Press [Enter] if you want to do it again: ";
            while(!ros::isShuttingDown())
            {
              int c = std::cin.get();
              if (c  == '\n')
                break;
            }

            printf("Reset the arm and repeat Pick and Place in 3 seconds\n");
            // try to initialize the arm
            std_srvs::Empty e;
            i_client_.call(e);

          }
          break;
      }
    }

    void setDisplay()
    {
      cv::imshow("in", cur_orig_color_frame_); 
      cv::waitKey(display_wait_ms_);
    }

    void initializeService()
    {
      ROS_DEBUG(">> Hooking Up The Service");
      ros::NodeHandle n;
      v_client_ = n.serviceClient<visual_servo::VisualServoTwist>("vs_twist");
      p_client_ = n.serviceClient<visual_servo::VisualServoPose>("vs_pose");
      i_client_ = n.serviceClient<std_srvs::Empty>("vs_init");

      // uncomment below depending on which arm you want
      // l_vs_client_ = new VisualServoClient(n_, "r_vs_controller/vsaction");
      l_vs_client_ = new VisualServoClient(n_, "l_vs_controller/vsaction");
      while(!l_vs_client_->waitForServer(ros::Duration(5.0))){
        ROS_INFO("Waiting for Visual Servo action...");
      }
    }


    /**********************
     * HELPER METHODS
     **********************/

    bool sendZeroVelocity()
    {
      // assume everything is init to zero
      visual_servo::VisualServoTwist v_srv;

      // need this to mvoe the arm
      v_srv.request.error = 1;
      return v_client_.call(v_srv);
    }

    visual_servo::VisualServoPose formPoseService(float px, float py, float pz)
    {
      return VisualServoMsg::createPoseMsg(px, py, pz, -0.4582, 0, 0.8889, 0);
    }

    void printMatrix(cv::Mat_<double> in)
    {
      for (int i = 0; i < in.rows; i++) {
        for (int j = 0; j < in.cols; j++) {
          printf("%+.5f\t", in(i,j)); 
        }
        printf("\n");
      }
    }


    /**
     * Executive control function for launching the node.
     */
    void spin()
    {
      while(n_.ok())
      {
        ros::spinOnce();
      }
    }

  protected:
    ros::NodeHandle n_;
    ros::NodeHandle n_private_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Synchronizer<MySyncPolicy> sync_;
    image_transport::ImageTransport it_;
    sensor_msgs::CameraInfo cam_info_;
    shared_ptr<tf::TransformListener> tf_;
    cv::Mat cur_color_frame_;
    cv::Mat cur_orig_color_frame_;
    cv::Mat cur_depth_frame_;
    cv::Mat cur_workspace_mask_;
    std_msgs::Header cur_camera_header_;
    XYZPointCloud cur_point_cloud_;

    // visual servo object
    shared_ptr<VisualServo> vs_;

    bool have_depth_data_;
    int display_wait_ms_;
    int num_downsamples_;
    std::string workspace_frame_;
    std::string optical_frame_;
    bool camera_initialized_;
    bool desire_points_initialized_;
    std::string cam_info_topic_;
    int tracker_count_;

    // clients to services provided by vs_controller.py
    ros::ServiceClient v_client_;
    ros::ServiceClient p_client_;
    ros::ServiceClient i_client_;


    // segmenting
    int target_hue_value_;
    int target_hue_threshold_;
    int gripper_tape_hue_value_;
    int gripper_tape_hue_threshold_;
    int default_sat_bot_value_;
    int default_sat_top_value_;
    int default_val_value_;
    double min_contour_size_;

    // Other params
    int jacobian_type_;
    double vs_err_term_threshold_;
    double pose_servo_z_offset_;
    double place_z_velocity_;
    double tape1_offset_x_;
    double tape1_offset_y_;
    double tape1_offset_z_;

    // State machine variable
    unsigned int PHASE;

    // gripper sensor action clients
    VisualServoClient* l_vs_client_;
    VisualServoClient* r_vs_client_;

    //  drawing
    std::vector<cv::Point> temp_draw_;
    cv::Rect original_box_;

    // collision detection
    bool is_detected_;
    bool place_detection_;
    float object_z_;

    float close_gripper_dist_;

};

int main(int argc, char ** argv)
{
  srand(time(NULL));
  ros::init(argc, argv, "visual_servo_node");

  log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
  my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);

  ros::NodeHandle n;
  VisualServoNode vs_node(n);
  vs_node.spin();
}
