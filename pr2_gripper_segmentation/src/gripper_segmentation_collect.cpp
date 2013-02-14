/*
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
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/JointState.h>

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
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/features2d.hpp>


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
#include <sys/time.h>

// cpl_visual_features
#include <cpl_visual_features/helpers.h>
#include <cpl_visual_features/features/shape_context.h>

// others
#include <pr2_controllers_msgs/JointTrajectoryControllerState.h>
#include <pr2_gripper_segmentation/GripperPose.h>

/*
   typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
   sensor_msgs::Image,
   sensor_msgs::Image,
   sensor_msgs::PointCloud2> MySyncPolicy;
 */
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
        sensor_msgs::Image,
        sensor_msgs::PointCloud2> MySyncPolicy;

typedef pcl::PointCloud<pcl::PointXYZ> XYZPointCloud;

using sensor_msgs::JointState;
using boost::shared_ptr;
using geometry_msgs::TwistStamped;
using geometry_msgs::PointStamped;
using geometry_msgs::PoseStamped;
using geometry_msgs::QuaternionStamped;
using geometry_msgs::Point;
using cpl_visual_features::downSample;

struct Distance {
  float x;
  float y;
  float z;
};

struct Data{
  std::string frame;
  std::vector<cv::KeyPoint> kp;
  std::vector<Distance> d;
  std::vector<float> dist;
  // extracted
  cv::Mat desc;
};

struct mouseEvent {
  cv::Point cursor;
  int event;
};


void onMouse(int event, int x, int y, int flags, void* param)
{
  //std::map <std::vector<cv::Point>, void*> *user_data_n_ptr; 
  //std::vector<cv::Point> points = reinterpret_cast<std::map< std::vector<cv::Point>, void*> *>(param);

  mouseEvent *pt = static_cast<mouseEvent *>(param);
  switch(event)
  {
    case CV_EVENT_LBUTTONUP:
      {
        pt->cursor = cv::Point(x,y);
        pt->event = event;
        // ROS_DEBUG("PRESSED AT [%d, %d]", x,y);
      }
      break;
    case CV_EVENT_MOUSEMOVE:
      {
        // if previous event was mouse move too
        if (pt->event == CV_EVENT_MOUSEMOVE)
        {
          pt->cursor = cv::Point(x,y);
          pt->event = event;
        }
      }
      break;
  }
  return;
}


class GripperSegmentationCollector
{
  public:
    GripperSegmentationCollector(ros::NodeHandle &n) :
      n_(n), n_private_("~"),
      image_sub_(n_, "color_image_topic", 1),
      depth_sub_(n_, "depth_image_topic", 1),
      mask_sub_(n_, "mask_image_topic", 1),
      cloud_sub_(n_, "point_cloud_topic", 1),
      // sync_(MySyncPolicy(15), image_sub_, depth_sub_, mask_sub_, cloud_sub_),
      sync_(MySyncPolicy(15), image_sub_, depth_sub_, cloud_sub_),
      it_(n_), tf_(), camera_initialized_(false)
  {
    tf_ = shared_ptr<tf::TransformListener>(new tf::TransformListener());

    n_private_.param("display_wait_ms", display_wait_ms_, 3);
    n_private_.param("num_downsmaples", num_downsamples_, 2);

    std::string default_optical_frame = "/head_mount_kinect_rgb_optical_frame";
    n_private_.param("optical_frame", optical_frame_, default_optical_frame);
    std::string cam_info_topic_def = "/kinect_head/rgb/camera_info";
    n_private_.param("cam_info_topic", cam_info_topic_, cam_info_topic_def);
    std::string default_workspace_frame = "/torso_lift_link";
    n_private_.param("workspace_frame", workspace_frame_, default_workspace_frame);

    std::string default_left_tool_frame = "/l_gripper_tool_frame";
    n_private_.param("l_tool_frame", left_tool_frame_, default_left_tool_frame);
    std::string default_right_tool_frame = "/r_gripper_tool_frame";
    n_private_.param("r_tool_frame", right_tool_frame_, default_right_tool_frame);

    n_private_.param("target_hue_value", target_hue_value_, 350);
    n_private_.param("target_hue_threshold", target_hue_threshold_, 48);
    n_private_.param("default_sat_bot_value", default_sat_bot_value_, 20);
    n_private_.param("default_sat_top_value", default_sat_top_value_, 100);
    n_private_.param("default_val_value", default_val_value_, 100);
    n_private_.param("min_contour_size", min_contour_size_, 10.0);


    // Setup ros node connections
    sync_.registerCallback(&GripperSegmentationCollector::sensorCallback, this);
    // n_.subscribe("/joint_states", 1, &GripperSegmentationCollector::jointStateCallback, this);
    p_client_ = n.serviceClient<pr2_gripper_segmentation::GripperPose>("pgs_pose");
    textFile.open("/u/swl33/data/myData.csv");

    x_= 0.55;
    y_= 0.0;
    z_= -0.15;
    which_arm_ = "l";
    mode = 0;
    counter_ = 0;

    ROS_INFO("[GripperSeg] Initial Pose");
    pr2_gripper_segmentation::GripperPose psrv;
    PoseStamped p;
    p.header.frame_id = workspace_frame_;
    p.pose.position.x = x_;
    p.pose.position.y = y_;
    p.pose.position.z = z_;
    p.pose.orientation.w = 1;
    psrv.request.arm = which_arm_;
    psrv.request.p = p;
    if (p_client_.call(psrv)){}
    else {ROS_WARN("Service Fail");}


    ROS_INFO("[GripperSeg] Node Initialization Complete");
  }

    // destructor
    ~GripperSegmentationCollector()
    {
      textFile.close();
    }


  private:

    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& js)
      //void jointStateCallback(boost::shared_ptr<sensor_msgs::JointState const> js)
    {
      for (unsigned int i = 0; i < js->name.size(); i++)
      {
        ROS_INFO_STREAM("Joint Name" << js->name[i]);
      }
    }
    unsigned int counter;
    /**
     * Called when Kinect information is avaiable. Refresh rate of about 30Hz 
     */
    /*
       void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg,
       const sensor_msgs::ImageConstPtr& depth_msg,
       const sensor_msgs::ImageConstPtr& mask_msg,
       const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
     */

    void move(cv::Point p, std::vector<cv::KeyPoint> *from, std::vector<cv::KeyPoint> *to)
    {
      for (unsigned int i = 0; i <  from->size(); i++)
      {
        if (abs(from->at(i).pt.x - p.x) < 7
            && abs(from->at(i).pt.y - p.y) < 7)
        {
          // ROS_INFO("[GripperSeg] Added at [%0.3f, %0.3f]", kps_.at(i).pt.x, kps_.at(i).pt.y);
          to->push_back(from->at(i));
          from->erase(from->begin() + i);
          return;
        }
      }
    }

    void process(mouseEvent me)
    {
      cv::Point p = me.cursor;

      if (me.event == CV_EVENT_LBUTTONUP)
      {
        if (mode == 0)
        {
          pcl::PointXYZ tcp = cloud_.at(p.x, p.y);
          if (isnan(tcp.x)||isnan(tcp.y)||isnan(tcp.z))
          {
            ROS_WARN("Invalid Point. Try again");
            return;
          }
          float rmse = pow(pow(tcp.x - tcp_e_.x, 2) + pow(tcp.y - tcp_e_.y, 2) + pow(tcp.z - tcp_e_.z, 2),0.5);
          ROS_INFO("*** Tooltip added at [%.5f, %.5f, %.5f] RMSE: [%.7f]", tcp.x, tcp.y, tcp.z, rmse);
          tooltip_ = p;
          mode = 1;
        }
        else
        {
          pcl::PointXYZ td = cloud_.at(p.x, p.y);
          // cannot add these
          if (isnan(td.x) || isnan(td.y) || isnan(td.z))
          {
            move(p, &kps_, &kps_bad_);
            return;
          }
          move(p, &kps_, &kpsc_);
          /*
             for (unsigned int i = 0; i <  kps_.size(); i++)
             {
             if (abs(kps_.at(i).pt.x - p.x) < 7
             && abs(kps_.at(i).pt.y - p.y) < 7)
             {
             ROS_INFO("[GripperSeg] Added at [%0.3f, %0.3f]", kps_.at(i).pt.x, kps_.at(i).pt.y);
             kpsc_.push_back(kps_.at(i));
             kps_.erase(kps_.begin() + i);
             break;
             }
             }
           */
        }
      }
    }
    void setDisplay(cv::Mat color_frame, mouseEvent me)
    {
      setDisplay(color_frame, me, "");
    }

    void setDisplay(cv::Mat color_frame, mouseEvent me, std::string frame)
    {
      if (frame.length() > 3) 
      {
        std::ostringstream os;
        os << " [Working on: " << frame << "]";
        cv::putText(color_frame, os.str(), cv::Point(5,15), 1, 1, cv::Scalar(255, 255, 255), 1, 465, false);
      } 
      else 
      {
        std::ostringstream os;
        os << " [" << ++counter_ << "]";
        cv::putText(color_frame, os.str(), cv::Point(5,15), 1, 1, cv::Scalar(255, 255, 255), 1, 465, false);
      }
      if (me.event != CV_EVENT_LBUTTONUP)
      {
        cv::circle(color_frame, me.cursor, 4, cv::Scalar(235, 235, 235), 1);
      }

      cv::putText(color_frame, "Press [N] to move to next frame", cv::Point(5,15), 1, 1, cv::Scalar(255, 255, 255), 1, 8, false);

      if (mode == 0)
      {
        // red box to indicate the first mode
        cv::rectangle(color_frame, cv::Point(2,2), cv::Point(637,477), cv::Scalar(0, 0, 255), 3);
      }
      else
      {
        // green to indicate the next step
        cv::rectangle(color_frame, cv::Point(1,1), cv::Point(637,3), cv::Scalar(0, 255, 0), 2);

        // show current TCP
        /*
        cv::circle(color_frame, tooltip_, 4, cv::Scalar(0, 255, 0), 4);
        pcl::PointXYZ tcp = cloud_.at(tooltip_.x, tooltip_.y);
        std::ostringstream os;
        os << " [" << tcp.x << ", " << tcp.y << ", " << tcp.z << "]";
        cv::putText(color_frame, os.str(), tooltip_, 2, 0.5, cv::Scalar(0, 255, 0), 1);
        */
      }

      for (unsigned int i = 0 ; i < kps_bad_.size(); i++)
      {
        cv::KeyPoint kp = kps_bad_.at(i);
        // cv::circle(color_frame, (int)(kp.pt.x), (int)(kp.pt.y), 2, cv::Scalar(0, 0, 255), 2);
        cv::circle(color_frame, cv::Point(kp.pt.x, kp.pt.y), 2, cv::Scalar(0, 0, 255), 2);
      }
      for (unsigned int i = 0 ; i < kps_.size(); i++)
      {
        cv::KeyPoint kp = kps_.at(i);
        // cv::circle(color_frame, (int)(kp.pt.x), (int)(kp.pt.y), 2, cv::Scalar(0, 0, 255), 2);
        cv::circle(color_frame, cv::Point(kp.pt.x, kp.pt.y), 2, cv::Scalar(50, 245, 170), 2);
      }
      for (unsigned int i = 0 ; i < kpsc_.size(); i++)
      {
        cv::KeyPoint kp = kpsc_.at(i);
        // cv::circle(color_frame, cv::Point(kp.pt.x, kp.pt.y), 2, cv::Scalar(255, 255, 0), 0);
        cv::circle(color_frame, cv::Point(kp.pt.x, kp.pt.y), 2, cv::Scalar(255, 40, 30), 2);
      }

      if (good_matches_.size() > 0)
      {
        for (unsigned int i = 0; i < good_matches_.size(); i++)
        {
          cv::Point2f cp = kpso_[good_matches_[i].trainIdx].pt;
          cv::circle(color_frame, cv::Point(cp.x, cp.y), 5, cv::Scalar(200, 70, 200), 0);
        }
        cv::Point p = projectPointIntoImage(tcp_e_, "/torso_lift_link", "/head_mount_kinect_rgb_optical_frame", tf_);
        cv::circle(color_frame, p, 4, cv::Scalar(255, 0, 0), 3);
      }
      cv::imshow("what", color_frame);
    }

    XYZPointCloud averagePC(XYZPointCloud cloud1, XYZPointCloud cloud2)
    {
      float alpha = 0.0;
      for (int i = 0; i < 640; i++)
      {
        for (int j = 0; j < 480; j++)
        {
          pcl::PointXYZ p1 = cloud1.at(i,j);
          pcl::PointXYZ p2 = cloud2.at(i,j);
          p2.x = isnan(p1.x) ? p2.x : isnan(p2.x) ? p1.x : alpha*p1.x + (1-alpha)*p2.x;
          p2.y = isnan(p1.y) ? p2.y : isnan(p2.y) ? p1.y : alpha*p1.y + (1-alpha)*p2.y;
          p2.z = isnan(p1.z) ? p2.z : isnan(p2.z) ? p1.z : alpha*p1.z + (1-alpha)*p2.z;
          cloud2.at(i,j) = p2;
        }
      }
      return cloud2;
    }

    void sensorCallback(const sensor_msgs::ImageConstPtr& img_msg,
        const sensor_msgs::ImageConstPtr& depth_msg,
        const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
    {
      // Store camera information only once
      if (!camera_initialized_)
      {
        cam_info_ = *ros::topic::waitForMessage<sensor_msgs::CameraInfo>(cam_info_topic_, n_, ros::Duration(3.0));
        camera_initialized_ = true;
        counter = 0;
        ROS_INFO("[GripperCollector]Initialization: Camera Info Done");

      }

      timespec t1, t2;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);


      cv::Mat color_frame, depth_frame, self_mask;
      cv_bridge::CvImagePtr color_cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
      cv_bridge::CvImagePtr depth_cv_ptr = cv_bridge::toCvCopy(depth_msg);

      color_frame = color_cv_ptr->image;
      depth_frame = depth_cv_ptr->image;

      // cv_bridge::CvImagePtr mask_cv_ptr = cv_bridge::toCvCopy(mask_msg);
      // self_mask = mask_cv_ptr->image;
      XYZPointCloud cloud;
      pcl::fromROSMsg(*cloud_msg, cloud);
      tf_->waitForTransform(workspace_frame_, cloud.header.frame_id,
          cloud.header.stamp, ros::Duration(33e-3));
      pcl_ros::transformPointCloud(workspace_frame_, cloud, cloud, *tf_);


      cur_camera_header_ = img_msg->header;
      if (cloud_.size() == 0)
        cloud_ = cloud;
      else
        cloud_ = averagePC(cloud_, cloud);


      pcl::PointXYZ est1 = printValue1(color_frame);
      pcl::PointXYZ est2 = printValue2(color_frame); 

      // Keypoint selection
      cv::SURF surf;
      cv::Mat mask;
      std::vector<cv::KeyPoint> keypoints;
      surf(color_frame, mask, keypoints);
      kps_ = keypoints;
      kpso_ = keypoints;

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
      // printf("SCB: "); printTime(t1, t2);




      // if we have arm model, estimate the tooltip

      if (ds_.size() > 0 && ds_[0].kp.size() > 0)
      {
        pcl::PointXYZ est;
        for (unsigned int i = 0; i < ds_.size(); i++) 
        {
          pcl::PointXYZ est3;
          good_matches_ = matchAndQuery(ds_[0], kps_, color_frame, &est3);
          ROS_WARN("  EST[%f %f %f]", est3.x, est3.y, est3.z);
          //ROS_WARN("EST-Color[%f][%f %f %f]", pow(pow(est3.x-est1.x,2)+pow(est3.y-est1.y,2)+pow(est3.z-est1.z,2), 0.5) ,est3.x-est1.x,est3.y-est1.y,est3.z-est1.z);
          est.x += est3.x;
          est.y += est3.y;
          est.z += est3.z;
        }
        ROS_FATAL("EST[%f %f %f]", est.x/2, est.y/2, est.z/2);

         //ROS_DEBUG("E3-E1[%f][%f %f %f]", pow(pow(est3.x-est1.x,2)+pow(est3.y-est1.y,2)+pow(est3.z-est1.z,2), 0.5) ,est3.x-est1.x,est3.y-est1.y,est3.z-est1.z);
         //ROS_DEBUG("E3-E2[%f][%f %f %f]", pow(pow(est3.x-est2.x,2)+pow(est3.y-est2.y,2)+pow(est3.z-est2.z,2), 0.5) ,est3.x-est2.x,est3.y-est2.y,est3.z-est2.z);
      }

      // GUI Related
      int key = 0;
      cv::namedWindow("what", CV_WINDOW_AUTOSIZE);
      mouseEvent m;
      setDisplay(color_frame.clone(), m);
      cv::setMouseCallback("what", onMouse, (void*) &m);
      key = cv::waitKey(5);
      kpsc_.clear();
      kps_bad_.clear();

/*==================
      // terminate if no key is pressed
      if (key < 32)
      return;

      // current point cloud is unorganized (because of + operation)
      // organize pc

      // ========== End of Accumulation. Now GUI =========
      cv::SURF surf;
      cv::Mat mask;
      std::vector<cv::KeyPoint> keypoints;
      surf(color_frame, mask, keypoints);
      kps_ = keypoints;
      kpso_ = keypoints;


      // do matching and all 
      if (ds.kp.size() > 0)
      {
        pcl::PointXYZ est3;
        good_matches_ = matchAndQuery(ds, kps_, color_frame, est3);

        ROS_INFO("E3-E1[%f][%f %f %f]", pow(pow(est3.x-est1.x,2)+pow(est3.y-est1.y,2)+pow(est3.z-est1.z,2), 0.5) ,est3.x-est1.x,est3.y-est1.y,est3.z-est1.z);
        ROS_INFO("E3-E2[%f][%f %f %f]", pow(pow(est3.x-est2.x,2)+pow(est3.y-est2.y,2)+pow(est3.z-est2.z,2), 0.5) ,est3.x-est2.x,est3.y-est2.y,est3.z-est2.z);
        }

      // clear variables
      mode = 0;
      kpsc_.clear();
      kps_bad_.clear();
      setDisplay(color_frame.clone(), m);
      cv::setMouseCallback("what", onMouse, (void*) &m);
================= */

      if (counter_ < 5)
      {
        return;
      }
      counter_ = 0;
      std::vector<std::string> frames;
      frames.push_back("/l_gripper_tool_frame");
      frames.push_back("/l_wrist_roll_joint");
      if (mode == 0)
      {
        ds_.clear();
        for (unsigned int frame_counter = 0; frame_counter < frames.size(); frame_counter++)
        {
          ROS_INFO(">> Working on %u", frame_counter);// frames.at(frame_counter));
          Data d;
          ds_.push_back(d);
          kpsc_.clear();

          while (true)
          {
            // [n], [N] Next
            if (key == 110 || key == 78)
            {
              key = 0;
              break;
            }
            // [Space] revert the last insert
            else if (key == 32 || key == 122 || key == 90)
            {
              if (kpsc_.size() > 0)
              {
                cv::KeyPoint k = kpsc_.back();
                ROS_INFO("Reverting last insert at [%.3f %.3f]", k.pt.x, k.pt.y);
                kps_.push_back(k);
                kpsc_.erase(kpsc_.end());
              }
            }
            // [Esc], [Q], [q] Exit
            else if (key == 27 || key == 113 || key == 81)
            {
              ros::shutdown();
              return;
            }
            process(m);
            setDisplay(color_frame.clone(), m);
            m.event = 0;
            key = cv::waitKey(2);
          }

          ROS_INFO("\t>> Now Storing the ARM MODEL for %d", frame_counter);

          // ==========ARM MODEL STORAGE ===========/
          cv::SurfDescriptorExtractor surfDesc;
          cv::Mat descriptors1;
          surfDesc.compute(color_frame, kpsc_, descriptors1);
          ds_[frame_counter].frame = frames[frame_counter];
          ds_[frame_counter].kp = kpsc_;
          ds_[frame_counter].desc = descriptors1;
          pcl::PointXYZ tcp = cloud_.at(tooltip_.x, tooltip_.y); 

          for (unsigned int i = 0; i < kpsc_.size(); i++)
          {
            Distance d;
            pcl::PointXYZ cur = cloud_.at(kpsc_.at(i).pt.x, kpsc_.at(i).pt.y);
            d.x = tcp.x - cur.x;
            d.y = tcp.y - cur.y;
            d.z = tcp.z - cur.z;
            //ds.dist.push_back(pow(pow(tcp.x-cur.x,2)+pow(tcp.y-cur.y,2)+pow(tcp.z-cur.z,2),0.5));
            ds_[frame_counter].dist.push_back(distance(tcp, cur));
            ds_[frame_counter].d.push_back(d);
          }
          // mode = 0;
          ROS_INFO(" >> Added %d", (int)ds_[frame_counter].d.size());
        }

        // ========== RESET VALUES BEFORE ACCUMULATION ========= /
        cloud_ = cloud;
        // counter_ = 0;

      }

      // Downsample everything first
      /*
         cv::Mat color_frame_down = downSample(color_frame, num_downsamples_);
         cv::Mat depth_frame_down = downSample(depth_frame, num_downsamples_);
         cv::Mat self_mask_down = downSample(self_mask, num_downsamples_);
       */

      /*

      // ====== L,R JOINT ANGLE ======
      double current_angles[14];
      get_current_joint_angles(current_angles);
      textFile << counter << ",";
      for (int i = 0; i < 14; i++)
      {
      textFile << current_angles[i] << ",";
      }
      // ====== FOWARD KINEMATICS L,R TOOLTIP POSE ======
      get_fk_tooltip_pose(current_angles);
      for (int i = 0; i < 14; i++)
      {
      textFile << current_angles[i] << ",";
      }

      textFile << "\n";

      std::string file1 = getFileName(counter, which_arm_);
      cv::imwrite(file1 + ".jpg", color_frame);
      cv::imwrite(file1 + "m.jpg", self_mask);

      //cv::imshow("show", color_frame);
      //cv::waitKey(3);
       */ 
      // ROS_INFO("[%u][%f %f %f] Sampled", counter, x_, y_,z_);
      //#else

      // Move the arm
      counter++;
      y_+= 0.025;
      if (y_ > 0.35)
      {
        y_ = -0.2;
        x_ += 0.1;
        if (x_ > 0.6)
        {
          x_ = 0.4;
          z_+= 0.10;
          if (z_ > 0.00)
          {
            z_ = -0.2;
          }
        }
      }
      pr2_gripper_segmentation::GripperPose psrv;
      PoseStamped p;
      p.header.frame_id = workspace_frame_;
      p.pose.position.x = x_;
      p.pose.position.y = y_;
      p.pose.position.z = z_;
      p.pose.orientation.w = 1;
      psrv.request.arm = which_arm_;
      psrv.request.p = p;
      if (p_client_.call(psrv)){}
      else {ROS_WARN("Service Fail");}
    }


    /**
     * 
     * @param in  single channel image input
     * @param color_frame  need the original image for debugging and imshow
     * 
     * @return    returns ALL moment of specific color in the image
     **/
    std::vector<cv::Moments> findMoments(cv::Mat in, cv::Mat &color_frame, unsigned int max_num = 3) 
    {
      cv::Mat temp = in.clone();
      std::vector<std::vector<cv::Point> > contours; contours.clear();
      cv::findContours(temp, contours, cv::RETR_CCOMP,CV_CHAIN_APPROX_NONE);
      std::vector<cv::Moments> moments; moments.clear();

      for (unsigned int i = 0; i < contours.size(); i++) {
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 > min_contour_size_) {
          // first add the forth element
          moments.push_back(m);
          // find the smallest element of 4 and remove that
          if (moments.size() > max_num) {
            double small(moments.at(0).m00);
            unsigned int smallInd(0);
            for (unsigned int j = 1; j < moments.size(); j++){
              if (moments.at(j).m00 < small) {
                small = moments.at(j).m00;
                smallInd = j;
              }
            }
            moments.erase(moments.begin() + smallInd);
          }
        }
      }
      return moments;
    }

    cv::Mat colorSegment(cv::Mat color_frame, int hue, int threshold)
    {
      /*
       * Often value = 0 or 255 are very useless. 
       * The distance at those end points get very close and it is not useful
       * Same with saturation 0. Low saturation makes everything more gray scaled
       * So the default setting are below 
       */
      return colorSegment(color_frame, hue - threshold, hue + threshold,
          default_sat_bot_value_, default_sat_top_value_, 40, default_val_value_);
    }

    /** 
     * Very Basic Color Segmentation done in HSV space
     * Takes in Hue value and threshold as input to compute the distance in color space
     *
     * @param color_frame   color input from image
     *
     * @return  mask from the color segmentation 
     */
    cv::Mat colorSegment(cv::Mat color_frame, int _hue_n, int _hue_p, int _sat_n, int _sat_p, int _value_n,  int _value_p)
    {
      cv::Mat temp (color_frame.clone());
      cv::cvtColor(temp, temp, CV_BGR2HSV);
      std::vector<cv::Mat> hsv;
      cv::split(temp, hsv);

      // so it can support hue near 0 & 360
      _hue_n = (_hue_n + 360);
      _hue_p = (_hue_p + 360);

      // masking out values that do not fall between the condition 
      cv::Mat wm(color_frame.rows, color_frame.cols, CV_8UC1, cv::Scalar(0));
      for (int r = 0; r < temp.rows; r++)
      {
        uchar* workspace_row = wm.ptr<uchar>(r);
        for (int c = 0; c < temp.cols; c++)
        {
          int hue     = 2*(int)hsv[0].at<uchar>(r, c) + 360;
          float sat   = 0.392*(int)hsv[1].at<uchar>(r, c); // 0.392 = 100/255
          float value = 0.392*(int)hsv[2].at<uchar>(r, c);

          if (_hue_n < hue && hue < _hue_p)
            if (_sat_n < sat && sat < _sat_p)
              if (_value_n < value && value < _value_p)
                workspace_row[c] = 255;
        }
      }

      return wm;
    }

    std::vector<cv::Point> getMomentCoordinates(std::vector<cv::Moments> ms)
    {
      std::vector<cv::Point> ret;
      ret.clear();
      if (ms.size() == 3) { 
        double centroids[3][2];
        for (int i = 0; i < 3; i++) {
          cv::Moments m0 = ms.at(i);
          double x0, y0;
          x0 = m0.m10/m0.m00;
          y0 = m0.m01/m0.m00;
          centroids[i][0] = x0;
          centroids[i][1] = y0;
        }

        // find the top left corner using distance scheme
        cv::Mat vect = cv::Mat::zeros(3,2, CV_32F);
        vect.at<float>(0,0) = centroids[0][0] - centroids[1][0];
        vect.at<float>(0,1) = centroids[0][1] - centroids[1][1];
        vect.at<float>(1,0) = centroids[0][0] - centroids[2][0];
        vect.at<float>(1,1) = centroids[0][1] - centroids[2][1];
        vect.at<float>(2,0) = centroids[1][0] - centroids[2][0];
        vect.at<float>(2,1) = centroids[1][1] - centroids[2][1];

        double angle[3];
        angle[0] = abs(vect.row(0).dot(vect.row(1)));
        angle[1] = abs(vect.row(0).dot(vect.row(2)));
        angle[2] = abs(vect.row(1).dot(vect.row(2)));

        // printMatrix(vect); 
        double min = angle[0]; 
        int one = 0;
        for (int i = 0; i < 3; i++)
        {
          // printf("[%d, %f]\n", i, angle[i]);
          if (angle[i] < min)
          {
            min = angle[i];
            one = i;
          }
        }

        // index of others depending on the index of the origin
        int a = one == 0 ? 1 : 0;
        int b = one == 2 ? 1 : 2; 
        // vectors of origin to a point
        double vX0, vY0, vX1, vY1, result;
        vX0 = centroids[a][0] - centroids[one][0];
        vY0 = centroids[a][1] - centroids[one][1];
        vX1 = centroids[b][0] - centroids[one][0];
        vY1 = centroids[b][1] - centroids[one][1];
        cv::Point pto(centroids[one][0], centroids[one][1]);
        cv::Point pta(centroids[a][0], centroids[a][1]);
        cv::Point ptb(centroids[b][0], centroids[b][1]);

        // cross-product: simplified assuming that z = 0 for both
        result = vX1*vY0 - vX0*vY1;
        ret.push_back(pto);
        if (result >= 0) {
          ret.push_back(ptb);
          ret.push_back(pta);
        }
        else {
          ret.push_back(pta);
          ret.push_back(ptb);
        }
      }
      return ret;
    } 


    std::string getFileName(unsigned int counter, std::string which_arm)
    {
      std::ostringstream os;
      os << "/u/swl33/data/";
      if (counter / 100 == 0)
        os << "0";
      if (counter / 10 == 0)
        os << "0";
      os <<  counter;
      os << which_arm;
      return os.str();
    }

    void get_fk_tooltip_pose(double cur[14])
    {
      PoseStamped p, pl, pr;
      p.pose.orientation.w = 1;

      p.header.frame_id = "/l_gripper_tool_frame";
      tf_->transformPose(workspace_frame_, p, pl);
      cur[0]=pl.pose.position.x;
      cur[1]=pl.pose.position.y;
      cur[2]=pl.pose.position.z;
      cur[3]=pl.pose.orientation.x;
      cur[4]=pl.pose.orientation.y;
      cur[5]=pl.pose.orientation.z;
      cur[6]=pl.pose.orientation.w;

      p.header.frame_id = "/r_gripper_tool_frame";
      tf_->transformPose(workspace_frame_, p, pr);
      cur[7]=pr.pose.position.x;
      cur[8]=pr.pose.position.y;
      cur[9]=pr.pose.position.z;
      cur[10]=pr.pose.orientation.x;
      cur[11]=pr.pose.orientation.y;
      cur[12]=pr.pose.orientation.z;
      cur[13]=pr.pose.orientation.w;
    }

    pcl::PointXYZ printValue1(cv::Mat color_frame)
    {
      cv::Mat tape_mask = colorSegment(color_frame, target_hue_value_, target_hue_threshold_);
      cv::Mat element_b = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
      cv::morphologyEx(tape_mask, tape_mask, cv::MORPH_OPEN, element_b);
      std::vector<cv::Moments> ms = findMoments(tape_mask, color_frame, 1); 

      if (ms.size() > 0)
      {
        cv::Moments m0 = ms.at(0);
        double x0, y0;
        x0 = m0.m10/m0.m00;
        y0 = m0.m01/m0.m00;

        cv::circle(color_frame, cv::Point(x0, y0), 4, cv::Scalar(50, 255, 50), 3);
        ROS_INFO("Tape >> [%f, %f, %f]", cloud_.at(x0, y0).x, cloud_.at(x0, y0).y, cloud_.at(x0, y0).z);
        return cloud_.at(x0,y0);
      }
      return pcl::PointXYZ();
    }

    pcl::PointXYZ printValue2(cv::Mat color_frame)
    {
      try
      {
        double gripper_pose[14];
        get_fk_tooltip_pose(gripper_pose);
        // 0, 1, 2
        ROS_INFO("FK   >> l:[%f, %f, %f]\tr:[%f, %f, %f]",
            gripper_pose[0],gripper_pose[1],gripper_pose[2],
            gripper_pose[7],gripper_pose[8],gripper_pose[9]
            );

        cv::Point lgp = projectPointIntoImage(pcl::PointXYZ(gripper_pose[0], gripper_pose[1], gripper_pose[2]), "/torso_lift_link", "/head_mount_kinect_rgb_optical_frame", tf_);

        cv::circle(color_frame, lgp, 4, cv::Scalar(0, 0, 255), 3);

        return pcl::PointXYZ(gripper_pose[0],gripper_pose[1],gripper_pose[2]);
      }
      catch (tf::TransformException e)
      {
        ROS_ERROR("[vs]%s", e.what());
      }
      return pcl::PointXYZ();
    }

    std::vector<cv::DMatch> matchAndQuery(Data ds, std::vector<cv::KeyPoint> kps, cv::Mat color_frame, pcl::PointXYZ *out)
    {
      // profile
      timespec t1, t2;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);

      cv::SurfDescriptorExtractor surfDes;
      cv::Mat descriptors2;
      surfDes.compute(color_frame, kps, descriptors2);

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
      // printf("M&Q: "); printTime(t1, t2);


      cv::FlannBasedMatcher matcher;
      std::vector<cv::DMatch> matches;
      matcher.match(ds.desc, descriptors2, matches);
      std::vector< cv::DMatch> good_matches;
      good_matches.clear();

      double min_dist = 10000; double max_dist = 0;
      for (unsigned int i = 0; i < matches.size(); i++)
      {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist < max_dist) max_dist = dist;
      }

      for (unsigned int i = 0; i < matches.size(); i++)
      {
        if (matches[i].distance < 10 * min_dist)
        {
          good_matches.push_back(matches[i]);
          // ROS_DEBUG("[Match %u: {%d}->{%d} %.4f]", i, matches[i].queryIdx, matches[i].trainIdx, matches[i].distance);
        }
      }
      tcp_e_ = pcl::PointXYZ(0,0,0);
      if (good_matches.size() >= 3)
      {
        /*
        for (unsigned int i = 0; i < 3; i++)
        {
          int pi = good_matches[i].queryIdx;
          int ci = good_matches[i].trainIdx;
          Distance d = ds.d[pi];
          cv::Point2f cp = kps_[ci].pt;
          pcl::PointXYZ c3d = cloud_.at(cp.x, cp.y);
          if (!isnan(c3d.x)&&!isnan(c3d.y)&&!isnan(c3d.z))
          {
            // ROS_INFO(">>> [%f, %f, %f] + [%f %f %f]", c3d.x, c3d.y, c3d.z, d.x, d.y, d.z);
            tcp_e_.x += c3d.x + d.x;
            tcp_e_.y += c3d.y + d.y;
            tcp_e_.z += c3d.z + d.z;
          }
        }

        // RANSAC1
        double tcp_e2[3];
        RANSAC(ds, matches, good_matches, kps, tcp_e2);
        */

        // RANSAC 2
        double fkpose[14];
        get_fk_tooltip_pose(fkpose);
        double tcp_e3[3];
        for (int j = 0; j < 3; j++)
          tcp_e3[j] = fkpose[j];
        int n = RANSAC2(ds, matches, good_matches, kps, tcp_e3);
        tcp_e_.x = tcp_e3[0];
        tcp_e_.y = tcp_e3[1];
        tcp_e_.z = tcp_e3[2];

        ROS_INFO("EST[%2d]: [%f, %f %f]", n, tcp_e_.x, tcp_e_.y, tcp_e_.z);
        // print
        // ROS_INFO("Est TCP: **[%f, %f, %f] ** [%.4f, %.4f, %.4f] ** [%.4f, %.4f %.4f]", tcp_e3[0], tcp_e3[1], tcp_e3[2], tcp_e2[0],tcp_e2[1],tcp_e2[2], tcp_e_.x, tcp_e_.y, tcp_e_.z);

        /*
        out.x = tcp_e2[0];
        out.y = tcp_e2[1];
        out.z = tcp_e2[2];
        */
        out->x = tcp_e3[0];
        out->y = tcp_e3[1];
        out->z = tcp_e3[2];
      }



      return good_matches;
    }
    void printTime(timespec start, timespec end)
    {
      //ROS_INFO("Time: %.6fms", (end.tv_nsec - start.tv_nsec)/1e6);
      ROS_INFO ("Time: %.6fms\n", (end.tv_sec - start.tv_sec)*1e3 + (end.tv_nsec - start.tv_nsec)/1e6);
    }
    int RANSAC2(Data ds, std::vector<cv::DMatch> m, std::vector<cv::DMatch> gm, std::vector<cv::KeyPoint> kp, double ybar[3]){
      // copy
      double yo[3];
      for (int i = 0; i < 3; i++)
        yo[i] = ybar[i];

      if (gm.size() < 3)
      {
        ROS_WARN("Too little features on the target. Abort");
        return -1;
      }

      int nbar = 0;
      double dssbar =1e6;
      std::vector<int> conbar; conbar.clear();

      // profile
      timespec t1, t2;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);


      for (int rep = 0; rep < 14; rep++)
      {
        double ybart[3];
        // copying values
        for (int i = 0; i < 3; i++)
          ybart[i] = yo[i];
        int n = 0;
        double dss = 0;

        // updates ybar
        std::vector<cv::DMatch> gm2(gm.begin(), gm.begin()+2);
        int res = gradDescent(ds, ybart, kp, gm2);
        if (res == 0) 
        {
          std::vector<int> cons; cons.clear();
          // find the consensus set
          for (unsigned int i = 0; i < gm.size(); i++)
          {
            int pi = gm[i].queryIdx;
            int ci = gm[i].trainIdx;
            pcl::PointXYZ xi_pc = cloud_.at(kp[ci].pt.x, kp[ci].pt.y);
            double xi[3] = {xi_pc.x, xi_pc.y, xi_pc.z};
            double cur_dist = distance(ybart, xi);
            if (!isnan(cur_dist) && fabs(cur_dist - ds.dist[pi]) < ds.dist[pi]*0.30)
            {
              // ROS_DEBUG("RANSAC2 >> curdist[%f] dsdist[%f]", cur_dist, ds.dist[pi]);
              dss += fabs(cur_dist - ds.dist[pi]);
              n++;
              cons.push_back(i);
            }
          }

          ROS_DEBUG("RANSAC2 [@rep=%d][n=%2d] [%f, %f, %f]", rep, n, ybart[0], ybart[1], ybart[2]); 
          if ((n > nbar) || (n == nbar && (dss < dssbar)))
          {
            nbar = n;
            for (int i = 0; i < 3; i++)
              ybar[i] = ybart[i];
            dssbar = dss;
            conbar = cons;
          }
        }
        else if (res < 0)
        {
          // could not find the value
        }
        std::random_shuffle( gm.begin(), gm.end() );
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
      // printf("RANSAC2: "); printTime(t1, t2);


      std::vector<cv::DMatch> gmbar; gmbar.clear();
      for (unsigned int i = 0; i < conbar.size(); i++)
      {
        gmbar.push_back(gm[conbar[i]]);
      }
      gradDescent(ds, ybar, kp, gmbar);
      ROS_DEBUG("RANSAC2 [N=%2d] [%f, %f, %f]", nbar, ybar[0], ybar[1], ybar[2]);
      return nbar;
    }

    void RANSAC(Data ds, std::vector<cv::DMatch> m, std::vector<cv::DMatch> gm, std::vector<cv::KeyPoint> kp, double result[3])
    {
      result[0] = 0.0;
      result[1] = 0.0;
      result[2] = 0.0;
      if (gm.size() < 3)
        return;
      int numBestMatch = 0;
      double temp[3];
      for (unsigned int i = 0; i < gm.size()-2; i++)
      {
        int p0 = gm[0].queryIdx;
        int c0 = gm[0].trainIdx;
        int p1 = gm[1].queryIdx;
        int c1 = gm[1].trainIdx;
        int p2 = gm[2].queryIdx;
        int c2 = gm[2].trainIdx;
        x3Sphere(
            cloud_.at(kp[c0].pt.x,kp[c0].pt.y),
            cloud_.at(kp[c1].pt.x,kp[c1].pt.y),
            cloud_.at(kp[c2].pt.x,kp[c2].pt.y),
            ds.dist[p0],ds.dist[p1],ds.dist[p2], temp);
        if (temp[0] != 0 && temp[1] != 0 && temp[2] != 0)
        {
          int nm = numConsensus(ds, m, kp, temp);
          //ROS_INFO("| %d |Temp: [%f, %f, %f], Res: [%f, %f, %f]", nm, temp[0], temp[1], temp[2], result[0], result[1], result[2]);
          if (nm > numBestMatch)
          {
            // copying
            result[0] = temp[0];
            result[1] = temp[1];
            result[2] = temp[2];
            numBestMatch = nm;
          }
        }
        std::random_shuffle( gm.begin(), gm.end() );
      }
      ROS_DEBUG("RANSAC1 | %d | Res: [%f, %f, %f]", numBestMatch, result[0], result[1], result[2]);
    }

    int numConsensus(Data ds, std::vector<cv::DMatch> matches, std::vector<cv::KeyPoint> kps, double estimate[3])
    {
      int count = 0;
      double thresh = 0.08; // 3 cm threshold
      for (unsigned int i = 0; i < matches.size(); i++)
      {
        int p = matches[i].queryIdx;
        int c = matches[i].trainIdx;
        pcl::PointXYZ pp = cloud_.at(kps[c].pt.x, kps[c].pt.y);
        double distance = fabs(ds.dist[p] - pow(pow(pp.x - estimate[0],2)+pow(pp.y - estimate[1],2)+pow(pp.z - estimate[2],2), 0.5));

        // ROS_WARN("Dist @ %d: %f", i, distance);
        if (distance < thresh)
          count++;
      }
      return count;
    }

    int gradDescent(Data ds, double est[3], std::vector<cv::KeyPoint> kp, std::vector<cv::DMatch> gm)
    {
      // profile
      timespec t1, t2;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);

      for (int trial = 0; trial < 50; trial++)
      {
        double grad[3] = {0, 0, 0};
        unsigned int count = 0;
        // gm is shuffled need only two
        for (unsigned int i = 0; i < gm.size(); i++)
        {
          int pi = gm[i].queryIdx;
          int ci = gm[i].trainIdx;
          pcl::PointXYZ xi_pc = cloud_.at(kp[ci].pt.x, kp[ci].pt.y);
          double xi[3] = {xi_pc.x, xi_pc.y, xi_pc.z};
          double temp = distance(est, xi)- ds.dist[pi];

          // pointcloud can be nan
          if (!isnan(temp))
          {
            grad[0] += (est[0]-xi[0])*temp;
            grad[1] += (est[1]-xi[1])*temp;
            grad[2] += (est[2]-xi[2])*temp;
            count++;
          }
        }
        int exit = 0;
        if (count == 0) return -3;
        for (int i = 0; i < 3; i++)
        {
          double param = 10/count;
          // updating estimate using gradient
          est[i] -= (grad[i]*param);
          // ROS_WARN("%.6f*%f (%lu) =%.6f", grad[i], param, gm.size(), grad[i]*param);
          if (isnan(grad[i])) {exit = -1; break;};
          if (fabs(grad[i]) < 1e-5) exit++;
          else if (grad[i] > 1) {exit = -2; break;};
        }
        if (exit < 0 || exit == 3)
        {
          // ROS_WARN("GradDesc: Finished at [%3d with id: %d] Est: [%f, %f %f]", trial, exit, est[0], est[1], est[2]); 
          clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
          // printf("gradDesc: "); printTime(t1, t2);
          return exit == 3 ? 0 : exit;
        }
      }

      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);
      // printTime(t1, t2);
      return 0;
      //ROS_DEBUG("GradDesc: Never finished Est: [%f, %f %f]", est[0], est[1], est[2]); 
    }

    double distance(pcl::PointXYZ yi, pcl::PointXYZ xi)
    {
      double ys[3] = {yi.x, (double)yi.y, yi.z};
      double xs[3] = {xi.x, xi.y, xi.z};
      return distance(ys, xs);
    }

    double distance(double y[3], double x[3])
    {
      return pow(y[0]-x[0],2)+pow(y[1]-x[1],2)+pow(y[2]-x[2],2);
    }

    void x3Sphere(pcl::PointXYZ X1,pcl::PointXYZ X2,pcl::PointXYZ X3,double r1, double r2, double r3, double result[3])
    {
      double x1, y1, z1;
      double x2, y2, z2;
      double x3, y3, z3;
      x1 = X1.x; y1 = X1.y; z1 = X1.z;
      x2 = X2.x; y2 = X2.y; z2 = X2.z;
      x3 = X3.x; y3 = X3.y; z3 = X3.z;

      if(isnan(x1) || isnan(x2) || isnan(x3))
        return;
      if(isnan(y1) || isnan(y2) || isnan(y3))
        return;
      if(isnan(z1) || isnan(z2) || isnan(z3))
        return;
      if(isnan(r1) || isnan(r2) || isnan(r3))
        return;
      /*
      ROS_WARN("-------------------- ");
      ROS_WARN("||| %f %f %f ||| ", x1, y1, z1);
      ROS_WARN("||| %f %f %f ||| ", x2, y2, z2);
      ROS_WARN("||| %f %f %f ||| ", x3, y3, z3);
      ROS_WARN("||| %f %f %f ||| ", r1, r2, r3);
      ROS_WARN("-------------------- ");
      */

      x2=x2-x1; y2=y2-y1; z2=z2-z1;
      x3=x3-x1; y3=y3-y1; z3=z3-z1;

      double a=(16*pow(y2,2)*z3*pow(y3,2)*z2*x3*pow(r1,2)*x2-4*pow(y2,3)*z3*y3*z2*x3*pow(r1,2)*x2+4*pow(y2,3)*z3*y3*z2*x3*x2*pow(r3,2)-4*y2*pow(y3,3)*z2*x2*z3*pow(r1,2)*x3+4*y2*pow(y3,3)*z2*x2*z3*pow(r2,2)*x3+16*z2*pow(x3,2)*pow(x2,2)*pow(r1,2)*y3*y2*z3-4*z2*pow(x3,3)*x2*pow(r1,2)*y3*y2*z3+4*z2*pow(x3,3)*x2*y2*z3*pow(r2,2)*y3-4*pow(x2,3)*z3*x3*y2*pow(r1,2)*z2*y3+4*pow(x2,3)*z3*x3*y2*pow(r3,2)*z2*y3-4*y2*z3*pow(z2,3)*y3*x3*pow(r1,2)*x2+4*y2*z3*pow(z2,3)*y3*x3*x2*pow(r3,2)+8*y2*pow(z3,2)*pow(z2,2)*y3*x2*pow(r1,2)*x3-4*y2*pow(z3,2)*pow(z2,2)*y3*x2*pow(r2,2)*x3+4*pow(x2,2)*pow(y3,2)*z2*pow(y2,2)*z3*pow(x3,2)-4*pow(x2,4)*pow(z3,2)*pow(x3,2)*y3*y2-2*pow(z2,2)*pow(x2,4)*pow(x2,2)*pow(y2,2)+2*pow(z2,2)*pow(x3,3)*pow(y2,2)*pow(r1,2)*x2-2*pow(z2,2)*pow(x3,3)*pow(y2,2)*x2*pow(r3,2)+2*pow(z2,2)*pow(x3,5)*x2*y2*y3+2*pow(x2,5)*pow(z3,2)*x3*y3*y2+2*pow(z2,3)*pow(y3,2)*pow(y2,2)*z3*pow(x3,2)+2*pow(z2,3)*pow(y3,2)*pow(x2,2)*z3*pow(x3,2)+2*pow(z2,3)*pow(y3,2)*pow(x2,2)*z3*pow(r1,2)-2*pow(z2,3)*pow(y3,2)*pow(x2,2)*z3*pow(r3,2)+2*pow(x2,2)*pow(y3,4)*z2*pow(y2,2)*z3-4*pow(x2,2)*pow(y3,2)*pow(z2,2)*pow(x3,2)*pow(y2,2)+2*pow(x2,4)*pow(y3,2)*z2*z3*pow(x3,2)-2*pow(x2,2)*pow(y3,3)*pow(z2,2)*y2*pow(x3,2)+2*pow(x2,2)*pow(y3,3)*pow(z2,2)*y2*pow(z3,2)+2*pow(x2,3)*pow(y3,2)*pow(z2,2)*x3*pow(z3,2)+2*pow(x2,2)*pow(y3,2)*z2*pow(y2,2)*pow(z3,3)+2*pow(x2,2)*pow(y3,3)*pow(z2,2)*y2*pow(r1,2)+2*pow(x2,2)*pow(y3,2)*pow(z2,2)*pow(x3,2)*pow(r2,2)+2*pow(x2,4)*pow(y3,2)*z2*z3*pow(r1,2)-2*pow(x2,4)*pow(y3,2)*z2*z3*pow(r3,2)-2*pow(x2,2)*pow(y3,3)*pow(z2,2)*y2*pow(r3,2)+2*pow(x2,3)*pow(y3,2)*pow(z2,2)*x3*pow(r1,2)-2*pow(x2,3)*pow(y3,2)*pow(z2,2)*x3*pow(r3,2)+2*pow(y2,4)*z3*pow(x3,2)*pow(y3,2)*z2+2*pow(y2,2)*z3*pow(x2,4)*z2*pow(x2,2)-4*pow(y2,2)*pow(z3,2)*pow(x3,2)*pow(x2,2)*pow(y3,2)+2*pow(y2,3)*pow(z3,2)*pow(x3,2)*pow(z2,2)*y3+2*pow(y2,2)*pow(z3,2)*pow(x3,3)*x2*pow(z2,2)-2*pow(y2,3)*pow(z3,2)*pow(x3,2)*pow(x2,2)*y3+2*pow(y2,3)*pow(z3,2)*pow(x3,2)*pow(r1,2)*y3+2*pow(y2,2)*z3*pow(x2,4)*z2*pow(r1,2)-2*pow(y2,2)*z3*pow(x2,4)*z2*pow(r2,2)+2*pow(y2,2)*pow(z3,2)*pow(x3,2)*pow(x2,2)*pow(r3,2)-2*pow(y2,3)*pow(z3,2)*pow(x3,2)*pow(r2,2)*y3+2*pow(y2,2)*pow(z3,2)*pow(x3,3)*x2*pow(r1,2)-2*pow(y2,2)*pow(z3,2)*pow(x3,3)*x2*pow(r2,2)-2*pow(y2,2)*pow(z3,2)*pow(y3,2)*pow(x2,3)*x3-4*pow(y2,4)*pow(z3,2)*pow(y3,2)*x2*x3+2*pow(y2,2)*pow(z3,2)*pow(y3,2)*pow(x2,2)*pow(r3,2)+4*pow(y2,3)*pow(z3,2)*y3*pow(x2,3)*x3+2*pow(y2,5)*pow(z3,2)*y3*x2*x3+4*y2*pow(y3,3)*pow(z2,2)*pow(x3,3)*x2+2*y2*pow(y3,6)*pow(z2,2)*x3*x2-2*pow(y2,2)*pow(y3,2)*pow(z2,2)*pow(x3,3)*x2-4*pow(y2,2)*pow(y3,4)*pow(z2,2)*x3*x2+2*pow(y2,2)*pow(y3,2)*pow(z2,2)*pow(x3,2)*pow(r2,2)-4*pow(z2,2)*pow(x2,4)*pow(x2,2)*y2*y3+2*z2*pow(x3,2)*pow(x2,2)*pow(y2,2)*pow(z3,3)+2*z2*pow(x3,2)*pow(y2,4)*z3*pow(r1,2)+2*pow(z2,2)*pow(x3,2)*pow(y2,3)*pow(r1,2)*y3-2*z2*pow(x3,2)*pow(y2,4)*z3*pow(r3,2)-2*pow(z2,2)*pow(x3,2)*pow(y2,3)*pow(r3,2)*y3-pow(z2,4)*pow(y3,2)*pow(x3,2)*pow(x2,2)-pow(z2,4)*pow(y3,2)*pow(x3,2)*pow(y2,2)+2*pow(z2,3)*pow(y3,4)*pow(x2,2)*z3+2*pow(z2,3)*pow(y3,2)*pow(x2,2)*pow(z3,3)+2*pow(x2,2)*pow(y3,6)*pow(z2,2)*y2-2*pow(x2,2)*pow(y3,4)*pow(z2,2)*pow(y2,2)-2*pow(x2,4)*pow(y3,2)*pow(z2,2)*pow(x3,2)+2*pow(x2,3)*pow(y3,2)*pow(z2,2)*pow(x3,3)+2*pow(x2,4)*pow(y3,4)*z2*z3+2*pow(x2,3)*pow(y3,4)*pow(z2,2)*x3+2*pow(x2,2)*pow(y3,4)*pow(z2,2)*pow(r2,2)-2*pow(y2,4)*pow(z3,2)*pow(x3,2)*pow(y3,2)+2*pow(y2,5)*pow(z3,2)*pow(x3,2)*y3+2*pow(y2,4)*z3*pow(x2,4)*z2+2*pow(y2,2)*pow(z3,2)*pow(x3,3)*pow(x2,3)-2*pow(y2,2)*pow(z3,2)*pow(x2,4)*pow(x2,2)+2*pow(y2,4)*pow(z3,2)*pow(x3,3)*x2+2*pow(y2,2)*z3*pow(x2,4)*pow(z2,3)-pow(y2,2)*pow(z3,4)*pow(x3,2)*pow(x2,2)+2*pow(y2,4)*pow(z3,2)*pow(x3,2)*pow(r3,2)-2*pow(y2,2)*pow(z3,2)*pow(y3,4)*pow(x2,2)+2*pow(y2,3)*pow(z3,2)*pow(y3,3)*pow(x2,2)-pow(y2,2)*pow(z3,4)*pow(y3,2)*pow(x2,2)-pow(y2,4)*pow(z3,2)*pow(y3,2)*pow(x2,2)+2*pow(y2,3)*pow(y3,3)*pow(z2,2)*pow(x3,2)-pow(y2,2)*pow(y3,4)*pow(z2,2)*pow(x3,2)-2*pow(y2,4)*pow(y3,2)*pow(z2,2)*pow(x3,2)-pow(z2,4)*pow(y3,4)*pow(x2,2)-2*pow(x2,4)*pow(y3,4)*pow(z2,2)-2*pow(y2,4)*pow(z3,2)*pow(x2,4)-pow(y2,4)*pow(z3,4)*pow(x3,2)-2*pow(z2,2)*pow(x2,4)*pow(y2,4)-pow(z2,4)*pow(x2,4)*pow(y2,2)-2*pow(x2,4)*pow(z3,2)*pow(y3,4)-pow(x2,4)*pow(z3,4)*pow(y3,2)-pow(z3,2)*pow(y2,6)*pow(x3,2)-pow(x3,6)*pow(z2,2)*pow(y2,2)-pow(z3,2)*pow(x2,6)*pow(y3,2)-2*pow(y2,4)*pow(x2,4)*pow(y3,2)+2*pow(y2,4)*pow(x2,4)*pow(r3,2)+2*pow(y2,5)*pow(x2,4)*y3-pow(y2,4)*pow(x3,2)*pow(r1,4)-pow(y2,4)*pow(x3,2)*pow(y3,4)+2*pow(y2,5)*pow(x3,2)*pow(y3,3)-pow(y2,4)*pow(x3,2)*pow(r3,4)-pow(y2,6)*pow(x3,2)*pow(y3,2)-pow(y2,2)*pow(x2,4)*pow(r1,4)-pow(y2,2)*pow(x2,4)*pow(x2,4)+2*pow(y2,2)*pow(x3,5)*pow(x2,3)-pow(y2,2)*pow(x2,4)*pow(r2,4)-pow(y2,2)*pow(x3,6)*pow(x2,2)-2*pow(y2,4)*pow(x2,4)*pow(x2,2)+2*pow(y2,4)*pow(x2,4)*pow(r2,2)+2*pow(y2,4)*pow(x3,5)*x2+2*pow(x2,4)*pow(y3,6)*y2-2*pow(x2,4)*pow(y3,4)*pow(y2,2)+2*pow(x2,4)*pow(y3,4)*pow(r2,2)-pow(x2,2)*pow(y3,4)*pow(r1,4)-pow(x2,2)*pow(y3,6)*pow(y2,2)+2*pow(x2,2)*pow(y3,6)*pow(y2,3)-pow(x2,2)*pow(y3,4)*pow(y2,4)-pow(x2,2)*pow(y3,4)*pow(r2,4)-pow(x2,4)*pow(y3,2)*pow(r1,4)-pow(x2,6)*pow(y3,2)*pow(x3,2)+2*pow(x2,5)*pow(y3,2)*pow(x3,3)-pow(x2,4)*pow(y3,2)*pow(x2,4)-pow(x2,4)*pow(y3,2)*pow(r3,4)+2*pow(x2,5)*pow(y3,4)*x3-2*pow(x2,4)*pow(y3,4)*pow(x3,2)+2*pow(x2,4)*pow(y3,4)*pow(r3,2)-pow(y3,6)*pow(z2,2)*pow(x2,2)-pow(y2,4)*pow(x3,6)-pow(y2,6)*pow(x2,4)-pow(x2,6)*pow(y3,4)-pow(x2,4)*pow(y3,6)-2*z2*pow(x3,2)*pow(r1,2)*pow(y2,2)*z3*pow(r3,2)-2*z2*pow(y3,2)*pow(r2,2)*pow(x2,2)*z3*pow(r1,2)+2*z2*pow(y3,2)*pow(r2,2)*pow(x2,2)*z3*pow(r3,2)+2*pow(x2,4)*pow(y3,2)*z2*pow(z3,3)+2*y2*pow(r1,4)*pow(z2,2)*y3*x3*x2+2*pow(x2,2)*pow(y3,2)*z2*pow(y2,2)*z3*pow(r1,2)-8*pow(x2,2)*pow(y3,3)*z2*pow(r1,2)*y2*z3-2*pow(x2,2)*pow(y3,2)*z2*pow(y2,2)*z3*pow(r3,2)-8*pow(x2,3)*pow(y3,2)*z2*z3*pow(r1,2)*x3+2*pow(y2,2)*z3*pow(x3,2)*pow(r1,2)*pow(y3,2)*z2-8*pow(y2,3)*z3*pow(x3,2)*pow(r1,2)*z2*y3-2*pow(y2,2)*z3*pow(x3,2)*z2*pow(y3,2)*pow(r2,2)-8*pow(y2,2)*z3*pow(x3,3)*z2*pow(r1,2)*x2-4*pow(y2,2)*pow(z3,2)*pow(y3,2)*x2*pow(z2,2)*x3-4*pow(y2,2)*pow(z3,2)*pow(y3,2)*x2*pow(r1,2)*x3+4*pow(y2,2)*pow(z3,2)*pow(y3,2)*x2*pow(r2,2)*x3-4*pow(y2,3)*z3*y3*z2*pow(x3,3)*x2-4*pow(y2,3)*z3*pow(y3,3)*z2*x3*x2-4*pow(y2,3)*pow(z3,3)*y3*z2*x3*x2+4*pow(y2,3)*pow(z3,2)*y3*x2*pow(z2,2)*x3-4*pow(y2,3)*pow(z3,2)*y3*x2*pow(r2,2)*x3-4*y2*pow(y3,3)*z2*pow(x2,3)*z3*x3+4*y2*pow(y3,3)*pow(z2,2)*x3*x2*pow(z3,2)-4*y2*pow(y3,3)*pow(z2,3)*x2*z3*x3-4*y2*pow(y3,3)*pow(z2,2)*x3*x2*pow(r3,2)-4*pow(y2,2)*pow(y3,2)*pow(z2,2)*x3*pow(r1,2)*x2+4*pow(y2,2)*pow(y3,2)*pow(z2,2)*x3*x2*pow(r3,2)-4*pow(z2,2)*pow(x3,2)*pow(x2,2)*y2*pow(z3,2)*y3+2*z2*pow(x3,2)*pow(x2,2)*pow(y2,2)*z3*pow(r1,2)-4*pow(z2,2)*pow(x3,2)*pow(x2,2)*y2*pow(r1,2)*y3-2*z2*pow(x3,2)*pow(x2,2)*pow(y2,2)*z3*pow(r3,2)+4*pow(z2,2)*pow(x3,2)*pow(x2,2)*y2*pow(r3,2)*y3-4*pow(z2,3)*pow(x3,3)*x2*y2*z3*y3+4*pow(z2,2)*pow(x3,3)*x2*y2*pow(z3,2)*y3-4*z2*pow(x3,3)*pow(x2,3)*y3*y2*z3-4*pow(z2,2)*pow(x3,3)*x2*y2*pow(r3,2)*y3+4*pow(x2,3)*pow(z3,2)*x3*y2*pow(z2,2)*y3-4*pow(x2,3)*pow(z3,3)*x3*y2*z2*y3-4*pow(x2,3)*pow(z3,2)*x3*y2*pow(r2,2)*y3+2*pow(x2,2)*z3*pow(x3,2)*pow(r1,2)*pow(y3,2)*z2-4*pow(x2,2)*pow(z3,2)*pow(x3,2)*pow(r1,2)*y3*y2-2*pow(x2,2)*z3*pow(x3,2)*z2*pow(y3,2)*pow(r2,2)+4*pow(x2,2)*pow(z3,2)*pow(x3,2)*y2*pow(r2,2)*y3-4*y2*pow(z3,3)*pow(z2,3)*y3*x3*x2+2*y2*pow(z3,2)*pow(z2,4)*y3*x2*x3+2*y2*pow(z3,4)*pow(z2,2)*y3*x3*x2-2*pow(r1,2)*pow(y3,2)*z2*pow(x2,2)*z3*pow(r3,2)-2*pow(y2,2)*z3*pow(r1,2)*z2*pow(x3,2)*pow(r2,2)+2*pow(r1,4)*y3*y2*pow(z3,2)*x2*x3+2*pow(z2,2)*pow(x3,5)*pow(y2,2)*x2+2*pow(z2,2)*pow(x2,4)*pow(y2,3)*y3+2*z2*pow(x3,2)*pow(y2,4)*pow(z3,3)+2*pow(z2,2)*pow(x2,4)*pow(y2,2)*pow(r2,2)-pow(z2,2)*pow(x2,4)*pow(x2,2)*pow(y3,2)+2*pow(x2,5)*pow(z3,2)*x3*pow(y3,2)-pow(x2,4)*pow(z3,2)*pow(x3,2)*pow(y2,2)-2*pow(x2,4)*pow(z3,2)*pow(x3,2)*pow(y3,2)+2*pow(x2,4)*pow(z3,2)*pow(y3,3)*y2+2*pow(x2,4)*pow(z3,2)*pow(y3,2)*pow(r3,2)-2*pow(y2,2)*pow(x2,4)*pow(z2,2)*pow(y3,2)-2*pow(z2,2)*pow(x3,2)*pow(x2,2)*pow(y3,4)-2*pow(x2,2)*pow(z3,2)*pow(y2,4)*pow(x3,2)-2*pow(x2,4)*pow(y3,2)*pow(y2,2)*pow(z3,2)+2*pow(y2,2)*pow(z3,3)*pow(z2,3)*pow(x3,2)-pow(z3,2)*pow(y2,2)*pow(r1,4)*pow(x3,2)-pow(z3,2)*pow(y2,2)*pow(z2,4)*pow(x3,2)-pow(z3,2)*pow(y2,2)*pow(r2,4)*pow(x3,2)-2*pow(z3,2)*pow(y2,4)*pow(x3,2)*pow(z2,2)+2*pow(z3,2)*pow(y2,4)*pow(x3,2)*pow(r2,2)-2*pow(x2,4)*pow(z2,2)*pow(y2,2)*pow(z3,2)+2*pow(x2,4)*pow(z2,2)*pow(y2,2)*pow(r3,2)-pow(x3,2)*pow(z2,2)*pow(y2,2)*pow(r1,4)-pow(x3,2)*pow(z2,2)*pow(y2,2)*pow(z3,4)-pow(x3,2)*pow(z2,2)*pow(y2,2)*pow(r3,4)-2*pow(z3,2)*pow(x2,4)*pow(y3,2)*pow(z2,2)+2*pow(z3,2)*pow(x2,4)*pow(y3,2)*pow(r2,2)-pow(z3,2)*pow(x2,2)*pow(r1,4)*pow(y3,2)-pow(z3,2)*pow(x2,2)*pow(z2,4)*pow(y3,2)-pow(z3,2)*pow(x2,2)*pow(r2,4)*pow(y3,2)+2*pow(y2,3)*pow(x2,4)*pow(r1,2)*y3-2*pow(y2,3)*pow(x2,4)*pow(x2,2)*y3-2*pow(y2,3)*pow(x2,4)*pow(r2,2)*y3+2*pow(y2,4)*pow(x3,3)*pow(r1,2)*x2-2*pow(y2,4)*pow(x3,3)*x2*pow(y3,2)-2*pow(z2,2)*pow(x3,2)*pow(x2,2)*pow(y3,2)*pow(z3,2)+2*pow(z2,2)*pow(x3,2)*pow(x2,2)*pow(y3,2)*pow(r3,2)-2*pow(x2,2)*pow(z3,2)*pow(y2,2)*pow(x3,2)*pow(z2,2)+2*pow(x2,2)*pow(z3,2)*pow(y2,2)*pow(x3,2)*pow(r2,2)+2*pow(x2,2)*pow(y3,2)*pow(y2,2)*pow(z3,2)*pow(r2,2)+2*pow(y2,2)*pow(z3,3)*z2*pow(x3,2)*pow(r1,2)-2*pow(y2,2)*pow(z3,3)*z2*pow(x3,2)*pow(r2,2)+2*pow(z2,3)*pow(x3,2)*pow(y2,2)*z3*pow(r1,2)-2*pow(z2,3)*pow(x3,2)*pow(y2,2)*z3*pow(r3,2)+2*pow(x2,2)*pow(z3,3)*pow(r1,2)*pow(y3,2)*z2-2*pow(x2,2)*pow(z3,3)*z2*pow(y3,2)*pow(r2,2)+2*pow(r1,4)*pow(y3,2)*z2*pow(x2,2)*z3+2*pow(y2,2)*z3*pow(r1,4)*z2*pow(x3,2)+2*pow(x2,2)*z3*pow(y3,4)*pow(r1,2)*z2+2*pow(x2,2)*pow(z3,2)*pow(y3,3)*pow(r1,2)*y2-2*pow(x2,2)*z3*pow(y3,4)*z2*pow(r2,2)-2*pow(x2,2)*pow(z3,2)*pow(y3,3)*y2*pow(r2,2)+2*pow(x2,3)*pow(z3,2)*pow(y3,2)*pow(r1,2)*x3-2*pow(x2,3)*pow(z3,2)*pow(y3,2)*pow(r2,2)*x3-2*pow(y2,2)*pow(z3,2)*pow(z2,2)*pow(y3,2)*pow(x2,2)-2*pow(y2,2)*pow(x3,2)*pow(z2,2)*pow(y3,2)*pow(z3,2)+2*pow(y2,2)*pow(x3,2)*pow(z2,2)*pow(y3,2)*pow(r3,2)-4*y2*pow(z3,2)*pow(z2,2)*y3*x3*x2*pow(r3,2)-4*y2*pow(z3,3)*z2*y3*x2*pow(r1,2)*x3+4*y2*pow(z3,3)*z2*y3*x2*pow(r2,2)*x3-4*pow(r1,4)*y3*y2*z3*z2*x3*x2-4*pow(r1,2)*y3*y2*pow(z3,2)*x2*pow(r2,2)*x3-4*y2*pow(r1,2)*pow(z2,2)*y3*x3*x2*pow(r3,2)+4*pow(r1,2)*y3*y2*z3*z2*x3*x2*pow(r3,2)+4*y2*pow(r1,2)*z2*y3*x2*z3*pow(r2,2)*x3+2*z2*pow(x3,2)*pow(r2,2)*pow(y2,2)*z3*pow(r3,2)+2*y2*pow(z3,2)*pow(r2,4)*y3*x2*x3+2*y2*pow(r3,4)*pow(z2,2)*y3*x3*x2+4*pow(y2,2)*x3*x2*pow(y3,2)*pow(r1,2)*pow(r3,2)+4*pow(y2,2)*x3*x2*pow(y3,2)*pow(r1,2)*pow(r2,2)-4*pow(y2,2)*x3*x2*pow(y3,2)*pow(r3,2)*pow(r2,2)+4*y2*pow(x3,2)*pow(x2,2)*y3*pow(r1,2)*pow(r2,2)+4*y2*pow(x3,2)*pow(x2,2)*y3*pow(r1,2)*pow(r3,2)-4*y2*pow(x3,2)*pow(x2,2)*y3*pow(r2,2)*pow(r3,2)-4*pow(y2,3)*x3*x2*y3*pow(z3,2)*pow(r3,2)-4*y2*x3*x2*pow(y3,3)*pow(r1,2)*pow(r2,2)-4*pow(y2,3)*x3*x2*y3*pow(r1,2)*pow(r3,2)-4*y2*x3*x2*pow(y3,3)*pow(z2,2)*pow(r2,2)-4*y2*x3*pow(x2,3)*y3*pow(r1,2)*pow(r3,2)-4*y2*pow(x3,3)*x2*y3*pow(r1,2)*pow(r2,2)-4*y2*pow(x3,3)*x2*y3*pow(z2,2)*pow(r2,2)-4*y2*x3*pow(x2,3)*y3*pow(z3,2)*pow(r3,2)-4*pow(z3,2)*pow(y2,2)*pow(r1,2)*pow(x3,2)*pow(z2,2)+2*pow(z3,2)*pow(y2,2)*pow(r1,2)*pow(x3,2)*pow(r2,2)+2*pow(z3,2)*pow(y2,2)*pow(z2,2)*pow(x3,2)*pow(r2,2)+2*pow(x3,2)*pow(z2,2)*pow(y2,2)*pow(z3,2)*pow(r3,2)+2*pow(x3,2)*pow(z2,2)*pow(y2,2)*pow(r1,2)*pow(r3,2)-4*pow(z3,2)*pow(x2,2)*pow(r1,2)*pow(y3,2)*pow(z2,2)+2*pow(z3,2)*pow(x2,2)*pow(r1,2)*pow(y3,2)*pow(r2,2)+2*pow(z3,2)*pow(x2,2)*pow(z2,2)*pow(y3,2)*pow(r2,2)-2*pow(y2,3)*pow(x3,2)*pow(r1,2)*y3*pow(r3,2)-2*pow(y2,3)*pow(x3,2)*pow(x2,2)*y3*pow(r1,2)+2*pow(y2,3)*pow(x3,2)*pow(x2,2)*y3*pow(r3,2)-2*pow(y2,3)*pow(x3,2)*pow(r1,2)*pow(r2,2)*y3+2*pow(y2,3)*pow(x3,2)*pow(r3,2)*pow(r2,2)*y3-2*pow(y2,2)*pow(x3,3)*pow(r1,2)*x2*pow(r2,2)-2*pow(y2,2)*pow(x3,3)*pow(r1,2)*x2*pow(y3,2)-2*pow(y2,2)*pow(x3,3)*pow(r1,2)*x2*pow(r3,2)+2*pow(y2,2)*pow(x3,3)*pow(r2,2)*x2*pow(y3,2)+2*pow(y2,2)*pow(x3,3)*pow(r2,2)*x2*pow(r3,2)+2*pow(y2,2)*pow(x3,2)*pow(r1,2)*pow(y3,2)*pow(r2,2)+4*pow(y2,2)*pow(x3,2)*pow(x2,2)*pow(y3,2)*pow(r2,2)+2*pow(y2,2)*pow(x3,2)*pow(r1,2)*pow(x2,2)*pow(r3,2)+4*pow(y2,2)*pow(x3,2)*pow(x2,2)*pow(y3,2)*pow(r3,2)-8*pow(y2,3)*pow(x3,3)*pow(r1,2)*x2*y3-2*pow(x2,2)*pow(y3,3)*pow(r1,2)*y2*pow(x3,2)-2*pow(x2,2)*pow(y3,3)*pow(r1,2)*y2*pow(r3,2)-2*pow(x2,2)*pow(y3,3)*y2*pow(r1,2)*pow(r2,2)+2*pow(x2,2)*pow(y3,3)*y2*pow(x3,2)*pow(r2,2)+2*pow(x2,2)*pow(y3,3)*y2*pow(r3,2)*pow(r2,2)-2*pow(x2,3)*pow(y3,2)*pow(r1,2)*pow(y2,2)*x3-2*pow(x2,3)*pow(y3,2)*pow(r1,2)*pow(r2,2)*x3-2*pow(x2,3)*pow(y3,2)*pow(r1,2)*x3*pow(r3,2)+2*pow(x2,3)*pow(y3,2)*pow(y2,2)*x3*pow(r3,2)+2*pow(x2,3)*pow(y3,2)*pow(r2,2)*x3*pow(r3,2)+2*pow(x2,2)*pow(y3,2)*pow(y2,2)*pow(r1,2)*pow(r3,2)-4*y2*z3*pow(r2,2)*y3*z2*x3*x2*pow(r3,2)-4*pow(x2,4)*pow(y3,2)*pow(r1,2)*pow(x3,2)+2*pow(x2,4)*pow(y3,2)*pow(r1,2)*pow(r3,2)+2*pow(x2,3)*pow(y3,2)*pow(r1,2)*pow(x3,3)+2*pow(x2,4)*pow(y3,2)*pow(x3,2)*pow(r2,2)-2*pow(x2,5)*pow(y3,2)*x3*pow(r3,2)-2*pow(x2,3)*pow(y3,2)*pow(r2,2)*pow(x3,3)+2*pow(x2,4)*pow(y3,2)*pow(x3,2)*pow(r3,2)-pow(y3,2)*pow(z2,2)*pow(r1,4)*pow(x2,2)-pow(y3,2)*pow(z2,2)*pow(x2,2)*pow(z3,4)-pow(y3,2)*pow(z2,2)*pow(x2,2)*pow(r3,4)-2*pow(y3,4)*pow(z2,2)*pow(x2,2)*pow(z3,2)+2*pow(y3,4)*pow(z2,2)*pow(x2,2)*pow(r3,2)+4*pow(y2,3)*x3*pow(x2,3)*pow(y3,3)+4*pow(y2,3)*pow(x3,3)*x2*pow(y3,3)+2*y2*x3*pow(x2,5)*pow(y3,3)+2*pow(y2,3)*pow(x3,5)*x2*y3+2*pow(y2,3)*x3*x2*pow(y3,6)-4*pow(y2,4)*x3*x2*pow(y3,4)+2*pow(y2,5)*x3*x2*pow(y3,3)+2*y2*pow(x3,3)*pow(x2,5)*y3-4*y2*pow(x2,4)*pow(x2,4)*y3+2*pow(y2,5)*pow(x3,3)*x2*y3+2*y2*pow(x3,5)*pow(x2,3)*y3+2*y2*x3*pow(x2,3)*pow(y3,6)+4*pow(y2,3)*pow(x3,3)*pow(x2,3)*y3+4*y2*pow(x3,3)*pow(x2,3)*pow(y3,3)-2*pow(y2,4)*pow(x3,3)*x2*pow(r3,2)-2*pow(y2,5)*pow(x3,2)*pow(r3,2)*y3+2*pow(y2,4)*pow(x3,2)*pow(y3,2)*pow(r2,2)+2*pow(y2,3)*pow(x3,2)*pow(r1,2)*pow(y3,3)+2*pow(y2,3)*pow(x3,2)*pow(r1,4)*y3-4*pow(y2,4)*pow(x3,2)*pow(r1,2)*pow(y3,2)-3*pow(y2,4)*pow(x3,2)*pow(x2,2)*pow(y3,2)+2*pow(y2,4)*pow(x3,2)*pow(r1,2)*pow(r3,2)+2*pow(y2,5)*pow(x3,2)*pow(r1,2)*y3+2*pow(y2,4)*pow(x3,2)*pow(y3,2)*pow(r3,2)-2*pow(y2,3)*pow(x3,2)*pow(y3,3)*pow(r2,2)-pow(y2,2)*pow(x3,2)*pow(r1,4)*pow(y3,2)-3*pow(y2,2)*pow(x3,2)*pow(x2,4)*pow(y3,2)-pow(y2,2)*pow(x3,2)*pow(r2,4)*pow(y3,2)-pow(y2,2)*pow(x3,2)*pow(r1,4)*pow(x2,2)-3*pow(y2,2)*pow(x3,2)*pow(x2,2)*pow(y3,4)-pow(y2,2)*pow(x3,2)*pow(x2,2)*pow(r3,4)+2*pow(y2,2)*pow(x3,3)*pow(r1,4)*x2+2*pow(y2,2)*pow(x3,3)*pow(r1,2)*pow(x2,3)-4*pow(y2,2)*pow(x2,4)*pow(r1,2)*pow(x2,2)+2*pow(y2,2)*pow(x2,4)*pow(r1,2)*pow(r2,2)+2*pow(y2,2)*pow(x3,5)*pow(r1,2)*x2+2*pow(y2,2)*pow(x2,4)*pow(x2,2)*pow(r2,2)-2*pow(y2,2)*pow(x3,3)*pow(x2,3)*pow(r3,2)-2*pow(y2,2)*pow(x3,5)*pow(r2,2)*x2-3*pow(y2,2)*pow(x2,4)*pow(x2,2)*pow(y3,2)+2*pow(y2,2)*pow(x2,4)*pow(x2,2)*pow(r3,2)+2*pow(x2,4)*pow(y3,3)*y2*pow(r1,2)-2*pow(x2,4)*pow(y3,3)*y2*pow(x3,2)-2*pow(x2,4)*pow(y3,3)*y2*pow(r3,2)+2*pow(x2,3)*pow(y3,4)*pow(r1,2)*x3-2*pow(x2,3)*pow(y3,4)*pow(y2,2)*x3-2*pow(x2,3)*pow(y3,4)*pow(r2,2)*x3-2*pow(x2,2)*pow(y3,3)*pow(y2,3)*pow(r3,2)+2*pow(x2,2)*pow(y3,4)*pow(y2,2)*pow(r2,2)+2*pow(x2,2)*pow(y3,6)*pow(r1,2)*y2+2*pow(x2,2)*pow(y3,3)*pow(r1,4)*y2-4*pow(x2,2)*pow(y3,4)*pow(r1,2)*pow(y2,2)+2*pow(x2,2)*pow(y3,4)*pow(r1,2)*pow(r2,2)+2*pow(x2,2)*pow(y3,3)*pow(y2,3)*pow(r1,2)+2*pow(x2,2)*pow(y3,4)*pow(y2,2)*pow(r3,2)-2*pow(x2,2)*pow(y3,6)*y2*pow(r2,2)-pow(x2,2)*pow(y3,2)*pow(y2,2)*pow(r1,4)-pow(x2,2)*pow(y3,2)*pow(y2,2)*pow(r3,4)-pow(x2,2)*pow(y3,2)*pow(r1,4)*pow(x3,2)-pow(x2,2)*pow(y3,2)*pow(r2,4)*pow(x3,2)+2*pow(x2,3)*pow(y3,2)*pow(r1,4)*x3+2*pow(x2,5)*pow(y3,2)*pow(r1,2)*x3+2*pow(x2,2)*pow(y3,2)*pow(r1,2)*pow(x3,2)*pow(r2,2)-8*pow(x2,3)*pow(y3,3)*pow(r1,2)*y2*x3+2*pow(y3,2)*pow(z2,2)*pow(r1,2)*pow(x2,2)*pow(r3,2)+2*pow(y3,2)*pow(z2,2)*pow(x2,2)*pow(z3,2)*pow(r3,2)+4*pow(y2,4)*x3*x2*pow(y3,2)*pow(r3,2)+4*pow(y2,3)*x3*x2*pow(y3,3)*pow(z2,2)-4*pow(y2,3)*x3*x2*pow(y3,3)*pow(r2,2)-4*pow(y2,2)*x3*x2*pow(y3,4)*pow(r1,2)-4*pow(y2,2)*x3*x2*pow(y3,2)*pow(r1,4)+8*pow(y2,3)*x3*x2*pow(y3,3)*pow(r1,2)+4*y2*x3*pow(x2,3)*pow(y3,3)*pow(z2,2)-4*y2*x3*pow(x2,3)*pow(y3,3)*pow(r2,2)-4*pow(y2,4)*x3*x2*pow(y3,2)*pow(r1,2)+4*pow(y2,3)*pow(x3,3)*x2*y3*pow(z3,2)-4*pow(y2,3)*pow(x3,3)*x2*y3*pow(r3,2)+4*pow(y2,3)*x3*x2*pow(y3,3)*pow(z3,2)-4*pow(y2,3)*x3*x2*pow(y3,3)*pow(r3,2)+4*pow(y2,2)*x3*x2*pow(y3,4)*pow(r2,2)+2*y2*x3*x2*pow(y3,3)*pow(r1,4)+2*pow(y2,3)*x3*x2*y3*pow(r1,4)+2*pow(y2,3)*x3*x2*y3*pow(z3,4)+2*pow(y2,3)*x3*x2*y3*pow(r3,4)+2*y2*x3*x2*pow(y3,3)*pow(z2,4)+2*y2*x3*x2*pow(y3,3)*pow(r2,4)+2*y2*x3*pow(x2,3)*y3*pow(r1,4)+2*y2*pow(x3,3)*x2*y3*pow(r1,4)+2*y2*pow(x3,3)*x2*y3*pow(z2,4)+2*y2*pow(x3,3)*x2*y3*pow(r2,4)+2*y2*x3*pow(x2,3)*y3*pow(z3,4)+2*y2*x3*pow(x2,3)*y3*pow(r3,4)-4*y2*pow(x3,2)*pow(x2,2)*y3*pow(r1,4)-4*y2*pow(x3,2)*pow(x2,4)*y3*pow(r1,2)+8*y2*pow(x3,3)*pow(x2,3)*y3*pow(r1,2)-4*y2*pow(x2,4)*pow(x2,2)*y3*pow(r1,2)+4*y2*pow(x3,3)*pow(x2,3)*y3*pow(z2,2)-4*y2*pow(x3,3)*pow(x2,3)*y3*pow(r2,2)+4*y2*pow(x3,2)*pow(x2,4)*y3*pow(r3,2)+4*pow(y2,3)*pow(x3,3)*x2*y3*pow(z2,2)-4*pow(y2,3)*pow(x3,3)*x2*y3*pow(r2,2)+4*y2*pow(x2,4)*pow(x2,2)*y3*pow(r2,2)+4*y2*pow(x3,3)*pow(x2,3)*y3*pow(z3,2)-4*y2*pow(x3,3)*pow(x2,3)*y3*pow(r3,2)+4*y2*x3*pow(x2,3)*pow(y3,3)*pow(z3,2)-4*y2*x3*pow(x2,3)*pow(y3,3)*pow(r3,2)+16*pow(y2,2)*pow(x3,2)*pow(x2,2)*pow(y3,2)*pow(r1,2));
      double b=(-pow(z2,3)*pow(y3,2)-pow(x2,2)*pow(y3,2)*z2-pow(y2,2)*z3*pow(x3,2)-pow(y2,2)*z3*pow(y3,2)+pow(y2,3)*z3*y3+y2*pow(y3,3)*z2-pow(y2,2)*pow(y3,2)*z2-z2*pow(x3,2)*pow(x2,2)-z2*pow(x3,2)*pow(y2,2)+z2*pow(x3,3)*x2+pow(x2,3)*z3*x3-pow(x2,2)*z3*pow(x3,2)-pow(x2,2)*z3*pow(y3,2)+y2*z3*pow(z2,2)*y3+y2*pow(x3,2)*z2*y3+y2*pow(z3,2)*z2*y3+z2*x3*x2*pow(y3,2)+z2*x3*x2*pow(z3,2)+x2*z3*pow(y2,2)*x3+x2*z3*pow(z2,2)*x3+pow(x2,2)*y3*y2*z3-pow(y2,2)*pow(z3,3)-pow(z2,3)*pow(x3,2)-pow(x2,2)*pow(z3,3)-pow(r1,2)*pow(y3,2)*z2-pow(y2,2)*z3*pow(r1,2)+pow(r1,2)*y3*y2*z3+y2*pow(r1,2)*z2*y3+z2*pow(y3,2)*pow(r2,2)-z2*pow(x3,2)*pow(r1,2)+z2*pow(x3,2)*pow(r2,2)-pow(x2,2)*z3*pow(r1,2)+pow(x2,2)*z3*pow(r3,2)+pow(y2,2)*z3*pow(r3,2)-y2*z3*pow(r2,2)*y3-y2*pow(r3,2)*z2*y3+z2*x3*pow(r1,2)*x2-z2*x3*x2*pow(r3,2)+x2*z3*pow(r1,2)*x3-x2*z3*pow(r2,2)*x3);
      double c=(-2*y2*z3*z2*y3-2*z2*x3*x2*z3+pow(z3,2)*pow(y2,2)+pow(x3,2)*pow(z2,2)+pow(z3,2)*pow(x2,2)+pow(y2,2)*pow(x3,2)+pow(x2,2)*pow(y3,2)+pow(y3,2)*pow(z2,2)-2*y2*x3*x2*y3);

      double x,y,z,za,zb;
      if(a<0||c==0) 
        return;
      za=-1/2*(b-pow(a,(1/2)))/c;
      zb=-1/2*(b+pow(a,(1/2)))/c;
      if(za>zb)
        z=zb;
      else
        z=za;
      a=(2*z*z2*x3-2*x2*z*z3+pow(r1,2)*x2-pow(r1,2)*x3-pow(x2,2)*x3-pow(y2,2)*x3-pow(z2,2)*x3+pow(r2,2)*x3+x2*pow(x3,2)+x2*pow(y3,2)+x2*pow(z3,2)-x2*pow(r3,2));
      b=(-2*y2*x3+2*x2*y3);
      y=a/b;
      x = 1/2*(pow(r1,2)+pow(x2,2)-2*y*y2+pow(y2,2)-2*z*z2+pow(z2,2)-pow(r2,2))/x2;

      result[0] = x1 + x;
      result[1] = y1 + y;
      result[2] = z1 + z;

      // ROS_WARN("||| %f %f %f ||| ", result[0], result[1], result[2]);
      return;
    }

    //figure out where the arm is now  
    void get_current_joint_angles(double current_angles[14]){
      int i;
      //get a single message from the topic 'l_arm_controller/state'
      pr2_controllers_msgs::JointTrajectoryControllerStateConstPtr l_state_msg = ros::topic::waitForMessage<pr2_controllers_msgs::JointTrajectoryControllerState>("l_arm_controller/state");
      //extract the joint angles from it
      for(i=0; i<7; i++){
        current_angles[i] = l_state_msg->actual.positions[i];
      }
      //get a single message from the topic 'r_arm_controller/state'
      pr2_controllers_msgs::JointTrajectoryControllerStateConstPtr r_state_msg = ros::topic::waitForMessage<pr2_controllers_msgs::JointTrajectoryControllerState>("r_arm_controller/state");
      for(i=7; i<14; i++){
        current_angles[i] = r_state_msg->actual.positions[i-7];
      }
    }

    cv::Point projectPointIntoImage(pcl::PointXYZ cur_point_pcl,
        std::string point_frame, std::string target_frame, shared_ptr<tf::TransformListener> tf_)
    {
      PointStamped cur_point;
      cur_point.header.frame_id = point_frame;
      cur_point.point.x = cur_point_pcl.x;
      cur_point.point.y = cur_point_pcl.y;
      cur_point.point.z = cur_point_pcl.z;
      return projectPointIntoImage(cur_point, target_frame, tf_);
    }
    cv::Point projectPointIntoImage(PointStamped cur_point,
        std::string target_frame, shared_ptr<tf::TransformListener> tf_)
    {
      if (K.rows == 0 || K.cols == 0 || K.at<float>(0,0) == 0) {
        // Camera intrinsic matrix
        K  = cv::Mat(cv::Size(3,3), CV_64F, &(cam_info_.K));
        K.convertTo(K, CV_32F);
      }
      cv::Point img_loc;
      try
      {
        // Transform point into the camera frame
        PointStamped image_frame_loc_m;
        tf_->transformPoint(target_frame, cur_point, image_frame_loc_m);
        /*
           ROS_INFO(">> %f, %f %f", 
           K.at<float>(0,0),
           K.at<float>(1,1),
           K.at<float>(0,2)
           );
        //cur_point.point.x,cur_point.point.y,cur_point.point.z);
         */
        // Project point onto the image
        img_loc.x = static_cast<int>((K.at<float>(0,0)*image_frame_loc_m.point.x +
              K.at<float>(0,2)*image_frame_loc_m.point.z) /
            image_frame_loc_m.point.z);
        img_loc.y = static_cast<int>((K.at<float>(1,1)*image_frame_loc_m.point.y +
              K.at<float>(1,2)*image_frame_loc_m.point.z) /
            image_frame_loc_m.point.z);
      }
      catch (tf::TransformException e)
      {
        ROS_ERROR("[vs]%s", e.what());
      }
      return img_loc;
    }


  protected:
    ros::NodeHandle n_;
    ros::NodeHandle n_private_;
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Synchronizer<MySyncPolicy> sync_;
    image_transport::ImageTransport it_;
    sensor_msgs::CameraInfo cam_info_;
    shared_ptr<tf::TransformListener> tf_;

    // frames
    cv::Mat cur_workspace_mask_;
    std_msgs::Header cur_camera_header_;
    XYZPointCloud cloud_;
    int display_wait_ms_;
    int num_downsamples_;
    std::string workspace_frame_;
    std::string optical_frame_;
    std::string left_tool_frame_;
    std::string right_tool_frame_;
    std::string tool_frame_;
    std::string task_frame_;
    bool camera_initialized_;
    bool desire_points_initialized_;
    std::string cam_info_topic_;
    cv::Mat K;

    std::ofstream textFile;
    ros::ServiceClient p_client_;

    // segmenting
    int target_hue_value_;
    int target_hue_threshold_;
    int default_sat_bot_value_;
    int default_sat_top_value_;
    int default_val_value_;
    double min_contour_size_;


    float x_, y_, z_;
    std::string which_arm_;
    cv::Point tooltip_;
    int mode;
    uint counter_;
    std::vector<cv::KeyPoint> kpso_;
    std::vector<cv::KeyPoint> kps_;
    std::vector<cv::KeyPoint> kpsc_;
    std::vector<cv::KeyPoint> kps_bad_;
    pcl::PointXYZ tcp_e_;
    //std::vector<Descriptor>  ds;
    std::vector<cv::DMatch> good_matches_;

    std::vector<Data> ds_;
    cv::Point tooltip_star_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gripper_seg");

  log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
  my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
  //my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Debug]);
  ros::NodeHandle n;
  GripperSegmentationCollector vsa(n);
  ros::spin();
  return 0;
}
