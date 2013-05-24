
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>

#include "opencv2/opencv.hpp"

#include <ros/ros.h>
#include <ros/package.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <std_msgs/String.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose.h>
#include <ar_track_alvar/AlvarMarkers.h>
#include <cv_bridge/cv_bridge.h>

//#define USE_ROS_RATE

#define ROS_TOPIC_LEFTHAND  "/left_hand"
#define ROS_TOPIC_RIGHTHAND "/right_hand"
#define ROS_TOPIC_BINMARKES "/ar_pose_marker"
#define ROS_TOPIC_IMAGE     "/kinect0/rgb/image_rect_color"

#define TF_WORLD     "base_link"
#define TF_KINECT    "kinect0_rgb_optical_frame"
#define TF_WEBCAM    "lifecam1_optical_frame"

#define BIN_NUM      20

#define VIDEO_FILE  "out.avi"
#define TXT_FILE    "out.txt"

#define FPS 30

char * output_video = VIDEO_FILE;
char * output_text  = TXT_FILE;
int    fps          = FPS;

bool quit = false;
cv::VideoWriter vid;
FILE * pfile = NULL;
int framecount = 0;


cv::Mat image  = cv::Mat(500, 500, CV_8UC3);
cv::Mat image2 = cv::Mat(500, 500, CV_8UC3);
geometry_msgs::PoseStamped lefthand, righthand;
std::vector<ar_track_alvar::AlvarMarker> bins;

int lefthand_msgnum  = 0;
int righthand_msgnum = 0;
int bins_msgnum      = 0;
int image_msgnum     = 0;

///////////////////////////////////////////////////////////////////////////
// Transform
/////////////////////////////////////////////////////////////////////////////

tf::TransformListener * tl = NULL;

tf::StampedTransform kinect_to_w;
tf::StampedTransform webcam_to_w;

void calculate_camera_transforms()
{
    bool success = false;


    while (success == false)
    {
        try
        {
            tl->lookupTransform(TF_WORLD, TF_KINECT, ros::Time(0), kinect_to_w);
            tl->lookupTransform( TF_WORLD, TF_WEBCAM, ros::Time(0), webcam_to_w);

            success = true;

        } catch (std::exception e)
        {
            std::cout << e.what() << std::endl;

            ros::Rate r(10);
            r.sleep();
        }
    }
}

geometry_msgs::Point transform_point(geometry_msgs::Point point, tf::StampedTransform transform)
{
    tf::Vector3 r = transform * tf::Vector3(point.x, point.y, point.z);
    point.x = r.getX();
    point.y = r.getY();
    point.z = r.getZ();
    return point;
}

geometry_msgs::Pose transform_pose(geometry_msgs::Pose pose, tf::StampedTransform transform)
{
    tf::Pose posetf;
    tf::poseMsgToTF(pose, posetf);
    posetf.mult(transform, posetf);
    tf::poseTFToMsg(posetf, pose);
    return pose;
}

///////////////////////////////////////////////////////////////////////////
// Record data
/////////////////////////////////////////////////////////////////////////////

void record_nextframe()
{
    geometry_msgs::Point point;
    geometry_msgs::Pose  pose;

    if (lefthand_msgnum <= 0 || righthand_msgnum <= 0 || bins_msgnum <= 0 || image_msgnum <= 0)
        return;

    if (vid.isOpened() == false)
    {
        vid.open(output_video, CV_FOURCC('X','V','I','D'), fps,
                 cv::Size(image.cols * 1, image.rows), true);
        assert(vid.isOpened());
    }

    framecount++;

    // write & show image
    image.copyTo(image2);
    cv::circle(image2, cv::Point(lefthand.pose.position.x * 525  / lefthand.pose.position.z + 320,
                                 lefthand.pose.position.y * 525  / lefthand.pose.position.z + 240),
        6, cv::Scalar(0, 0, 255), -1);
    cv::circle(image2, cv::Point(righthand.pose.position.x * 525  / righthand.pose.position.z + 320,
                                 righthand.pose.position.y * 525  / righthand.pose.position.z + 240),
        6, cv::Scalar(0, 255, 0), -1);
    static char text[1024];
    sprintf(text, "Frame %d, lefthand %d, righthand %d, bins %d, image %d", 
              framecount, lefthand_msgnum, righthand_msgnum, bins_msgnum, image_msgnum);
    cv::putText(image2, text, cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    vid.write(image2);


    // write text file
    fprintf(pfile, "Frame %d, lefthand_msgnum %d, righthand_msgnum %d, bins_msgnum %d, image_msgnum %d \n", 
              framecount, lefthand_msgnum, righthand_msgnum, bins_msgnum, image_msgnum);
    printf("Frame %d, lefthand_msgnum %d, righthand_msgnum %d, bins_msgnum %d, image_msgnum %d \n", 
              framecount, lefthand_msgnum, righthand_msgnum, bins_msgnum, image_msgnum);


    // hands in world frame
    pose = transform_pose(lefthand.pose, kinect_to_w);
    fprintf(pfile, "%f %f %f\n", pose.position.x, pose.position.y, pose.position.z);
    pose = transform_pose(righthand.pose, kinect_to_w);
    fprintf(pfile, "%f %f %f\n", pose.position.x, pose.position.y, pose.position.z);

    // bins in world frame
    for (int i = 1; i <= BIN_NUM; i++)
    {
        // find bin i
        int id = -1;
        for (int j = 0; j < bins.size(); j++)
        {
            if (bins[j].id == i)
            {
                id = j;
            }
        }

        // ok
        if (id >= 0)
        {
            pose = transform_pose(bins[id].pose.pose, webcam_to_w);
            geometry_msgs::Point p = pose.position;
            geometry_msgs::Quaternion q = pose.orientation;
            fprintf(pfile, "%f %f %f %f %f %f %f\n", p.x, p.y, p.z, q.x, q.y, q.z, q.w);
        } else
        {
            fprintf(pfile, "%f %f %f %f %f %f %f\n", -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0);
        }
    }
}

///////////////////////////////////////////////////////////////////////////
// callbacks
/////////////////////////////////////////////////////////////////////////////

void image_callback(const sensor_msgs::Image::Ptr & msg)
{
    //printf("image callback\n");
    cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(msg);
    img->image.copyTo(image);
    assert(image.type() == CV_8UC3);
    image_msgnum++;
}

void lefthand_callback(const geometry_msgs::PoseStamped::Ptr & msg)
{
    //printf("righthand callback\n");
    lefthand = *msg;
    lefthand_msgnum++;
}

void righthand_callback(const geometry_msgs::PoseStamped::Ptr & msg)
{
    //printf("righthand callback\n");
    righthand = *msg;
    righthand_msgnum++;
}

void artags_callback(const ar_track_alvar::AlvarMarkers::Ptr & msg)
{
    //printf("artags callback\n");
    bins = msg->markers;
    bins_msgnum++;

}

///////////////////////////////////////////////////////////////////////////
// Main
/////////////////////////////////////////////////////////////////////////////

int main(int argc, char ** argv)
{
    printf("hello\n");
    ROS_ERROR("WTF");
    if (argc == 4)
    {
        fps          = atoi(argv[1]);
        output_video = argv[2];
        output_text  = argv[3];
    }

    // init file
    pfile = fopen(output_text, "wt");
    assert(pfile);

    // init ros & subscribe
    ros::init(argc, argv, "x_recording_training_data", 0);
    ros::NodeHandle nh;
    ros::Subscriber sub1 = nh.subscribe(ROS_TOPIC_IMAGE, 1000, image_callback);
    ros::Subscriber sub2 = nh.subscribe(ROS_TOPIC_LEFTHAND, 1000, lefthand_callback);
    ros::Subscriber sub3 = nh.subscribe(ROS_TOPIC_RIGHTHAND, 1000, righthand_callback);
    ros::Subscriber sub4 = nh.subscribe(ROS_TOPIC_BINMARKES, 1000, artags_callback);

    // lookup camera transform
    tl = new tf::TransformListener();
    calculate_camera_transforms();

    // Loop
    long int begin_time     = ros::Time::now().toNSec();
    long int elapsedtime    = 0;
    long int waittime       = 0;

    ros::Rate r(fps);

    while (quit == false)
    {
        ros::spinOnce();
        record_nextframe();

        cv::imshow("image", image2);

#ifndef USE_ROS_RATE
        elapsedtime = ros::Time::now().toNSec() - begin_time;
        waittime    = (long int) framecount * 1000000000 / fps - elapsedtime;
        int k       = cv::waitKey(MAX(1, waittime / 1000000));
#else
        r.sleep();
        int k = cv::waitKey(1);
#endif  

        if (k > 0)
            quit = true;
    }

    // ok
    fprintf(pfile, "%d frames in %f seconds, fps: %f\n", framecount,
           (ros::Time::now().toNSec() - begin_time) / 1000000000.0,
           framecount * 1000000000.0 / (ros::Time::now().toNSec() - begin_time));
    printf("%d frames in %f seconds, fps: %f\n", framecount,
           (ros::Time::now().toNSec() - begin_time) / 1000000000.0,
           framecount * 1000000000.0 / (ros::Time::now().toNSec() - begin_time));
    delete tl;
    fclose(pfile);
    vid.release();
    return 1;
}











