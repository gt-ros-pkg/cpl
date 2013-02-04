// openni_tracker.cpp

#include <string>
#include <vector>
#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include <kdl/frames.hpp>
#include <openni_tracker_msgs/jointData.h>
#include <openni_tracker_msgs/skeletonData.h>

#include <XnOpenNI.h>
#include <XnCodecIDs.h>
#include <XnCppWrapper.h>

using std::string;
using namespace std;
using namespace xn;

xn::Context        g_Context;
xn::DepthGenerator g_DepthGenerator;
xn::UserGenerator  g_UserGenerator;

XnBool g_bNeedPose   = FALSE;
XnChar g_strPose[20] = "";

int kinectToUse;
string frame_id;
ros::Publisher skeleton_pub;

void XN_CALLBACK_TYPE User_NewUser(xn::UserGenerator& generator, XnUserID nId, void* pCookie) {
        ROS_INFO("New User %d", nId);

        if (g_bNeedPose)
                g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
        else
                g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}

void XN_CALLBACK_TYPE User_LostUser(xn::UserGenerator& generator, XnUserID nId, void* pCookie) {
        ROS_INFO("Lost user %d", nId);
}

void XN_CALLBACK_TYPE UserCalibration_CalibrationStart(xn::SkeletonCapability& capability, XnUserID nId, void* pCookie) {
        ROS_INFO("Calibration started for user %d", nId);
}

void XN_CALLBACK_TYPE UserCalibration_CalibrationEnd(xn::SkeletonCapability& capability, XnUserID nId, XnBool bSuccess, void* pCookie) {
        if (bSuccess) {
                ROS_INFO("Calibration complete, start tracking user %d", nId);
                g_UserGenerator.GetSkeletonCap().StartTracking(nId);
        }
        else {
                ROS_INFO("Calibration failed for user %d", nId);
                if (g_bNeedPose)
                        g_UserGenerator.GetPoseDetectionCap().StartPoseDetection(g_strPose, nId);
                else
                        g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
        }
}

void XN_CALLBACK_TYPE UserPose_PoseDetected(xn::PoseDetectionCapability& capability, XnChar const* strPose, XnUserID nId, void* pCookie) {
    ROS_INFO("Pose %s detected for user %d", strPose, nId);
    g_UserGenerator.GetPoseDetectionCap().StopPoseDetection(nId);
    g_UserGenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}

void publishTransform(XnUserID const& user, XnSkeletonJoint const& joint, string const& frame_id, string const& child_frame_id) {
    static tf::TransformBroadcaster br;

    XnSkeletonJointPosition joint_position;
    g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(user, joint, joint_position);
    double x = -joint_position.position.X / 1000.0;
    double y = joint_position.position.Y / 1000.0;
    double z = joint_position.position.Z / 1000.0;

    XnSkeletonJointOrientation joint_orientation;
    g_UserGenerator.GetSkeletonCap().GetSkeletonJointOrientation(user, joint, joint_orientation);

    XnFloat* m = joint_orientation.orientation.elements;
    KDL::Rotation rotation(m[0], m[1], m[2],
                                           m[3], m[4], m[5],
                                           m[6], m[7], m[8]);
    double qx, qy, qz, qw;
    rotation.GetQuaternion(qx, qy, qz, qw);

    char child_frame_no[128];
    snprintf(child_frame_no, sizeof(child_frame_no), "%s_%d", child_frame_id.c_str(), user);

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(x, y, z));
    transform.setRotation(tf::Quaternion(qx, -qy, -qz, qw));

    // #4994
    tf::Transform change_frame;
    change_frame.setOrigin(tf::Vector3(0, 0, 0));
    tf::Quaternion frame_rotation;
    frame_rotation.setEulerZYX(1.5708, 0, 1.5708);
    change_frame.setRotation(frame_rotation);

    transform = change_frame * transform;

    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), frame_id, child_frame_no));
}

void publishTransforms(const std::string& frame_id) {
    XnUserID users[15];
    XnUInt16 users_count = 15;
    g_UserGenerator.GetUsers(users, users_count);

    for (int i = 0; i < users_count; ++i) {
        XnUserID user = users[i];
        if (!g_UserGenerator.GetSkeletonCap().IsTracking(user))
            continue;

        openni_tracker_msgs::skeletonData skeletonMsg;
        skeletonMsg.kinectID=kinectToUse;
        skeletonMsg.userID=i;
        skeletonMsg.header.stamp = ros::Time::now();
        skeletonMsg.header.frame_id = frame_id;
        for(int k=0;k<=XN_SKEL_RIGHT_FOOT;k++)
          {

            XnSkeletonJointPosition joint_position;
            g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(user,(XnSkeletonJoint) k, joint_position);
            XnSkeletonJointOrientation joint_orientation;
            g_UserGenerator.GetSkeletonCap().GetSkeletonJointOrientation(user,(XnSkeletonJoint) k, joint_orientation);
            XnFloat* m = joint_orientation.orientation.elements;
            KDL::Rotation rotation(m[0], m[1], m[2],
                                   m[3], m[4], m[5],
                                   m[6], m[7], m[8]);
            double x = -joint_position.position.X / 1000.0;
            double y = joint_position.position.Y / 1000.0;
            double z = joint_position.position.Z / 1000.0;          
            double qx, qy, qz, qw;
            rotation.GetQuaternion(qx, qy, qz, qw);
            tf::Transform transform;
            transform.setOrigin(tf::Vector3(x, y, z));
            transform.setRotation(tf::Quaternion(qx, -qy, -qz, qw));
            
            
            tf::Transform change_frame;
            change_frame.setOrigin(tf::Vector3(0, 0, 0));
            tf::Quaternion frame_rotation;
            frame_rotation.setEulerZYX(1.5708, 0, 1.5708);
            change_frame.setRotation(frame_rotation);
            transform = change_frame * transform;
            

            openni_tracker_msgs::jointData jointData;
            jointData.pose.position.x=transform.getOrigin().x();
            jointData.pose.position.y=transform.getOrigin().y();
            jointData.pose.position.z=transform.getOrigin().z();
            jointData.pose.orientation.x=0.0;
            jointData.pose.orientation.y=0.0;
            jointData.pose.orientation.z=0.0;
            jointData.pose.orientation.w=0.0;
            jointData.jointID=k;
            jointData.confidence=joint_position.fConfidence;
            skeletonMsg.joints.push_back(jointData);
          }
        skeleton_pub.publish(skeletonMsg);
        /*
        publishTransform(user, XN_SKEL_HEAD,           frame_id, "head");
        publishTransform(user, XN_SKEL_NECK,           frame_id, "neck");
        publishTransform(user, XN_SKEL_TORSO,          frame_id, "torso");

        publishTransform(user, XN_SKEL_LEFT_SHOULDER,  frame_id, "left_shoulder");
        publishTransform(user, XN_SKEL_LEFT_ELBOW,     frame_id, "left_elbow");
        publishTransform(user, XN_SKEL_LEFT_HAND,      frame_id, "left_hand");

        publishTransform(user, XN_SKEL_RIGHT_SHOULDER, frame_id, "right_shoulder");
        publishTransform(user, XN_SKEL_RIGHT_ELBOW,    frame_id, "right_elbow");
        publishTransform(user, XN_SKEL_RIGHT_HAND,     frame_id, "right_hand");

        publishTransform(user, XN_SKEL_LEFT_HIP,       frame_id, "left_hip");
        publishTransform(user, XN_SKEL_LEFT_KNEE,      frame_id, "left_knee");
        publishTransform(user, XN_SKEL_LEFT_FOOT,      frame_id, "left_foot");

        publishTransform(user, XN_SKEL_RIGHT_HIP,      frame_id, "right_hip");
        publishTransform(user, XN_SKEL_RIGHT_KNEE,     frame_id, "right_knee");
        publishTransform(user, XN_SKEL_RIGHT_FOOT,     frame_id, "right_foot");
        */
    }
}

#define CHECK_RC(nRetVal, what)                                                                         \
        if (nRetVal != XN_STATUS_OK)                                                                    \
        {                                                                                                                               \
                ROS_ERROR("%s failed: %s", what, xnGetStatusString(nRetVal));\
                return nRetVal;                                                                                         \
        }

int main(int argc, char **argv) {
    ros::init(argc, argv, "openni_tracker_double");
    ros::NodeHandle nh("~");
    skeleton_pub = nh.advertise<openni_tracker_msgs::skeletonData>("/skeleton_data", 1000);

    string configFilename = ros::package::getPath("openni_tracker") + "/openni_tracker.xml";
    XnStatus nRetVal = g_Context.InitFromXmlFile(configFilename.c_str());
    CHECK_RC(nRetVal, "InitFromXml");

    nh.param("kinectToUse", kinectToUse, 0);
    std::cout<<"kinectToUse: "<<kinectToUse<<std::endl;
    
    string temp="openni_depth_frame";
    nh.param("frame_id",frame_id, temp);
    std::cout<<"frame_id: "<<frame_id<<std::endl;

/*
    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator);
    CHECK_RC(nRetVal, "Find depth generator");
*/

    NodeInfoList userList;
    NodeInfoList depthList;
    nRetVal = g_Context.EnumerateProductionTrees(XN_NODE_TYPE_DEPTH, NULL, depthList);
    xnPrintError(nRetVal, "Looking for depth generators..: ");

   int i = 0;
    for (NodeInfoList::Iterator it = depthList.Begin();it != depthList.End(); ++it)
    {

      if(i == kinectToUse)        
      {
        std::cout<<"i: "<<i<<std::endl;
            // Create the device node
            NodeInfo deviceInfo = *it;
            XnProductionNodeDescription nodeDesc = deviceInfo.GetDescription();
            cout << "Creating: " << nodeDesc.strName << " ,Vendor: " << nodeDesc.strVendor << ", Type: " << nodeDesc.Type << " Instance: " << deviceInfo.GetInstanceName() << endl;
            nRetVal = g_Context.CreateProductionTree(deviceInfo, g_DepthGenerator);
            xnPrintError(nRetVal, "Creating depthGen.: ");

            nRetVal = g_Context.EnumerateProductionTrees(XN_NODE_TYPE_USER, NULL, userList, NULL);
            xnPrintError(nRetVal, "Looking for user generators: ");

            if(userList.Begin() == userList.End())
              {
                cout << "There is no user generators" << endl;
              }
            
            int j = 0;
            int stop_ind = (kinectToUse == 0) ? 0 : 3;
            for(NodeInfoList::Iterator nodeIt = userList.Begin(); (nodeIt != userList.End()); nodeIt++)
            {
              j++;
              std::cout<<"j: "<<j<<std::endl;
              
              if(j++ == stop_ind) 
                {
                  NodeInfo info = *nodeIt;
                  XnProductionNodeDescription nodeDesc2 = info.GetDescription();
                  cout << "Info: " << nodeDesc2.strName << " ,Vendor: " << nodeDesc2.strVendor << ", Type: " << nodeDesc2.Type << " ,Instance: " << info.GetInstanceName() << endl;
                  nRetVal = g_Context.CreateProductionTree(info, g_UserGenerator);
                  xnPrintError(nRetVal, "Creating the user gen.: ");
                  
                  if(g_UserGenerator.IsValid())
                    {
                      cout << "User generator is working.." << endl;
                      
                      XnCallbackHandle h1, h2, h3;
                      g_UserGenerator.RegisterUserCallbacks(User_NewUser, User_LostUser, NULL, h1);
                      g_UserGenerator.GetPoseDetectionCap().RegisterToPoseCallbacks(UserPose_PoseDetected, NULL, NULL, h2);
                      g_UserGenerator.GetSkeletonCap().RegisterCalibrationCallbacks(UserCalibration_CalibrationStart, UserCalibration_CalibrationEnd, NULL, h3);
                      //genUser.GetSkeletonCap().SetSmoothing(1.0);
                      
                      // Set the profile
                      g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);
                    }
                }
              
            }
        }
        i++;
    }




/*
        nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_USER, g_UserGenerator);
        if (nRetVal != XN_STATUS_OK) {
                nRetVal = g_UserGenerator.Create(g_Context);
                CHECK_RC(nRetVal, "Find user generator");
                nh.param("kinectToUse", kinectToUse, 0);
        }

        if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_SKELETON)) {
                ROS_INFO("Supplied user generator doesn't support skeleton");
                return 1;
        }
*/


/*
    XnCallbackHandle hUserCallbacks;
        g_UserGenerator.RegisterUserCallbacks(User_NewUser, User_LostUser, NULL, hUserCallbacks);

        XnCallbackHandle hCalibrationCallbacks;
        g_UserGenerator.GetSkeletonCap().RegisterCalibrationCallbacks(UserCalibration_CalibrationStart, UserCalibration_CalibrationEnd, NULL, hCalibrationCallbacks);

*/



        if (g_UserGenerator.GetSkeletonCap().NeedPoseForCalibration()) {
                g_bNeedPose = TRUE;
                if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_POSE_DETECTION)) {
                        ROS_INFO("Pose required, but not supported");
                        return 1;
                }

                XnCallbackHandle hPoseCallbacks;
                g_UserGenerator.GetPoseDetectionCap().RegisterToPoseCallbacks(UserPose_PoseDetected, NULL, NULL, hPoseCallbacks);

                g_UserGenerator.GetSkeletonCap().GetCalibrationPose(g_strPose);
        }

        //g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);

        nRetVal = g_Context.StartGeneratingAll();
        CHECK_RC(nRetVal, "StartGenerating");

        ros::Rate r(30);
        
        while (ros::ok()) {
                g_Context.WaitAndUpdateAll();
                publishTransforms(frame_id);
                r.sleep();              
        }

        g_Context.Shutdown();
        return 0;
}
