// openni_tracker_modified.cpp

#include <ros/ros.h>
#include <ros/package.h>
#include <ros/param.h>
#include <tf/transform_broadcaster.h>
#include <kdl/frames.hpp>
#include <openni_tracker_msgs/jointData.h>
#include <openni_tracker_msgs/skeletonData.h>


#include <XnOpenNI.h>
#include <XnCodecIDs.h>
#include <XnCppWrapper.h>

using std::string;

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
    change_frame.setOrigin(tf::Vector3(0, 0, 0));//g_Device
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

    for (int i = 0; i < users_count; ++i) 
      {
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
	    jointData.pose.orientation.x=transform.getRotation()[0];
	    jointData.pose.orientation.y=transform.getRotation()[1];
	    jointData.pose.orientation.z=transform.getRotation()[2];
	    jointData.pose.orientation.w=transform.getRotation()[3];
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

#define CHECK_RC(nRetVal, what)                                     \
    if (nRetVal != XN_STATUS_OK)                                    \
    {                                                               \
        ROS_ERROR("%s failed: %s", what, xnGetStatusString(nRetVal));\
        return nRetVal;                                             \
    }

int main(int argc, char **argv) {
    ros::init(argc, argv, "openni_tracker", ros::init_options::AnonymousName);

    string configFilename = ros::package::getPath("openni_tracker_double") + "/openni_tracker.xml";
    XnStatus nRetVal = g_Context.InitFromXmlFile(configFilename.c_str());
    CHECK_RC(nRetVal, "InitFromXml");

    //XnStatus nRetVal = XN_STATUS_OK;

    //xnLogInitFromXmlFile(csXmlFile);

    nRetVal = g_Context.Init();
    XN_IS_STATUS_OK(nRetVal);


   // SELECTION OF THE DEVICE
    xn::EnumerationErrors errors;
    xn::Device g_Device;
        // find devices
    xn::NodeInfoList list;
    xn::NodeInfoList list_depth;
        nRetVal = g_Context.EnumerateProductionTrees(XN_NODE_TYPE_DEVICE, NULL, list, &errors);
    XN_IS_STATUS_OK(nRetVal);

    printf("The following devices were found:\n");
        int i = 1;
        for (xn::NodeInfoList::Iterator it = list.Begin(); it != list.End(); ++it, ++i)
        {
            xn::NodeInfo deviceNodeInfo = *it;

            xn::Device deviceNode;
            deviceNodeInfo.GetInstance(deviceNode);
            XnBool bExists = deviceNode.IsValid();
            if (!bExists)
            {
                g_Context.CreateProductionTree(deviceNodeInfo, deviceNode);
                // this might fail.
            }

            if (deviceNode.IsValid() && deviceNode.IsCapabilitySupported(XN_CAPABILITY_DEVICE_IDENTIFICATION))
            {
                const XnUInt32 nStringBufferSize = 200;
                XnChar strDeviceName[nStringBufferSize];
                XnChar strSerialNumber[nStringBufferSize];

                XnUInt32 nLength = nStringBufferSize;
                deviceNode.GetIdentificationCap().GetDeviceName(strDeviceName, nLength);
                nLength = nStringBufferSize;
                deviceNode.GetIdentificationCap().GetSerialNumber(strSerialNumber, nLength);
                printf("[%d] %s (%s)\n", i, strDeviceName, strSerialNumber);
            }
            else
            {
                printf("[%d] %s\n", i, deviceNodeInfo.GetCreationInfo());
            }

            // release the device if we created it
            if (!bExists && deviceNode.IsValid())
            {
                deviceNode.Release();
            }
        }
        printf("\n");
        printf("Choose device to open (1): ");

        int chosen = 1;
        int nRetval = scanf("%d", &chosen);

        // create it
        xn::NodeInfoList::Iterator it = list.Begin();
        for (i = 1; i < chosen; ++i)
        {
            it++;
        }

        xn::NodeInfo deviceNode = *it;
        nRetVal = g_Context.CreateProductionTree(deviceNode, g_Device);
        printf("Production tree of the device created.\n");

    // SELECTION OF THE DEPTH GENERATOR
        nRetVal = g_Context.EnumerateProductionTrees(XN_NODE_TYPE_DEPTH, NULL, list_depth, &errors);
        XN_IS_STATUS_OK(nRetVal);

        printf("The following devices were found:\n");
            int i_depth = 1;
            for (xn::NodeInfoList::Iterator it_depth = list_depth.Begin(); it_depth != list_depth.End(); ++it_depth, ++i_depth)
            {
                xn::NodeInfo depthNodeInfo = *it_depth;

                xn::Device depthNode;
                depthNodeInfo.GetInstance(depthNode);
                XnBool bExists_depth = depthNode.IsValid();
                if (!bExists_depth)
                {
                    g_Context.CreateProductionTree(depthNodeInfo, depthNode);
                    // this might fail.
                }

                if (depthNode.IsValid() && depthNode.IsCapabilitySupported(XN_CAPABILITY_DEVICE_IDENTIFICATION))
                {
                    const XnUInt32 nStringBufferSize = 200;
                    XnChar strDeviceName[nStringBufferSize];
                    XnChar strSerialNumber[nStringBufferSize];

                    XnUInt32 nLength = nStringBufferSize;
                    depthNode.GetIdentificationCap().GetDeviceName(strDeviceName, nLength);
                    nLength = nStringBufferSize;
                    depthNode.GetIdentificationCap().GetSerialNumber(strSerialNumber, nLength);
                    printf("[%d] %s (%s)\n", i, strDeviceName, strSerialNumber);
                }
                else
                {
                    printf("[%d] %s\n", i, depthNodeInfo.GetCreationInfo());
                }

                // release the device if we created it
                if (!bExists_depth && depthNode.IsValid())
                {
                    depthNode.Release();
                }
            }
            printf("\n");
        printf("Choose device to open (1): ");

        int chosen_depth = 1;
        int nRetval_depth = scanf("%d", &chosen);

        // create it
        xn::NodeInfoList::Iterator it_depth = list_depth.Begin();
        for (i = 1; i < chosen_depth; ++i)
        {
            it_depth++;
        }

        xn::NodeInfo depthNode = *it_depth;
        nRetVal = g_Context.CreateProductionTree(depthNode, g_DepthGenerator);
        printf("Production tree of the DepthGenerator created.\n");

        nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_DepthGenerator);
        printf("Production tree of the depth generator created.\n");
        XN_IS_STATUS_OK(nRetVal);
        printf("XN_IS_STATUS_OK(nRetVal).\n");



    CHECK_RC(nRetVal, "Find depth generator");
     printf("CHECK_RC(nRetVal, Find depth generator);\n");

    nRetVal = g_Context.FindExistingNode(XN_NODE_TYPE_USER, g_UserGenerator);
    printf("User generator found.\n");
    if (nRetVal != XN_STATUS_OK) {
        nRetVal = g_UserGenerator.Create(g_Context);
        printf("User generator created.\n");
        CHECK_RC(nRetVal, "Find user generator");
    }

    if (!g_UserGenerator.IsCapabilitySupported(XN_CAPABILITY_SKELETON)) {
        printf("Supplied user generator doesn't support skeleton.\n");
        ROS_INFO("Supplied user generator doesn't support skeleton");
        return 1;
    }

    XnCallbackHandle hUserCallbacks;
    g_UserGenerator.RegisterUserCallbacks(User_NewUser, User_LostUser, NULL, hUserCallbacks);

    XnCallbackHandle hCalibrationCallbacks;
    g_UserGenerator.GetSkeletonCap().RegisterCalibrationCallbacks(UserCalibration_CalibrationStart, UserCalibration_CalibrationEnd, NULL, hCalibrationCallbacks);

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

    g_UserGenerator.GetSkeletonCap().SetSkeletonProfile(XN_SKEL_PROFILE_ALL);

    nRetVal = g_Context.StartGeneratingAll();
    CHECK_RC(nRetVal, "StartGenerating");

    ros::NodeHandle pnh("~");
    ros::Rate r(30);
    string temp="openni_depth_frame";
    pnh.param("frame_id",frame_id, temp);    
    pnh.param("kinectToUse", kinectToUse, 0);
    std::cout<<"kinectToUse: "<<kinectToUse<<std::endl;
    skeleton_pub = pnh.advertise<openni_tracker_msgs::skeletonData>("/skeleton_data", 1000);

    while (ros::ok()) {
        g_Context.WaitAndUpdateAll();
        publishTransforms(frame_id);
        r.sleep();
    }

    g_Context.Shutdown();
    return 0;
}
