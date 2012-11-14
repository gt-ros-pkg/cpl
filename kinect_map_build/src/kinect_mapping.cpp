#include <stdio.h>
#include <iostream>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>

using namespace cv;
using namespace std;

typedef pcl::PointXYZRGB PRGB;
typedef pcl::PointCloud<PRGB> PCRGB;

void makePCColor(const PCRGB &pc, uint32_t rgb)
{
    for(size_t i=0;i<pc.size();i++)
        ((uint32_t*) &pc[i].rgb)[0] = rgb;
}

void visClouds(const PCRGB &pc_1, const PCRGB &pc_2)
{
    pcl::visualization::CloudViewer viewer("Viz");
    makePCColor(pc_1, 0xffff0000);
    makePCColor(pc_2, 0xff0000ff);
    viewer.showCloud(pc_1.makeShared(), "a");
    viewer.showCloud(pc_2.makeShared(), "b");
    while(!viewer.wasStopped()) {}
}

void getMatches(const Mat &img_1, const Mat &img_2,
                vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2,
                vector<DMatch> &matches) 
{
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;

    SurfFeatureDetector detector( minHessian );

    detector.detect( img_1, keypoints_1 );
    detector.detect( img_2, keypoints_2 );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;

    Mat descriptors_1, descriptors_2;

    extractor.compute( img_1, keypoints_1, descriptors_1 );
    extractor.compute( img_2, keypoints_2, descriptors_2 );

    //-- Step 3: Matching descriptor vectors with a brute force matcher
    BruteForceMatcher< L2<float> > matcher;
    matcher.match( descriptors_1, descriptors_2, matches );
    printf("Num features img 1: %d\n", keypoints_1.size());
    printf("Num features img 2: %d\n", keypoints_2.size());
    printf("Num matches: %d\n", matches.size());

}

void extractKeypoints(const PCRGB &pc, const vector<KeyPoint> &kps, PCRGB &pc_out)
{
    /*
            pcl::visualization::CloudViewer viewer1("Viz");
            viewer1.showCloud(pc.makeShared(), "b");
            while(!viewer1.wasStopped()) {}
    for(size_t i=0;i<pc.size();i++) {
        PRGB pt = pc.points.at(i);
        //pt.x /= pt.z; pt.y /= pt.z; pt.z = 1;
        pc_out.push_back(pt);
    }
    pcl::visualization::CloudViewer viewer("Viz");
    viewer.showCloud(pc_out.makeShared(), "b");
    while(!viewer.wasStopped()) {}
    */
    for(size_t i=0;i<kps.size();i++) {
        //PRGB pt = pc[kps.at(i).pt.y*640 + kps.at(i).pt.x];
        //pt.x /= pt.z; pt.y /= pt.z; pt.z = 1;
        //pt.x = kps.at(i).pt.x/200; pt.y = kps.at(i).pt.y/200; pt.z = 1;
        //pc_out.push_back(pt);
        pc_out.push_back(pc.at(kps.at(i).pt.x, kps.at(i).pt.y));
    }
    //pcl::visualization::CloudViewer viewer("Viz");
    //viewer.showCloud(pc_out.makeShared(), "b");
    //while(!viewer.wasStopped()) {}
}

void pruneMatches(const PCRGB &kps_1, const PCRGB &kps_2, const vector<DMatch> &matches_in,
                  vector<DMatch> &matches_out)
{
    int ind_1, ind_2;
    for(size_t i=0;i<matches_in.size();i++) {
        ind_1 = matches_in[i].queryIdx;
        ind_2 = matches_in[i].trainIdx;
        if(kps_1[ind_1].x != kps_1[ind_1].x || kps_2[ind_2].x != kps_2[ind_2].x)
            continue;
        matches_out.push_back(matches_in[i]);
    }
}

void extractMatches(const PCRGB &kps_1, const PCRGB &kps_2, const vector<DMatch> &matches, 
                    PCRGB &matches1, PCRGB &matches2)
{
    int ind_1, ind_2;
    for(size_t i=0;i<matches.size();i++) {
        ind_1 = matches[i].queryIdx;
        ind_2 = matches[i].trainIdx;
        matches1.push_back(kps_1[ind_1]);
        matches2.push_back(kps_2[ind_2]);
    }
    //visClouds(matches1, matches2);
}

Eigen::Matrix4f umeyamaRegistration(const PCRGB& pc1, const PCRGB& pc2)
{
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    Eigen::MatrixXf m1(3,pc1.size()), m2(3,pc1.size());
    for(size_t i=0;i<pc1.size();i++) {
        m1(0,i) = pc1[i].x; m1(1,i) = pc1[i].y; m1(2,i) = pc1[i].z; 
        m2(0,i) = pc2[i].x; m2(1,i) = pc2[i].y; m2(2,i) = pc2[i].z; 
    }
    Eigen::Vector3f mean1 = m1.rowwise().mean();
    Eigen::Vector3f mean2 = m2.rowwise().mean();
    /*
    cout<< "mean1" << mean1 << endl;
    cout<< "mean2" << mean2 << endl;
    cout<< "m1" << m1 << endl;
    cout<< "m2" << m2 << endl;
    */
    m1.colwise() -= mean1; m2.colwise() -= mean2;
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(m1 * m2.transpose(), 
                                          Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f rot = svd.matrixV() * svd.matrixU().transpose();
    trans.block<3,3>(0,0) = rot;
    trans.block<3,1>(0,3) = (mean2-rot*mean1);
    return trans;
}

void transPoints(const PCRGB &pc_in, const Eigen::Matrix4f &trans, PCRGB &pc_out)
{
    Eigen::MatrixXf m(4,pc_in.size());
    for(size_t i=0;i<pc_in.size();i++) {
        m(0,i) = pc_in[i].x; m(1,i) = pc_in[i].y; m(2,i) = pc_in[i].z; m(3,i) = 1;
    }
    m = trans * m;
    for(size_t i=0;i<pc_in.size();i++) {
        PRGB pt;
        pt.x = m(0,i); pt.y = m(1,i); pt.z = m(2,i); pt.rgb = pc_in[i].rgb;
        pc_out.push_back(pt);
    }
}

int numInliers(const PCRGB &pc_1, const PCRGB &pc_2, double dist_thresh)
{
    int num_inliers = 0;
    double dist, diffx, diffy, diffz;
    for(size_t i=0;i<pc_1.size();i++) {
        diffx = pc_1[i].x - pc_2[i].x;
        diffy = pc_1[i].y - pc_2[i].y;
        diffz = pc_1[i].z - pc_2[i].z;
        dist = diffx*diffx + diffy*diffy + diffz*diffz;
        if(dist < dist_thresh*dist_thresh)
            num_inliers++;
    }
    return num_inliers;
}

Eigen::Matrix4f findTransRansac(const PCRGB &kps_1, const PCRGB &kps_2, const vector<DMatch> &matches)
{
    int sample_size, num_iters, num_inliers, max_inliers = -1;
    double dist_thresh;
    ros::NodeHandle nh_priv("~");
    nh_priv.param<double>("dist_thresh", dist_thresh, 0.05);
    printf("%f\n", dist_thresh);
    nh_priv.param<int>("sample_size", sample_size, 5);
    nh_priv.param<int>("num_iters", num_iters, 100);
    boost::mt19937 rgen;
    boost::uniform_int<> randint(0,matches.size()-1);
    Eigen::Matrix4f best_trans, cur_trans;
    vector<DMatch> match_sample;
    PCRGB kp_sample1, kp_sample2, kp_all1, kp_all2, kp_all2_trans, kp_sample2_trans;
    extractMatches(kps_1, kps_2, matches, kp_all1, kp_all2);
    //visClouds(kp_all1, kp_all2);
    for(int i=0;i<num_iters;i++) {
        for(int j=0;j<sample_size;j++) 
            match_sample.push_back(matches[randint(rgen)]);
        extractMatches(kps_1, kps_2, match_sample, kp_sample1, kp_sample2);
        cur_trans = umeyamaRegistration(kp_sample1, kp_sample2);
        if(0) {
            printf("here a\n");
            visClouds(kp_sample1, kp_sample2);
            transPoints(kp_sample2, cur_trans.inverse(), kp_sample2_trans);
            printf("here b\n");
            visClouds(kp_sample1, kp_sample2_trans);
        }
        //cout<< "cur_trans" << cur_trans << endl;
        transPoints(kp_all2, cur_trans.inverse(), kp_all2_trans);
        num_inliers = numInliers(kp_all1, kp_all2_trans, dist_thresh);
        if(num_inliers > max_inliers) {
            max_inliers = num_inliers;
            best_trans = cur_trans;
        }
        match_sample.clear();
        kp_sample1.clear(); kp_sample2.clear(); kp_all2_trans.clear(); kp_sample2_trans.clear();
    }
    printf("Max inliers: %d\n", max_inliers);
    //cout << best_trans << endl;
    return best_trans;
}

Eigen::Matrix4f registerPCs(const Mat &img_1, const PCRGB &pc_1, const Mat &img_2, const PCRGB &pc_2)
{
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector< DMatch > matches, matches_pruned;
    PCRGB kps_1, kps_2;
    getMatches(img_1, img_2, keypoints_1, keypoints_2, matches);
    extractKeypoints(pc_1, keypoints_1, kps_1);
    extractKeypoints(pc_2, keypoints_2, kps_2);
    //visClouds(kps_1, kps_2);
    pruneMatches(kps_1, kps_2, matches, matches_pruned);
    return findTransRansac(kps_1, kps_2, matches_pruned);
}

void readBag(char* filename, vector<Mat> &imgs, vector<PCRGB::Ptr> &pcs, size_t num=-1)
{
    printf("Reading bag...");
    if(num <= 0)
        num = pcs.size();
    rosbag::Bag bag_in(filename, rosbag::bagmode::Read);
    vector<string> topics;
    string pc_topic = "/pc";
    string color_img_topic = "/img";
    topics.push_back(pc_topic);
    topics.push_back(color_img_topic);
    rosbag::View view(bag_in, rosbag::TopicQuery(topics));
    BOOST_FOREACH(rosbag::MessageInstance const m, view) {
        if(m.getTopic() == pc_topic) {
            PCRGB::Ptr new_pc(new PCRGB());
            pcl::fromROSMsg<PRGB>(*m.instantiate<sensor_msgs::PointCloud2>(),*new_pc);
            pcs.push_back(new_pc);
        }
        if(m.getTopic() == color_img_topic) {
            sensor_msgs::Image::ConstPtr img = m.instantiate<sensor_msgs::Image>();
            cv_bridge::CvImagePtr cv_ptr;
            cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
            imgs.push_back(cv_ptr->image);
        }
        if(--num == 0) break;
    }
    bag_in.close();
    printf("done.\n");
}

void writeBag(char* filename, PCRGB &pc)
{
    rosbag::Bag bag_out(filename, rosbag::bagmode::Write);
    sensor_msgs::PointCloud2 pc_msg;
    pcl::toROSMsg<PRGB>(pc, pc_msg);
    bag_out.write("pc", ros::Time::now(), pc_msg);
    bag_out.close();
}

int main( int argc, char** argv )
{
    ros::init(argc, argv, "kinect_mapping");
    ros::NodeHandle nh_priv("~");
    vector<Mat> imgs;
    vector<PCRGB::Ptr> pcs;
    readBag(argv[1], imgs, pcs, -1);
    int step, start;
    int pc1_ind, pc2_ind;
    nh_priv.param<int>("step", step, 10);
    nh_priv.param<int>("start", start, 0);
    PCRGB::Ptr pc_sum(new PCRGB());
    PCRGB pc_trans;
    *pc_sum += *pcs[0];
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    for(int i=0;i<10;i++) {
        pc1_ind = start + i*step; pc2_ind = start + (i+1)*step;
        Eigen::Matrix4f new_trans = registerPCs(imgs[pc1_ind], *pcs[pc1_ind], imgs[pc2_ind], *pcs[pc2_ind]);
        trans *= new_trans.inverse();
        //cout << trans << endl;
        transPoints(*pcs[pc2_ind], trans, pc_trans);
        *pc_sum += pc_trans;
        pc_trans.clear();
    }
    writeBag(argv[2], *pc_sum);
    pcl::visualization::CloudViewer viewer("Registrations");
    viewer.showCloud(pc_sum);
    while(!viewer.wasStopped()) {}
    return 0;
}
