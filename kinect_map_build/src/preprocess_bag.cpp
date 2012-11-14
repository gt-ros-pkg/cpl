
#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"
#include "sensor_msgs/PointCloud2.h"
#include "boost/foreach.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

int main(int argn, char** argv)
{
    rosbag::Bag bag_in(argv[1], rosbag::bagmode::Read);
    rosbag::Bag bag_out(argv[2], rosbag::bagmode::Write);
    vector<string> topics;
    vector<sensor_msgs::PointCloud2::ConstPtr> pcs_all, pcs_filt;
    vector<sensor_msgs::Image::ConstPtr> imgs_filt;
    string pc_topic = "/camera/depth_registered/points";
    string gray_img_topic = "/camera/rgb/image_rect";
    string color_img_topic = "/camera/rgb/image_rect_color";
    topics.push_back(pc_topic);
    topics.push_back(gray_img_topic);
    topics.push_back(color_img_topic);
    rosbag::View view(bag_in, rosbag::TopicQuery(topics));
    BOOST_FOREACH(rosbag::MessageInstance const m, view) {
        if(m.getTopic() == pc_topic)
            pcs_all.push_back(m.instantiate<sensor_msgs::PointCloud2>());
    }
    BOOST_FOREACH(rosbag::MessageInstance const m, view) {
        if(m.getTopic() == color_img_topic) {
            sensor_msgs::Image::ConstPtr img = m.instantiate<sensor_msgs::Image>();
            double min_diff = 0.1, diff;
            size_t min_ind;
            for(size_t i=0;i<pcs_all.size();i++) {
                diff = fabs(pcs_all.at(i)->header.stamp.toSec() - img->header.stamp.toSec());
                if(diff < min_diff) {
                    min_diff = diff;
                    min_ind = i;
                }
            }
            if(min_diff < 0.1) {
                bag_out.write("/pc", pcs_all.at(min_ind)->header.stamp, pcs_all.at(min_ind));
                bag_out.write("/img", pcs_all.at(min_ind)->header.stamp, img);
            }
        }
    }
    bag_in.close();
    bag_out.close();
    return 0;
}
