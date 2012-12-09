
#include "arm_mapping/corresp_pc_utils.h"

void readPCBag(const string& filename, const string& topic, PCRGB::Ptr& pc) 
{
    rosbag::Bag bag;
    bag.open(filename, rosbag::bagmode::Read);
    rosbag::View view(bag, rosbag::TopicQuery(topic.c_str()));
    BOOST_FOREACH(rosbag::MessageInstance const msg, view) {
        pc = msg.instantiate< PCRGB >();
        break;
    }
    bag.close();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pub_pc");
    if(argc < 6) {
      printf("pub_pc bag_name bag_topic out_topic out_frame rate\n");
      return -1;
    }
    ros::NodeHandle nh;
    PCRGB::Ptr input_pc;
    readPCBag(argv[1], argv[2], input_pc);

    ros::Publisher pc_pub = nh.advertise<sensor_msgs::PointCloud2>(argv[3], 100);
    ros::Rate r(atof(argv[5]));
    while(ros::ok()) {
        input_pc->header.stamp = ros::Time().now();
        input_pc->header.frame_id = argv[4];
        pc_pub.publish(input_pc);
        r.sleep();
    }
    return 0;
}
