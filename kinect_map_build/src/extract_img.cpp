#include "rosbag/bag.h"
#include "rosbag/view.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"
#include "boost/foreach.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

int main(int argn, char** argv)
{
    if(argn < 4)
        printf("extract_img bag topic ind output\n");
    int im_ind = atoi(argv[3]);
    rosbag::Bag bag(argv[1]);
    rosbag::View view(bag, rosbag::TopicQuery(argv[2]));
    BOOST_FOREACH(rosbag::MessageInstance const m, view) {
        if(im_ind-- > 0) 
            continue;
        sensor_msgs::Image::ConstPtr img = m.instantiate<sensor_msgs::Image>();
        if (img == NULL)
            continue;
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        cv::imwrite(argv[4], cv_ptr->image);
        bag.close();
        return 0;
    }
}
