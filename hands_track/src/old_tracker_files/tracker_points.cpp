#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <tf/transform_broadcaster.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/median.hpp>

#include <sstream>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cv.h>
#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace boost::accumulators;

typedef pcl::PointXYZRGB PRGB;
pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_pc(new pcl::PointCloud<pcl::PointXYZRGB>);

class handsTracker{

private:
  //declare frame globally
  ros::Time last_img_time;
  //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cur_pc;
  ros::NodeHandle nh;
  
  //kinect frame subscribe
  image_transport::Subscriber kinect_rgb_sub;
  //Point cloud subscribe	
  ros::Subscriber kinect_pc_sub ;

  //ASSUMING purple glove on right hand
  ros::Publisher purple_hand ;
  ros::Publisher green_hand ;
  ros::Publisher purple_hand_2d;
  ros::Publisher green_hand_2d;
  
  //Mats to be used
  Mat img_of_interest, bin_img, ioi_hsv;
  Mat temp_h, temp_s, temp_v, label_img;

  cv_bridge::CvImagePtr cv_ptr;

public:

  //constructor
  handsTracker ()
  {
  
    int img_ht=480, img_wd=640;
    double green_mean =  80.33/180, blue_mean =  126.993/180;
    double hard_th_green= 6.0, hard_th_blue= 2.0;
    int frames_to_hold =1;
    
    img_of_interest.create(img_ht, img_wd, CV_8UC3);
    bin_img.create(img_ht, img_wd, CV_8UC1);
    label_img.create(img_ht, img_wd, CV_32FC1);
    temp_h.create(img_ht, img_wd, CV_8UC1);
    temp_s.create(img_ht, img_wd, CV_8UC1);
    temp_v.create(img_ht, img_wd, CV_8UC1);
    
    
    image_transport::ImageTransport hand_tracking(nh);
    //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    //cur_pc = temp_pc;
    kinect_rgb_sub = hand_tracking.subscribe
      ("kinect0/rgb/image_rect_color", frames_to_hold, &handsTracker::kin_rgb_fram_sub, this);
    
    kinect_pc_sub = nh.subscribe
     ("kinect0/depth_registered/points",  1, &handsTracker::kin_pc_sub,this);

    //3d points
    purple_hand = nh.advertise<geometry_msgs::Point> ("right_hand", frames_to_hold);
    green_hand = nh.advertise<geometry_msgs::Point> ("left_hand", frames_to_hold);
    
    //2d points
    purple_hand_2d = nh.advertise<geometry_msgs::Point> ("right_hand_2d", 0);
    green_hand_2d = nh.advertise<geometry_msgs::Point> ("left_hand_2d", 0);
  }
  
  
  //read file with means, variance and covariance
  void read_file(const string file_name, float means[], float variance [], float* covariance)
  {
    ifstream read_it(file_name.c_str());
    string dummy;
    stringstream holder;

    if(read_it.is_open())
      {
	read_it >> dummy;
	read_it >> dummy;
	read_it >> means[0];
	read_it >> dummy;
	read_it >> means[1];
	read_it >> dummy;
	read_it >> dummy;
	read_it >> variance[0];
	read_it >> dummy;
	read_it >> variance[1];
	read_it >> dummy;
	read_it >> dummy;
	read_it >> *covariance;

      }
    else{
      cout<< "Can't open file!";
    }
  
  }

  //iterates through matrix once to compute binary image
  void thresh_one_go(double mean, double s_thresh, double v_thresh, double hard_thresh)
  {
    double diff;
    
    for (int r=0; r<temp_h.rows; r++)
      {
	const uchar* h_row = temp_h.ptr<uchar>(r);
	const uchar* s_row = temp_s.ptr<uchar>(r);
	const uchar* v_row = temp_v.ptr<uchar>(r);
	uchar* bin_row = bin_img.ptr<uchar>(r);
	
	for (int c=0; c<temp_h.cols; c++)
	  {
	    if((s_row[c]>=s_thresh) && (v_row[c]>=v_thresh))
	      {
		diff = fabs(float(h_row[c]) - float(mean));
		if (diff<=hard_thresh){bin_row[c]=1;}
		else{bin_row[c]=0;}
	      }
	    else
	      {
		bin_row[c] = 0;
	      }
	  }

      }

  }

  void hsv_get()
  {
    // Mat ioi_hsv, maha_dist;
    cvtColor(img_of_interest, ioi_hsv, CV_BGR2HSV);
    
    //split into different matrices the channels
    Mat hsv_container[] = {temp_h, temp_s, temp_v};
    int from_to[] = {0,0, 1,1, 2,2};
    mixChannels( &ioi_hsv, 1, hsv_container, 3, from_to, 3);
  }

  void new_comparison_mahalanobis(bool isGreen)
  {

    double hard_thresh, sigma, mean, s_thresh, v_thresh;
  
    int close_iterations = 1;
   
    if (isGreen)
      {
	hard_thresh = 6.0;
	sigma =  0.0026;
	mean =  80.33;
	s_thresh = 60;
	v_thresh= 50;

      }
    else
      {
	hard_thresh = 2.0;
	mean =  126.993;//0.4731; 
	s_thresh = 90;
	v_thresh= 50;

      }


    
    thresh_one_go(mean, v_thresh, s_thresh, hard_thresh);
    
    int struct_size = 3;
    Mat struc_elem = getStructuringElement(MORPH_ELLIPSE, Size(2*struct_size+1, 2*struct_size+1), Point (struct_size, struct_size));
    morphologyEx(bin_img, bin_img, MORPH_CLOSE, struc_elem, Point(-1,1), close_iterations);

  }


void find_blobs(const Mat &binary, vector < vector<Point2i> > &blobs, int* max_idx_out, int* max_area_out)
{
    blobs.clear();
    int max_area =0;
    int max_label =-1;
    vector<int> label_areas;
    label_areas.clear();


  
    binary.convertTo(label_img, CV_32FC1);
    
    float label_count = 2; // start labelling at 2 as 1,0 already in binary img

    for(int y=0; y < binary.rows; y++) {
      for(int x=0; x < binary.cols; x++) {

	if(int(label_img.at<float>(y,x)) != 1) {
	  continue;
	}

	Rect rect;
	floodFill(label_img, Point(x,y), Scalar(label_count), &rect);
	
	vector <Point2i> blob;
	blob.clear();

	int label_area=0;
	
	for(int i=rect.y; i < (rect.y+rect.height); i++) {
	  for(int j=rect.x; j < (rect.x+rect.width); j++) {
	    if(int(label_img.at<float>(i,j)) != label_count) {continue;}
	    blob.push_back(Point2i(j,i));

	    label_area++;
	  }
	}

	label_areas.push_back(label_area);
	blobs.push_back(blob);
	

	//check-max
	if (label_area>max_area)
	  {

	    max_area=label_area; 
	    max_label=label_count-2;
	  }
	
	label_count++;
      }
    }

    *max_idx_out = max_label;
    *max_area_out = max_area;
}





  //Finds color blob centroids
  bool get_centroid(Point2f*  out_blob_centroid, double area_thresh, bool isGreen, Point3f* centroid_3d)
  {
  
    new_comparison_mahalanobis(isGreen);
 

    int max_idx = -1;
    int max_area = 0;

    //compute area and centroids
    vector<vector<Point2i> > blobs;
   
    find_blobs(bin_img, blobs, &max_idx, &max_area);
  
    //iterate through all contours and find one with biggest area
    int num_blobs = blobs[max_idx].size();

    //debug
    /*Vec3b color(0,0,255);
    for( size_t idx = 0; idx < num_blobs; idx++ )
      {
	img_of_interest.at<Vec3b>(blobs[max_idx][idx].y, blobs[max_idx][idx].x) = color;
	}*/
    
  
    /* //debug - show image
    namedWindow("Binary Image", 1);
    imshow("Binary Image", img_of_interest);
    waitKey(0);
    return true;*/

    if ((max_idx>-1) && (max_area>area_thresh))
      {
	accumulator_set<double, features<tag::median> > x_acc, y_acc, z_acc;
	accumulator_set<int, features<tag::median> > u_acc, v_acc;



	
    	if(cur_pc && 
	   (fabs((cur_pc->header.stamp - last_img_time).toSec()) < 100.2)) 
	  {
	    for(size_t i = 0; i < blobs[max_idx].size(); i++) 
	      {
		Point2i pt_2d = blobs[max_idx][i];
		u_acc(pt_2d.x); v_acc(pt_2d.y);
		PRGB pt = cur_pc->at(pt_2d.x, pt_2d.y);
		
		if(pt.x == pt.x && pt.y == pt.y && pt.z == pt.z) {
		  x_acc(pt.x); y_acc(pt.y); z_acc(pt.z); 
		}
	      }
        
	    Point3f temp_3d;
	    temp_3d.x=median(x_acc); temp_3d.y=median(y_acc), temp_3d.z=median(z_acc);
	    Point2i temp_2d;
	    temp_2d.x=median(u_acc); temp_2d.y=median(v_acc);

	    *out_blob_centroid = temp_2d;
	    *centroid_3d = temp_3d;
	    return true;
	  } else {return false;}
      }
    else
      {
      //debug
      //cout<<"\nNot found blob\n";
      return false;
      }
    
  }


  void draw_centroid(Mat img_of_interest, Point2f in_centroid, const Scalar cross_color)
  {

    int line_thickness = 5;
    Point centroid_1, centroid_2;
    centroid_1 = centroid_2 = in_centroid;
    centroid_1.x = std::max(0.f, in_centroid.x-10);
    centroid_2.x = std::min(float(img_of_interest.cols), in_centroid.x+10);
    line(img_of_interest, centroid_1, centroid_2, cross_color, line_thickness);

    centroid_1 = centroid_2 = in_centroid;
    centroid_1.y = std::max(0.f, in_centroid.y-10);
    centroid_2.y = std::min(float(img_of_interest.cols), in_centroid.y+10);
    line(img_of_interest, centroid_1, centroid_2, cross_color, line_thickness);

  }


  //publish 3d position of centroid using input publisher
  //it publishes (inf,inf,inf) in case hand was not found
  void publish_centroid(Point3f centroid, bool found, ros::Publisher pub)
  {
  
    geometry_msgs::Point msg;
  
    if(found)
      {
 	msg.x=centroid.x;msg.y=centroid.y;msg.z=centroid.z;
      }else
      {
	double inf = std::numeric_limits<double>::infinity();
	msg.x=inf; msg.y=inf; msg.z=inf;
      }
  
    //publish message
    pub.publish(msg);
  
    //debug
    //ROS_INFO("Transmitted a hand %d");

  }

  //publish 2d position of centroid using argument pub
  //it publishes (u,v,inf) as 2d point
  //and (inf,inf,inf) in case hand not found
  void publish_centroid(Point2f centroid, bool found, ros::Publisher pub)
  {
  
    geometry_msgs::Point msg;
  
    if(found)
      {
 	msg.x=centroid.x;msg.y=centroid.y;msg.z=std::numeric_limits<double>::infinity();
      }else
      {
	double inf = std::numeric_limits<double>::infinity();
	msg.x=inf; msg.y=inf; msg.z=inf;
      }
  
    //publish message
    pub.publish(msg);
  
    //debug
    //ROS_INFO("Transmitted a 2d hand %d");

  }


  void frame_process()
  {

   
    if(!img_of_interest.empty())
      {

	hsv_get();

	//debug
	//debug - start counting time
	//double t = (double)getTickCount();
  
	//parameters for comparison
	Point2f blue_centroid, green_centroid;
	Point3f blue_centroid_3, green_centroid_3;
	  
	//currently only area used
	double blue_area_thresh = 15.0, green_area_thresh = 15.0;

	//call function for centroid computation
	bool found_blue = get_centroid( &blue_centroid, blue_area_thresh, false, &blue_centroid_3);
	bool found_green = get_centroid( &green_centroid, green_area_thresh,true, &green_centroid_3);
	  

	publish_centroid(blue_centroid_3, found_blue, purple_hand);
	publish_centroid(green_centroid_3, found_green, green_hand);
	publish_centroid(blue_centroid, found_blue, purple_hand_2d);	
	publish_centroid(green_centroid, found_green, green_hand_2d);
	//debug - stop counting time
	//t = ((double)getTickCount() - t)/getTickFrequency();
  

	//debug- draw a cross to denote centroids
	/*Scalar cross_color1(0, 235, 0);
	Scalar cross_color2(0, 0, 245);
		if(found_blue){draw_centroid(img_of_interest, blue_centroid, cross_color1);}
	else{cout<<"\nCouldn't find blue hand\n";}
	if(found_green){draw_centroid(img_of_interest, green_centroid, cross_color2);}
	else{cout<<"\nCouldn't find green hand\n";}
	//debug

	imshow( "Track Hands", img_of_interest );
	int delay=5;
	char c = (char)cvWaitKey(delay);
	if (c == 27) exit(-1);*/
	  
	//debug - display execution time
	//cout << "\nTimes passed in seconds: " << t << endl;
	  
      }

  }

  //callback - grabs frame from kinect and processes it
  void kin_rgb_fram_sub(const sensor_msgs::ImageConstPtr& msg)
  {
    
    last_img_time = msg->header.stamp;
    try
      {
	cv_ptr = cv_bridge::toCvCopy(msg,  sensor_msgs::image_encodings::BGR8);
      }
    catch (cv_bridge::Exception& e)
      {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	return;
      }
    
    //    namedWindow("show",1);

    img_of_interest = cv_ptr->image;
    
    //debug
    //cout << "\nNo.of channels"<< img_of_interest.channels();
    
    //debug
    //imshow("show", glob_img_of_interest);
    //waitKey(0);
    
    //process
    frame_process();
  }

  //callback - grabs point cloud
  void kin_pc_sub(sensor_msgs::PointCloud2::ConstPtr pc_msg) 
  {
    pcl::fromROSMsg(*pc_msg, *cur_pc);
  }

};


//MAIN
int main(int argc, char **argv)
{

  
  ros::init(argc, argv, "hand_tracker"); 
  handsTracker begin_track;

  
  ros::spin();

      

  return 0;
}

