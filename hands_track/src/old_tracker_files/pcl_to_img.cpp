void kin_rgb_fram_sub(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
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

    Mat img_of_interest = cv_ptr->image;
    //debug
    //imshow("show", glob_img_of_interest);
    //waitKey(0);
    
    //process
    frame_process(img_of_interest);
}

//callback - grabs point cloud
void kin_pc_sub(sensor_msgs::PointCloud2::ConstPtr pc_msg) {
	pcl::fromROSMsg(*pc_msg, *cur_pc);
  
  int tot_rows = 480, tot_cols = 480;

  Mat img_of_interest = Mat::zeros(tot_rows, tot_cols, CV_8UC3);
  
  if (cur_pc){

  for (int r=0; r<480; r++){
  	Vec3b* Mi = img_of_interest.ptr<Vec3b>(r);
  	//cout<<cur_pc->width;
	//cout<<"reach:"<<r<<endl;
  	for(int c=0; c<640; c++){
  	//cout<<"col:"<<c<<endl;
  		PRGB pt = cur_pc->at(c, r);		
  		
  		if(pt.x == pt.x && pt.y == pt.y && pt.z == pt.z) {
  			Mi[c][0] = ((uint8_t*)&pt.rgb)[0];
  			Mi[c][1] = ((uint8_t*)&pt.rgb)[1];
  			Mi[c][2] = ((uint8_t*)&pt.rgb)[2];
      		}else{
		  Mi[c][0] = 0;
		  Mi[c][1] = 0;
		  Mi[c][2] = 0;
      		}
	}
  }
  }
 
/* namedWindow("PointCloud",1);
 imshow("PointCloud", img_of_interest);	
 waitKey(0);*/
 
 frame_process(img_of_interest);
}
