#include <iostream>
#include<fstream>
#include<string>
#include<vector>
#include<queue>
#include<stdlib.h>
#include<sstream>
#include<iterator>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <tf/transform_listener.h>
#include<std_msgs/UInt8MultiArray.h>

#include <math.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <project_simulation/AlvarMarker.h>
#include <project_simulation/AlvarMarkers.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define EPSILON 8.8817841970012523e-016

using namespace std;

//GLOBAL VARIABLES
int PUB_RATE=20;
boost::mt19937 rng;
boost::normal_distribution<> nd(0.0, 1.0);
boost::variate_generator<boost::mt19937&, 
			 boost::normal_distribution<> > NORM_GEN(rng, nd);

class Task{

private:
  struct Step{
    string step_name;
    double duration_mean, duration_std;
  };

  struct Bin{
    size_t bin_id;
    string bin_name;
    queue<Step> step_list;
    //in case I want to wait before reaching into bin
    double wait_before;
  };
  
  vector<Bin> bin_list;
  

  void read_into_bin(string file_name, Bin* read_bin)
  {
    string file_path = "src/tasks/bins/";
    string file = file_path + file_name + ".txt";
    (*read_bin).bin_name = file_name;
    
    ifstream bin_file;
    bin_file.open(file.c_str());
    if(!bin_file.is_open())
      {
	cout << "\n Failed to open bin file "<< file << endl;
	exit(-1);
      }else
      {
	string cur_line;
	int type_counter=0;
	getline(bin_file, cur_line);

	while(bin_file.good())
	  {
	    if(cur_line[0]=='#'){++type_counter;}
	    else
	      {
		if(type_counter==1)
		  {
		    (*read_bin).bin_id = size_t(atoi(cur_line.c_str()));
		  }
		else if(type_counter==2)
		  {
		    istringstream iss(cur_line);
		    vector<string> tokens;
		    copy(istream_iterator<string>(iss),
			 istream_iterator<string>(),
			 back_inserter<vector<string> >(tokens));
		  
		    if(tokens.size()!=3){cout<<"Prob reading file "<<file; exit(-1);}
		    Step cur_step;
		    cur_step.step_name = tokens[0];
		    cur_step.duration_mean = atof(tokens[1].c_str());
		    cur_step.duration_std = atof(tokens[2].c_str());
		    
		    (*read_bin).step_list.push(cur_step);
		  }

	      }
	    getline(bin_file, cur_line);
	    
	  }
      }

  }
  
  void read_task(string file)
  {
    string file_path = "src/tasks/";
    ifstream task_file;
    string file_n_path = file_path+file;
    task_file.open(file_n_path.c_str());
    
    if(!task_file.is_open())
      {
	cout<<"\nFailed to open task file "<<file_path+file<<endl;
	exit(-1);
      }else
      {
	string cur_line;
	getline(task_file, cur_line);
	
	while(task_file.good())
	  {

	    if(cur_line[0]!='#')
	      {
		Bin cur_bin;
		read_into_bin(cur_line, &cur_bin);
		bin_list.push_back(cur_bin);
	      }
	    getline(task_file, cur_line);
	    
	  }
      }
  }

public:
  
  Task()
  {
    bin_list.clear();
  }

  void read_file(string file)  {read_task(file);}

  //returns false if no step left
  bool get_next_step(size_t* bin_no, double *dur_mean, double *dur_std, size_t* bin_id_to_go)//, string* b_nam, string* s_nam)
  {
    size_t cur_bin_no = *bin_no;
    Bin* cur_bin = &bin_list[cur_bin_no];
    
    //check if any steps left
    if ((*cur_bin).step_list.empty())
      {
	++cur_bin_no;
	if(bin_list.size()>cur_bin_no)
	  {
	    cur_bin = &bin_list[cur_bin_no];
	  }else
	  {
	    //tasks complete 
	    return false;
	  }
      }
    

    Step cur_step = (*cur_bin).step_list.front();
    
    (*cur_bin).step_list.pop();
    
    //set variables to return
    *bin_id_to_go = (*cur_bin).bin_id;
    *bin_no = cur_bin_no;
    *dur_mean = cur_step.duration_mean;
    *dur_std = cur_step.duration_std;
    
    //there is step to complete
    return true;
  }


};


class handSim
{
private:
  Task to_perform;
  size_t bin_no;
  //  double duration_m, duration_s;
  
  vector<double> lh_rest;
  vector<double> rh_rest;
  vector<double> fix_hand_orient;
  
  vector<double> lh_cur;
  vector<double> rh_cur;

  ros::NodeHandle nh;
  ros::Publisher lh_pose;
  ros::Publisher rh_pose;
  ros::Publisher viz_pub;

  ros::Subscriber ar_poses;
  ros::Subscriber workspace_bins;
  
  project_simulation::AlvarMarkers cur_markers;
  
  string frame_of_reference;
  string ar_pose_frame;

  long total_time_steps;
  long wait_time_steps;
  
  struct Bin_n_Loc
  {
    size_t id;
    vector<double> location;
  };
  vector<Bin_n_Loc> cur_bin_locations;
  vector<Bin_n_Loc> temp_bin_locations;

  double transform_translate[3];
  double transform_rot_mat[16];
  
  double motion_mean, motion_std;
  
  vector<size_t> workspace_bin_list;

  //noise
  //this represents noise in the tracker
  double pub_noise_dev;

  //are hands at rest positions
  bool is_lh_rest, is_rh_rest;


public:

  handSim(string task_name)
  {
    //load tasks
    task_name += ".txt";
    to_perform.read_file(task_name);

    //set workspace
    workspace_bin_list.clear();
    
    //set noise
    pub_noise_dev = 0.005;
    
    //initialize rest positions
    double init_pos_l[]= {-0.0813459, 0.258325,1.86598};
    double init_pos_r[]= {-0.1813459, 0.258325,1.86598};
    lh_rest.assign(&init_pos_l[0], &init_pos_l[0]+3);
    rh_rest.assign(&init_pos_r[0], &init_pos_r[0]+3);
    
    //initially, hands are at rest position
    lh_cur = lh_rest;
    rh_cur = rh_rest;
    is_lh_rest = true;
    is_rh_rest = true;

    //initialize fixed hands orientation
    double init_pose[] = {1.0, 0.0, 0.0, 0.0};
    fix_hand_orient.assign( &init_pose[0], &init_pose[0]+4);
    
    total_time_steps = 0;
    wait_time_steps = 0;
    double z =time_to_wait();


    //workspace
    vector<vector <double> > work_locs;

    //frame
    frame_of_reference = "/kinect0_rgb_optical_frame";
    ar_pose_frame = "/lifecam1_optical_frame";
    
    //time for moving hand to location or back
    motion_mean = 1.0;
    motion_std = 0.01;

    
    //clear bin vectors
    cur_bin_locations.clear();
    temp_bin_locations.clear();
    
    //ros-stuff
    lh_pose = nh.advertise<geometry_msgs::PoseStamped>("left_hand",1);
    rh_pose = nh.advertise<geometry_msgs::PoseStamped>("right_hand",1);
    viz_pub = nh.advertise<visualization_msgs::MarkerArray>("hands_viz", 1);
    
    ar_poses = nh.subscribe("ar_pose_marker", 0, &handSim::read_ar, this);  
    workspace_bins = nh.subscribe("workspace_bins", 0, &handSim::listen_workspace, this);
    
    //store transform
    tf::TransformListener tf_cam_to_kin;  
    try
      {
	ros::Time now = ros::Time(0);
	tf_cam_to_kin.waitForTransform(frame_of_reference, ar_pose_frame,
				        now,
				       ros::Duration(5.0));
	
	tf::StampedTransform temp_pos;
	tf_cam_to_kin.lookupTransform(frame_of_reference, ar_pose_frame, 
				       now,
				      temp_pos);
	double cam_to_kin_translate[] = {temp_pos.getOrigin().x(), 
					 temp_pos.getOrigin().y(),
					 temp_pos.getOrigin().z()};
	
	int N = (sizeof(cam_to_kin_translate)/sizeof(double));
	set_arr_equal(cam_to_kin_translate, transform_translate, N);
	
	double* cam_to_kin_quaternion;
	double temp[] = {temp_pos.getRotation().w(),
			 temp_pos.getRotation().x(), 
			 temp_pos.getRotation().y(),
			 temp_pos.getRotation().z()
			 };
	
	cam_to_kin_quaternion = temp;
	double* rot_mat = transform_rot_mat;
	quaternion_matrix(cam_to_kin_quaternion, rot_mat);
	
      }

    catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
      cout<< "Can't get transformation"<<endl;
      exit(-1);
    }
  }
  
  //set values of one array to another
  void set_arr_equal(double from[], double to[], int N)
  {
    if (sizeof(from) != sizeof(to))
      {
	cout<< "Cannot set equal array, sizes do not match";
	exit(-1);
      }
    
    //int N = sizeof(from)/sizeof(double);
    //debug
    //cout<<"N = "<<N<<endl;

    for(int i=0; i<N; i++)
      {
	to[i] = from[i];
	
	//debug
	//cout<<to[i]<<'='<<from[i]<<'?'<<endl;
      }
    
  }
  
  geometry_msgs::PoseStamped transform_view(geometry_msgs::PoseStamped to_change)
  {
    geometry_msgs::PoseStamped transformed;
    transformed = to_change;
    //rotate
    double homo_pos_vec[4] = {to_change.pose.position.x,
			      to_change.pose.position.y,
			      to_change.pose.position.z,
			      1.0};
    size_t size_mat[]={4,4}; size_t size_vec[]={4,1};
    double homo_vec_out[4];

    /*//debug
    cout<<"time for mat"<<endl;
    for (int i=0; i<4; i++){
      for(int j=0; j<4;j++)
	{cout<<transform_rot_mat[4*i+j]<<"     ";}
	cout<<endl;}*/
    mat_mul(transform_rot_mat, size_mat, homo_pos_vec, size_vec, homo_vec_out);
    
    //convert back from homogenous and translate
    trans_homo_vec(homo_vec_out);
    
    transformed.pose.position.x = homo_vec_out[0];
    transformed.pose.position.y = homo_vec_out[1];
    transformed.pose.position.z = homo_vec_out[2];

    transformed.header.frame_id = frame_of_reference;
    
    /*//debug
    cout<<"OK, got here"<<endl;
    cout<<to_change<<endl;
    cout<<transformed<<endl;
    char a='y';
    while(a!='n')
      a=getchar();
      exit(-1);*/

    return transformed;
  }
  
  void mat_mul(double mat_one[], size_t mat_one_size[2], double mat_two[], size_t mat_two_size[2], double mat_out[])
  { 
    if (mat_one_size[1]==mat_two_size[0])
      {
	size_t r1 = mat_one_size[0], rc = mat_one_size[1], c2 = mat_two_size[1];

	for (size_t i=0; i<r1; ++i)
	  {
	    for (size_t j=0; j<c2; ++j)
	      {
		mat_out[c2*i + j] = 0.0;
		
		for (size_t k=0; k<rc; ++k)
		  {
		    mat_out[c2*i + j] += mat_one[rc*i + k] * mat_two[c2*k + j]; 
		  }
	      }
	  }
      }else{cout<< "Can't multiple! Matrix size don't match" << endl;}
  }

  void trans_homo_vec(double homo_vec[])
  {
    //convert to regular vector
    homo_vec[0] /= homo_vec[3];
    homo_vec[1] /= homo_vec[3];
    homo_vec[2] /= homo_vec[3];
    homo_vec[3] /= homo_vec[3];
    
    //translate
    homo_vec[0] += transform_translate[0];
    homo_vec[1] += transform_translate[1];
    homo_vec[2] += transform_translate[2];

  }

  void read_ar(project_simulation::AlvarMarkers msg)
  {
    geometry_msgs::Point temp_pos;
    geometry_msgs::PoseStamped temp_pose;
    Bin_n_Loc temp_bin;

    temp_bin_locations.clear();

    for (size_t i=0; i<msg.markers.size(); i++)
      {
	//to transform ar-tags frame of reference
	temp_pose = transform_view(msg.markers[i].pose);
	temp_pos = temp_pose.pose.position;

	temp_bin.id = msg.markers[i].id;
	
	temp_bin.location.clear();
	
	temp_bin.location.push_back(temp_pos.x);
	temp_bin.location.push_back(temp_pos.y);
	temp_bin.location.push_back(temp_pos.z);
	
	temp_bin_locations.push_back(temp_bin);
      }
    
    cur_bin_locations = temp_bin_locations;
    
  }


  void listen_workspace(std_msgs::UInt8MultiArray work_bins)
  {
    size_t n_bins = work_bins.data.size();
    vector<size_t> temp_work;
    temp_work.clear();

    //debug
    //cout<<endl<<"Work bins: (";

    for (size_t i=0; i<n_bins; ++i)
      {
	temp_work.push_back(work_bins.data[i]);
	
	//debug
	//cout<<temp_work[i]<<',';
      }
    
    //debug
    //cout<<')'<<endl;

    workspace_bin_list.clear();
    workspace_bin_list = temp_work;    
  }
  
  //returns t/f if bin is/not in human workspace
  bool bin_in_workspace(size_t bin_id)
  {
    size_t n_bins = workspace_bin_list.size();
    bool bin_found = false;

    for (size_t i=0; i<n_bins; i++)
      {
	if(bin_id == workspace_bin_list[i])
	  {
	    bin_found = true;
	    break;
	  } 
      }

    return bin_found;

  }

  //does tasks in list
  void pub_hands()
  {
    
    size_t cur_bin_id, cur_bin_no=0;

    double duration_m, duration_s;

    //pop next task
    while(to_perform.get_next_step(&cur_bin_no, &duration_m, &duration_s, &cur_bin_id))
      {	
	//debug
	cout<< "Task - Bin-"<<cur_bin_id<<" ; mean std = "<<duration_m<<' '<<duration_s<<endl;
	
	perform_task(cur_bin_id, duration_m, duration_s);
      }
    
    //after tasks complete, just publish the rest position
    pub_hands_rest();
	        
  }
  
  //pick out of given bin in correct time
  void perform_task(size_t cur_bin, double dur_m, double dur_s)
  {
    vector<double> cur_bin_loc;
    
    //if bin to pick from unavailable
    if(!bin_in_position(cur_bin))
      {
	wait_for_bin(cur_bin);
	//always at rest after waiting
	is_lh_rest =true;
	is_rh_rest = true;
      }

    //after it becomes available
    //sample gaussian to determine time to perform task
    double time_task_rem = samp_gauss_pos(dur_m, dur_s);

    //wait at current positions- for hand after touched a bin or initially 
    //before reaching into the first bin
    double waiting_time = time_to_wait();
    time_task_rem -= waiting_time;
    wait_at_location(waiting_time);
    //add total time
    total_time_steps += (waiting_time/PUB_RATE);
    
    //its more of a debugging measure- one hand at least be at rest
    cur_bin_loc = find_bin_pos(cur_bin);
    if(!is_lh_rest && !is_rh_rest)
      {
	cout<<"Both hands not at rest"<<endl;
	exit(-1);
      }
    //should happen at the beginning of a task or after spending some time 
    //waiting
    else if(is_rh_rest && is_lh_rest)
      {
	//just go with righty
	double to_bin_time = samp_gauss_pos(motion_mean, motion_std);
	
	//if sampled time > time remaining then spend remaining time on moving
	if (to_bin_time>=time_task_rem){to_bin_time=time_task_rem;}
	else{
	  //wait before performing task
	  wait_at_location(time_task_rem-to_bin_time);
	}
	
	//then move right to bin
	move_to_loc(false, true, lh_cur, cur_bin_loc, to_bin_time);
	//right-hand was moved
	is_rh_rest = false;
      }
    else if(is_rh_rest)
      {
	double to_bin_time = samp_gauss_pos(motion_mean, motion_std);
	double to_rest_time = samp_gauss_pos(motion_mean, motion_std);
	double motion_time = to_bin_time+to_rest_time;
	double wait_bw_bins;
	//if sampled time > time remaining then spend remaining time on moving
	if (motion_time>=time_task_rem)
	  {
	    //scale both times for motion to just fit within the actual time
	    double time_scale = time_task_rem/motion_time;
	    to_bin_time *= time_scale;
	    to_rest_time *= time_scale;
	    wait_bw_bins = 0;
	  }else{wait_bw_bins = time_task_rem-motion_time;}

	//first bring left to rest
	move_to_loc(true, false, lh_rest, rh_cur, to_rest_time);
	is_lh_rest = true;
	
	//use the parts retrieved
	wait_at_location(wait_bw_bins);
	
	//then move right to bin
	move_to_loc(false, true, lh_cur, cur_bin_loc, to_bin_time);
	is_rh_rest = false;
      }

    else if(is_lh_rest)
      {

	double to_bin_time = samp_gauss_pos(motion_mean, motion_std);
	double to_rest_time = samp_gauss_pos(motion_mean, motion_std);
	double motion_time = to_bin_time+to_rest_time;
	double wait_bw_bins;
	//if sampled time > time remaining then spend remaining time on moving
	if (motion_time>=time_task_rem)
	  {
	    //scale both times for motion to just fit within the actual time
	    double time_scale = time_task_rem/motion_time;
	    to_bin_time *= time_scale;
	    to_rest_time *= time_scale;
	    wait_bw_bins = 0;
	  }else{wait_bw_bins = time_task_rem-motion_time;}

	//first bring right to rest
	move_to_loc(false, true, lh_cur, rh_rest, to_rest_time);
	is_rh_rest = true;

	//use the parts retrieved
	wait_at_location(wait_bw_bins);
	
	//then move left to bin
	move_to_loc(true, false, cur_bin_loc, rh_cur, to_bin_time);
	is_lh_rest = false;

      }


  }
  
  //calculate euclidean dist between two vectors
  double calc_euclid_dist(vector<double> vec_one, vector<double> vec_two)
  {
    if (vec_one.size()!=vec_two.size())
      {
	cout<< "Dimensionality mismatch: can't compute euclidean"<<endl; 
	exit(-1);
      }
    
    else
      {
	double distance=0.0;
	for (size_t i=0; i<vec_one.size();++i)
	  {
	    distance += pow(vec_one[i]-vec_two[i],2);
	  }
	return sqrt(distance);
      }

  }

  //return position of queried bin
  vector<double> find_bin_pos(size_t bin_to_find)
  {
    ros::spinOnce();
    
    for (size_t i=0; i<cur_bin_locations.size(); i++)
      {
	if(cur_bin_locations[i].id == bin_to_find)
	  {
	    return cur_bin_locations[i].location;
	  }
      }
    
    //reaching here means bin not found
    cout<< "Error:Bin not found"<<endl ;
    exit(-1);

  }

  //wait for bin to arrive
  void wait_for_bin(size_t bin_to_chk)
  {
    ros::Rate loop_rate(PUB_RATE);
    
    do
      {
	if (!(is_lh_rest && is_rh_rest))
	  {
	    double move_time = samp_gauss_pos(motion_mean, motion_std);
	    move_to_loc(!is_lh_rest, !is_rh_rest, 
		       lh_rest, rh_rest, move_time);
	    is_lh_rest = true; is_rh_rest=true;
	    wait_time_steps += move_time*PUB_RATE;
	    total_time_steps += move_time*PUB_RATE;
	  }
	publish_cur_hand_pos();
	ros::spinOnce();
	loop_rate.sleep();
	//increment time
	++wait_time_steps;
	++total_time_steps;
    
	//debug
	cout<<"Need Bin: "<<bin_to_chk<<endl;
      }while(ros::ok() && !bin_in_position(bin_to_chk));
  }

  //check if bin in position
  bool bin_in_position(size_t chk_bin)
  {
    ros::spinOnce();
    
    for (size_t i=0; i<cur_bin_locations.size(); i++)
      {
	if(cur_bin_locations[i].id == chk_bin)
	  {
	    if(bin_in_workspace(chk_bin)){return true;}
	    else{break;}
	  }
      }
    return false;
  }
  
  //return only positive samples
  double samp_gauss_pos(double mean, double std_dev)
  {
    double sample = mean + std_dev*NORM_GEN();
    while(sample<=0){sample = mean + std_dev*NORM_GEN();}
    return sample;
  }

  //return sample
  double samp_gauss(double mean, double std_dev)
  {
    double sample = mean + std_dev*NORM_GEN();
    return sample;
  }
  
  //publish present hand positions
  void publish_cur_hand_pos()
  {
    geometry_msgs::PoseStamped msg_l, msg_r;

    msg_l.header.frame_id = frame_of_reference;
    msg_r.header.frame_id = frame_of_reference;
	
    vec_to_position(msg_l.pose.position, lh_cur);
    vec_to_position(msg_r.pose.position, rh_cur);

    vec_to_orientation(msg_l.pose.orientation, fix_hand_orient);
    vec_to_orientation(msg_r.pose.orientation, fix_hand_orient);
    
    add_tracker_noise(msg_l);
    add_tracker_noise(msg_r);
    
    lh_pose.publish(msg_l);
    rh_pose.publish(msg_r);

    //publish visual markers
    pub_viz_marker(msg_l, msg_r);
  }
  
  void add_tracker_noise(geometry_msgs::PoseStamped &message)
  {
    message.pose.position.x += samp_gauss(0.0, pub_noise_dev);
    message.pose.position.y += samp_gauss(0.0, pub_noise_dev);
    message.pose.position.z += samp_gauss(0.0, pub_noise_dev);
  }

  //publish a visualization marker at given location
  void pub_viz_marker(geometry_msgs::PoseStamped lh_pose, geometry_msgs::PoseStamped rh_pose)
  {
    //left hand
    visualization_msgs::Marker lh_marker;
    lh_marker.header.frame_id = frame_of_reference;
    lh_marker.header.stamp = ros::Time();
    lh_marker.ns = "hands";
    //0-left 1-right
    lh_marker.id = 0;
    
    lh_marker.type = visualization_msgs::Marker::SPHERE;
    lh_marker.action = visualization_msgs::Marker::ADD;
    lh_marker.pose = lh_pose.pose;
    lh_marker.scale.x = 0.05;
    lh_marker.scale.y =0.05;
    lh_marker.scale.z = 0.05;
    lh_marker.color.a = 1.0;
    lh_marker.color.r = 0.0;
    lh_marker.color.g = 0.0;
    lh_marker.color.b = 1.0;

    //right hand
    visualization_msgs::Marker rh_marker;
    rh_marker.header.frame_id = frame_of_reference;
    rh_marker.header.stamp = ros::Time();
    rh_marker.ns = "hands";
    //0-left 1-right
    rh_marker.id = 1;
    
    rh_marker.type = visualization_msgs::Marker::SPHERE;
    rh_marker.action = visualization_msgs::Marker::ADD;
    rh_marker.pose = rh_pose.pose;
    rh_marker.scale.x = 0.05;
    rh_marker.scale.y = 0.05;
    rh_marker.scale.z = 0.05;
    rh_marker.color.a = 1.0;
    rh_marker.color.r = 1.0;
    rh_marker.color.g = 0.0;
    rh_marker.color.b = 1.0;

    visualization_msgs::MarkerArray viz_markers;
    viz_markers.markers.push_back(lh_marker);
    viz_markers.markers.push_back(rh_marker);

    viz_pub.publish( viz_markers);
  }

  //brings hands to rest positions and then task is completed
  void pub_hands_rest()
  {
    ros::Rate loop_rate(PUB_RATE);
	    
    //TODO: make use of publish pose function to remove redundancy
    geometry_msgs::PoseStamped msg_l, msg_r;

    msg_l.header.frame_id = frame_of_reference;
    msg_r.header.frame_id = frame_of_reference;
	
    vec_to_position(msg_l.pose.position, lh_rest);
    vec_to_position(msg_r.pose.position, rh_rest);
	
    vec_to_orientation(msg_l.pose.orientation, fix_hand_orient);
    vec_to_orientation(msg_r.pose.orientation, fix_hand_orient);
    
    double total_time = double(total_time_steps)/double(PUB_RATE);
    double wait_time_total = double(wait_time_steps)/double(PUB_RATE);
    
    if(!(is_lh_rest && is_rh_rest))
      {
	double to_wait = time_to_wait();
	wait_at_location(to_wait);
	double time_to_move = 1.0;
	move_to_loc(!is_lh_rest, !is_rh_rest, lh_rest, rh_rest, time_to_move);
	is_lh_rest=true; is_rh_rest=true;
	
	//wait sometime after tasks complete
	
      }
    

    cout<<"TASKS COMPLETE"<<endl;
    cout<<"Total, wait  "<<total_time<<", "<<wait_time_total<<endl;
  }
  
  //publish hands for given time in their current location
  void wait_at_location(double wait_time)
  {
    long tot_steps = floor(wait_time*double(PUB_RATE));
    long step_no=0;
    ros::Rate loop_rate(PUB_RATE);
    
    while(step_no < tot_steps && ros::ok())
      {
	publish_cur_hand_pos();
	++step_no;
	loop_rate.sleep();
	//increment total time
	++total_time_steps;
      }
  }


  //interpolate to given location in specified time
  void move_to_loc(bool move_left, bool move_right, vector<double> left_pos, 
		   vector<double> right_pos, double move_time)
  {
    long tot_steps = floor(move_time * PUB_RATE);
    ros::Rate loop_rate(PUB_RATE);
    
    //compute step differentials
    //left
    double lh_x_diff = (left_pos[0]-lh_cur[0])/double(tot_steps);
    double lh_y_diff = (left_pos[1]-lh_cur[1])/double(tot_steps);
    double lh_z_diff = (left_pos[2]-lh_cur[2])/double(tot_steps);

    //right
    double rh_x_diff = (right_pos[0]-rh_cur[0])/double(tot_steps);
    double rh_y_diff = (right_pos[1]-rh_cur[1])/double(tot_steps);
    double rh_z_diff = (right_pos[2]-rh_cur[2])/double(tot_steps);
    
    long steps=0;
    while(steps<tot_steps && ros::ok())
      {
	if(move_left)
	  {
	    lh_cur[0] += lh_x_diff;
	    lh_cur[1] += lh_y_diff;
	    lh_cur[2] += lh_z_diff;
	    
	    //TODO: make sure that the hand doesn't move ahead of its 
	    //intended destination by some if condition here

	  }
	
	if(move_right)
	  {
	    rh_cur[0] += rh_x_diff;
	    rh_cur[1] += rh_y_diff;
	    rh_cur[2] += rh_z_diff;
	  }
	
	publish_cur_hand_pos();
	++steps;
	
	//increment time
	
	loop_rate.sleep();
	
	
      }

    total_time_steps += tot_steps;

    //ensure at the correct positions after end of interpolation
    if(move_left){lh_cur = left_pos;}
    if(move_right{rh_cur = right_pos;}
  }

  /*bool is_lh_rest()
  {
    for (size_t i=0; i<lh_cur.size(); i++)
      if (!(lh_cur[i] == lh_rest[i]))
	return false;
    
    //
    return true;
  }

  bool is_rh_rest()
  {
    for (size_t i=0; i<rh_cur.size(); i++)
      if (!(rh_cur[i] == rh_rest[i]))
	return false;
    
    //
    return true;
    }*/

  //at bin
  double time_to_wait()
  {
    //wait a half-second for the moment
    return 0.1;
  }

  void vec_to_position(geometry_msgs::Point &out_pos, vector<double> in_pos)
  {
    if(in_pos.size()==3)
      {
	out_pos.x = in_pos[0];
	out_pos.y = in_pos[1];
	out_pos.z = in_pos[2];
      }else
      {
	cout<<"Can-not convert vector to point(x,y,z)"<<endl;
	exit(-1);
      }
  }
  
  void vec_to_orientation(geometry_msgs::Quaternion &out_orient, 
			  vector<double> in_orient)
  {
    if(in_orient.size()==4)
      {
	out_orient.x = in_orient[0];
	out_orient.y = in_orient[1];
	out_orient.z = in_orient[2];
	out_orient.w = in_orient[3];
      }else
      {
	cout<<"Can-not convert vector to orientation(x,y,z,w)"<<endl;
	exit(-1);
      }

  }
  
  /*
    Quaternion to rotation matrix.
    :Author:
    `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_
  */
  void quaternion_matrix(
			double *quat,    /* double[4]  */
			double *matrix)  /* double[16] */
  {
    double *M = matrix;
    double *q = quat;
    double n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);

    if (n < EPSILON) 
      {
	/* return identity matrix */
	memset(M, 0, 16*sizeof(double));
	M[0] = M[5] = M[10] = M[15] = 1.0;
      }else
      {
	q[0] /= n;
	q[1] /= n;
	q[2] /= n;
	q[3] /= n;
	{
	  double x2 = q[1]+q[1];
	  double y2 = q[2]+q[2];
	  double z2 = q[3]+q[3];
	  {
	    double xx2 = q[1]*x2;
	    double yy2 = q[2]*y2;
	    double zz2 = q[3]*z2;
	    M[0]  = 1.0 - yy2 - zz2;
	    M[5]  = 1.0 - xx2 - zz2;
	    M[10] = 1.0 - xx2 - yy2;
	  }{
	    double yz2 = q[2]*z2;
	    double wx2 = q[0]*x2;
	    M[6] = yz2 - wx2;
	    M[9] = yz2 + wx2;
	  }{
	    double xy2 = q[1]*y2;
	    double wz2 = q[0]*z2;
	    M[1] = xy2 - wz2;
	    M[4] = xy2 + wz2;
	  }{
	    double xz2 = q[1]*z2;
	    double wy2 = q[0]*y2;
	    M[8] = xz2 - wy2;
	    M[2] = xz2 + wy2;
	  }
	  M[3] = M[7] = M[11] = M[12] = M[13] = M[14] = 0.0;
	  M[15] = 1.0;
	}
      }
    return ;
  }  

};



int main(int argc, char** argv)
{
  ros::init(argc, argv, "hand_simulator");
  char do_another='n';

  do{
    char correct = 'n';
    string task;
    while(correct != 'y')
      {
	cout<<"Which task? helicopter, airplane1, or airplane2"<<endl;
	cin>> task;
	if(task=="helicopter" || task=="airplane1" || task=="airplane2")
	  {
	    string input;
	    cout<<"Ready?(y/n)";
	    cin>>input;
	    correct = input[0];
	  }else{cout<<"Incorrect task entered, try again."<<endl;}
      }
  
    handSim begin_it(task);
    begin_it.pub_hands();
  
    cout<<endl<<"Do another Task?(y/n)"<<endl;
    cin>>do_another;
  
  }while(do_another!='n');
  
  return 0;
}
