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
#include<std_msgs/Int8.h>

#include <math.h>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <project_simulation/AlvarMarker.h>
#include <project_simulation/AlvarMarkers.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

/*include to disp current directory
#include <stdio.h>  // defines FILENAME_MAX 
#ifdef WINDOWS
    #include <direct.h>
    #define GetCurrentDir _getcwd
#else
    #include <unistd.h>
    #define GetCurrentDir getcwd
    #endif
*/

#define EPSILON 8.8817841970012523e-016

using namespace std;

//GLOBAL VARIABLES
int PUB_RATE=30;
boost::mt19937 rng;
boost::normal_distribution<> nd(0.0, 1.0);
boost::variate_generator<boost::mt19937&, 
			 boost::normal_distribution<> > NORM_GEN(rng, nd);

int RNG_SEED = 5;//in case no seed is input as an argument
boost::mt19937 rng_rep;
boost::normal_distribution<> nd_rep(0.0, 1.0);
boost::variate_generator<boost::mt19937&, 
			 boost::normal_distribution<> > NORM_GEN_REP(rng_rep, nd_rep);


//-------------PARAMETERS
//standard deviation of the gaussian for simulated sensor noise
double PUB_NOISE_DEV = 0*0.005;
//standard deviation of the gaussian for random noise when performing 
//task with a part
double WALK_DEV = 0.1;
//parameters for gaussian representing time it takes
//for moving hand to a location or back
double MOTION_MEAN = 1.0;
double MOTION_STD = 0.01;
//Maximum velocity a hand can achieve between frames. This limits the amount of
//movement is possible for each hand between frames when human is performing a 
//task using a part from some bin
double HAND_MAX_VEL = 1.0;// m/s    
//amount of time the hand waits at a bin (simulate picking from it)
double TIME_TO_WAIT_AT_BIN = 0.3; //seconds
//time waiting before starting the first task
double WAIT_FIRST_ACTION = 6.0; //seconds
//probability of perceptual screw-up
double PROB_PERCEPT_SCREW;
//Duration of the screw-up, mean & standard-dev
double SCREW_M=3.0, SCREW_STD=1.0;
//Probability of the hand-marker jumping when a screw-up occurs
double PROB_JUMP=0.5;
//Constant-factor multiplying the std-dev of the duration
double ADD_DUR_STD;
//Constant-factor multiplying the std-dev of the hand-offset variance
double ADD_HAND_OFF_STD;
//timer-limit
size_t TIMER_LIMIT = 100;
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
    string file_path = "../src/tasks/bins/";
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
    bin_file.close();
    
  }
  
  void read_task(string file)
  {
    string file_path = "../src/tasks/";
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
    task_file.close();
  }

public:
  
  Task()
  {
    bin_list.clear();
  }

  void read_file(string file)  {read_task(file);}

  //returns false if no step left
  bool get_next_step(size_t* bin_no, double *dur_mean, double *dur_std, size_t* bin_id_to_go, string* t_nam)//, string* s_nam)
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
    *t_nam = cur_step.step_name;
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
  vector<double> lh_cheat_wait;
  vector<double> rh_cheat_wait;

  ros::NodeHandle nh;
  ros::Publisher lh_pose;
  ros::Publisher rh_pose;
  ros::Publisher viz_pub;
  ros::Publisher task_pub;
  ros::Publisher picking_bin;
  ros::Publisher human_wait_text;
  ros::Publisher lh_actual;
  ros::Publisher rh_actual;
  //Bin being picked right now, 0 if not started or ended, -1 if waiting.
  int currently_picked_bin;

  ros::Subscriber ar_poses;
  
  project_simulation::AlvarMarkers cur_markers;
  
  string frame_of_reference;
  string ar_pose_frame;

  long total_time_steps;
  long wait_time_steps;
  
  long longest_wait;
  
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
  
  //noise
  //this represents noise in the tracker
  double pub_noise_dev;
  double walk_dev;

  //are hands at rest positions
  bool is_lh_rest, is_rh_rest;

  //move hand to Position X while waiting
  bool cheat_at_waiting;
 
  //hand-offset learnt params
  double hand_t_mean[3];
  double hand_t_var[3];

  //max hand velocity
  double hand_max_vel;

  //screw-up variables
  bool cur_screw_l, cur_screw_r;
  vector<double> pos_screw_l, pos_screw_r;
  size_t screw_counter_l;
  size_t screw_counter_r;
  double duration_screw_l, duration_screw_r;

  //bin-remove mistakes
  size_t bin_rmv_mistakes;

public:
  handSim(string task_name, bool cheat);
  
  void set_arr_equal(double from[], double to[], int N);
  
  geometry_msgs::PoseStamped transform_view
  (geometry_msgs::PoseStamped to_change);

  void mat_mul(double mat_one[], size_t mat_one_size[2], double mat_two[], size_t mat_two_size[2], double mat_out[]);

  void trans_homo_vec(double homo_vec[]);
  void trans_homo_vec_hand_off(double homo_vec[], double translate[]);

  void read_ar(project_simulation::AlvarMarkers msg);
  
  void display_wait_marker();
  void delete_wait_marker();
  
  //does tasks in list
  void pub_hands();
  
  //pick out of given bin in correct time
  double perform_task(size_t cur_bin, double dur_m, double dur_s, double time_reach, bool pick_lefty, string cur_task_name, bool is_same_bin);

  
  //calculate euclidean dist between two vectors
  double calc_euclid_dist(vector<double> vec_one, vector<double> vec_two);

  //return position of queried bin
  vector<double> find_bin_pos(size_t bin_to_find);

  //wait for bin to arrive
  void wait_for_bin(size_t bin_to_chk);

  //check if bin in position
  bool bin_in_position(size_t chk_bin);
  
  //return only positive samples
  double samp_gauss_pos(double mean, double std_dev);

  //return sample
  double samp_gauss(double mean, double std_dev);

  //REPLICATABLE
  //return only positive samples
  double samp_gauss_pos_rep(double mean, double std_dev);
  //return sample
  double samp_gauss_rep(double mean, double std_dev);
  
  //publish present hand positions
  void publish_cur_hand_pos();
  
  void add_tracker_noise(geometry_msgs::PoseStamped &message);

  //publish a visualization marker at given location
  void pub_viz_marker(geometry_msgs::PoseStamped lh_pose, geometry_msgs::PoseStamped rh_pose);

  //brings hands to rest positions and then task is completed
  void pub_hands_rest();
  
  //publish hands for given time in their current location
  void wait_at_location(double wait_time);

  //interpolate to given location in specified time
  void move_to_loc(bool move_left, bool move_right, vector<double> left_pos, 
		   vector<double> right_pos, double move_time);
  
  //at bin
  double time_to_wait();
  
  void vec_to_position(geometry_msgs::Point &out_pos, vector<double> in_pos);
  
  void vec_to_orientation(geometry_msgs::Quaternion &out_orient, 
			  vector<double> in_orient);
  /*
    Quaternion to rotation matrix.
    Adapted from:
    `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_
  */
  void quaternion_matrix(
			double *quat,    /* double[4]  */
			double *matrix);  /* double[16] */
  void random_walking(double, double);

  vector<double> vec_difference(vector<double> vec_a, vector<double> vec_b);
  
  void cheat_wait(size_t bin_id);

  //change bin centroid position so hand can reach inside it
  geometry_msgs::PoseStamped transform_for_hand
  (geometry_msgs::PoseStamped bin_cur_pose);

  //choose location for perceptual screw-up
  vector<double> pick_percept_pos(bool is_left);
  
  void end_the_task();

};

//change bin centroid position so hand can reach inside it
geometry_msgs::PoseStamped handSim::transform_for_hand
(geometry_msgs::PoseStamped bin_cur_pose)
{
  double* bin_quaternion;
  double temp[] = {bin_cur_pose.pose.orientation.w,
		   bin_cur_pose.pose.orientation.x,
		   bin_cur_pose.pose.orientation.y,
		   bin_cur_pose.pose.orientation.z};
  
  bin_quaternion = temp;
  double bin_rot_mat[16];
  double* temp_rot_mat = bin_rot_mat;
  quaternion_matrix(bin_quaternion, temp_rot_mat);

  geometry_msgs::PoseStamped transformed;
  transformed = bin_cur_pose;

  //rotate
  double homo_pos_vec[4] = {samp_gauss(hand_t_mean[0], sqrt(hand_t_var[0])),
			    samp_gauss(hand_t_mean[1], sqrt(hand_t_var[1])),
			    samp_gauss(hand_t_mean[2], sqrt(hand_t_var[2])),
			    1.0};
  size_t size_mat[]={4,4}; size_t size_vec[]={4,1};
  double homo_vec_out[4];

  /*  //debug
  cout<<"time for mat "<<endl;
  for (int i=0; i<4; i++){
    for(int j=0; j<4;j++)
      {cout<<bin_rot_mat[4*i+j]<<"     ";}
      cout<<endl;}*/

  mat_mul(bin_rot_mat, size_mat, homo_pos_vec, size_vec, homo_vec_out);
    
  double translate_by[] = {bin_cur_pose.pose.position.x,
		   bin_cur_pose.pose.position.y,
		   bin_cur_pose.pose.position.z};
  double* temp_translate_by;
  temp_translate_by = translate_by;

  //convert back from homogenous and translate
  trans_homo_vec_hand_off(homo_vec_out, temp_translate_by);
    
  transformed.pose.position.x = homo_vec_out[0];
  transformed.pose.position.y = homo_vec_out[1];
  transformed.pose.position.z = homo_vec_out[2];

  //debug
  /*  cout<<"Hand positions initial:"<<bin_cur_pose<<endl;
      cout<<"Hand positions transform:"<<transformed<<endl;*/
  
  //  transformed.header.frame_id = frame_of_reference;
  
  return transformed;

}


handSim::handSim(string task_name, bool cheat)
{
  //cheat at the waiting game?
  cheat_at_waiting = cheat;

  //load tasks
  task_name += ".txt";
  to_perform.read_file(task_name);

  //set noise
  pub_noise_dev = PUB_NOISE_DEV;
  walk_dev = WALK_DEV;

  //picking which bin
  currently_picked_bin = 0;

  //initialize rest positions
  double init_pos_l[]= {-0.0813459, 0.258325,1.86598};
  double init_pos_r[]= {-0.1813459, 0.258325,1.86598};
  lh_rest.assign(&init_pos_l[0], &init_pos_l[0]+3);
  rh_rest.assign(&init_pos_r[0], &init_pos_r[0]+3);

  //if cheating- waiting position
  double cheat_l[]= {0.6813459, 0.308325,1.86598};
  lh_cheat_wait.assign(&cheat_l[0], &cheat_l[0]+3);

  double cheat_r[]= {-0.4813459, 0.308325,1.86598};
  rh_cheat_wait.assign(&cheat_r[0], &cheat_r[0]+3);
  
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
  longest_wait = 0;
  //frame
  frame_of_reference = "/kinect0_rgb_optical_frame";
  ar_pose_frame = "/lifecam1_optical_frame";
    
  //time for moving hand to location or back
  motion_mean = MOTION_MEAN;
  motion_std = MOTION_STD;
    
  //clear bin vectors
  cur_bin_locations.clear();
  temp_bin_locations.clear();
    
  //hand-offset in bin frame
  hand_t_mean[0]=0.0091831;hand_t_mean[1]=-0.13022;hand_t_mean[2]=-0.022461;
  //TODO - add the variances back in
  hand_t_var[0]=0*0.0006461+ADD_HAND_OFF_STD;hand_t_var[1]=0*0.0005190+ADD_HAND_OFF_STD;hand_t_var[2]=0*0.0001483+ADD_HAND_OFF_STD;  
  
  //percept screw-up
  cur_screw_l = false;
  cur_screw_r = false;
  
  //mistakes in bin-removal
  bin_rmv_mistakes = 0;

  //ros-stuff
  lh_pose = nh.advertise<geometry_msgs::PoseStamped>("left_hand",1);
  rh_pose = nh.advertise<geometry_msgs::PoseStamped>("right_hand",1);
  lh_actual = nh.advertise<geometry_msgs::PoseStamped>("actual_left_hand",1);
  rh_actual = nh.advertise<geometry_msgs::PoseStamped>("actual_right_hand",1);

  viz_pub = nh.advertise<visualization_msgs::MarkerArray>("hands_viz", 1);
  task_pub = nh.advertise<std_msgs::String>("action_name", 1);
  picking_bin = nh.advertise<std_msgs::Int8>("bin_being_picked", 1);
  human_wait_text = nh.advertise<visualization_msgs::Marker>("waiting_text",1);

  ar_poses = nh.subscribe("ar_pose_marker_hum", 0, &handSim::read_ar, this);  

  //publish name of the task
  std_msgs::String start_task_msg;
  start_task_msg.data= task_name;
  task_pub.publish(start_task_msg);

  //max-magnitude of hand velocity, to be used for random walk in m/s. 
  //TODO: maybe this should be checked every time the hand is published or 
  //something
  hand_max_vel = HAND_MAX_VEL;
    
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
void handSim::set_arr_equal(double from[], double to[], int N)
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
  
geometry_msgs::PoseStamped handSim::transform_view(geometry_msgs::PoseStamped to_change)
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
  
void handSim::mat_mul(double mat_one[], size_t mat_one_size[2], double mat_two[], size_t mat_two_size[2], double mat_out[])
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

  void handSim::trans_homo_vec(double homo_vec[])
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

void handSim::trans_homo_vec_hand_off(double homo_vec[], double translate[])
  {
    //convert to regular vector
    homo_vec[0] /= homo_vec[3];
    homo_vec[1] /= homo_vec[3];
    homo_vec[2] /= homo_vec[3];
    homo_vec[3] /= homo_vec[3];
    

    //translate
    homo_vec[0] += translate[0];
    homo_vec[1] += translate[1];
    homo_vec[2] += translate[2];

  }


  void handSim::read_ar(project_simulation::AlvarMarkers msg)
  {
    geometry_msgs::Point temp_pos;
    geometry_msgs::PoseStamped temp_pose;
    Bin_n_Loc temp_bin;

    temp_bin_locations.clear();

    for (size_t i=0; i<msg.markers.size(); i++)
      {
	
	//to transform each bin acc to learnt params to simulate hand reaching
	//into bin
	//debug 
	//cout<<"BIN ID::"<< msg.markers[i].id;
	temp_pose = transform_for_hand(msg.markers[i].pose);

	//to transform ar-tags frame of reference from webcam to kinect
	temp_pose = transform_view(temp_pose);

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

  //does tasks in list
  void handSim::pub_hands()
  {
    
    size_t cur_bin_id, cur_bin_no=0, prev_bin=0;
    string cur_task;
    double duration_m, duration_s;
    double time_to_next_touch = 0.0;
    bool pick_lefty = false, same_old_bin=false; //start picking from right hand

    //pop next task
    while(to_perform.get_next_step(&cur_bin_no, &duration_m, &duration_s, &cur_bin_id, &cur_task))
      {	

	same_old_bin = (cur_bin_id==prev_bin);
	prev_bin = cur_bin_id;

	//debug
	cout<< "Task - Bin-"<<cur_bin_id<<" ; mean std = "<<duration_m<<' '<<duration_s<<endl;

	//debug
	//duration_m *=3.0;
	//multiply duration std-dev with constant factor
	duration_s += ADD_DUR_STD;
	
	time_to_next_touch = perform_task(cur_bin_id, duration_m, duration_s, 
					  time_to_next_touch, pick_lefty, 
					  cur_task, same_old_bin);
	
	
	pick_lefty = !pick_lefty;
      }
    
    //after tasks complete, just publish the rest position
    currently_picked_bin =0;
    //publish tasks ended
    std_msgs::String end_task_msg;
    end_task_msg.data= "Complete";
    task_pub.publish(end_task_msg);

    pub_hands_rest();	        
  }
  
//pick out of given bin in correct time
double handSim::perform_task(size_t cur_bin, double dur_m, double dur_s, double time_reach, bool pick_lefty, string cur_task_name, bool is_same_bin)
  {
    vector<double> cur_bin_loc;
    
    //should be true only in case of picking the first bin
    if (time_reach<=0)
      {
	time_reach = samp_gauss_pos(motion_mean, motion_std);
	
	//wait at current positions- for hand after touched a bin or initially 
	//before reaching into the first bin
	double waiting_time = WAIT_FIRST_ACTION;
	wait_at_location(waiting_time);
    	
	//debug
	cout<<"First bin"<<endl;
      }
    
    //if bin to pick from unavailable
    if(!bin_in_position(cur_bin))
      {
	if (is_same_bin){++handSim::bin_rmv_mistakes;}
	//waiting-so change current picked bin
	int prev_bin = handSim::currently_picked_bin;
	handSim::currently_picked_bin = -1;

	//publish that human is currently waiting
	std_msgs::String wait_task_msg;
        string int_temp="Waiting_"; stringstream temp_int_stream;
	temp_int_stream << cur_bin;
	int_temp += temp_int_stream.str();
	wait_task_msg.data= int_temp;
	task_pub.publish(wait_task_msg);
	
	//debug
	cout<<"Waiting for bin "<<cur_bin<<endl;
	
	if(!cheat_at_waiting){wait_for_bin(cur_bin);}
	else{cheat_wait(cur_bin);}

	//always at rest after waiting
	is_lh_rest =true;
	is_rh_rest = true;
	
	//waiting over
	handSim::currently_picked_bin = prev_bin;
      }

    //after it becomes available
    cur_bin_loc = find_bin_pos(cur_bin);
    handSim::currently_picked_bin = int(cur_bin);

    //time to wait at bin
    double wait_at_bin = time_to_wait();

    //sample gaussian to determine time till next bin touch after present one
    double time_task_do = samp_gauss_pos_rep(dur_m, dur_s);

    //time for interpolating b/w rest and next bin
    double to_next_motion = samp_gauss_pos_rep(motion_mean, motion_std);
    double time_back_rest = samp_gauss_pos_rep(motion_mean, motion_std);

    //if sampled time < time for motion and waiting
    if ((wait_at_bin+to_next_motion+time_back_rest)>=time_task_do)
	  {
	    //scale both times to just fit within the actual time
	    double time_scale = time_task_do
	      /(wait_at_bin + to_next_motion+time_back_rest);
	    wait_at_bin *= time_scale;
	    to_next_motion *= time_scale;
	    time_back_rest *= time_scale;
	    time_task_do = 0;
	  }else{time_task_do -= (wait_at_bin+to_next_motion+time_back_rest);}

    
    move_to_loc(pick_lefty, !pick_lefty, cur_bin_loc, cur_bin_loc, time_reach);
    is_lh_rest = !pick_lefty;
    is_rh_rest = pick_lefty;

    //publish action as soon as bin is touched
    std_msgs::String task_msg;
    task_msg.data= cur_task_name;
    task_pub.publish(task_msg);

    //wait at the bin
    wait_at_location(wait_at_bin);
    
    //move back to rest position
    move_to_loc(!is_lh_rest, !is_rh_rest, lh_rest, rh_rest, time_back_rest);

    //use the parts retrieved
    //wait_at_location(wait_bw_bins);
    random_walking(time_task_do, walk_dev);
    
    //return the time for touching next bin
    return to_next_motion;
  }
  
  //calculate euclidean dist between two vectors
  double handSim::calc_euclid_dist(vector<double> vec_one, vector<double> vec_two)
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
  vector<double> handSim::find_bin_pos(size_t bin_to_find)
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

void handSim::end_the_task()
{
  std_msgs::String end_task_msg;
  end_task_msg.data= "Complete";
  handSim::task_pub.publish(end_task_msg);
  
  ofstream stats_file;
  stats_file.open("stats_big.txt", ios_base::app);  
  if (!stats_file.is_open()){cout<<"\nCOUDNOT WRITE STATISTICS. ABORT.\n"; exit(-1);}
  //stats_file<<"aborted"<<','<<handSim::bin_rmv_mistakes<<endl;
    
  double total_time = double(total_time_steps)/double(PUB_RATE);
  double wait_time_total = double(wait_time_steps)/double(PUB_RATE);
  double longest_wait_time = double(longest_wait)/double(PUB_RATE);

  stats_file<<total_time<<','<<wait_time_total<<','<<longest_wait_time<<','
            <<handSim::bin_rmv_mistakes<<','<<1<<endl;
  stats_file.close();
  
  exit(0);
}

  //wait for bin to arrive
  void handSim::wait_for_bin(size_t bin_to_chk)
  {
    ros::Rate loop_rate(PUB_RATE);
    
    //display viz marker saying human waits
    display_wait_marker();
    long temp_wait=0;
    bool exit_task = false;

    do
      {
	if (!(is_lh_rest && is_rh_rest))
	  {
	    double move_time = samp_gauss_pos(motion_mean, motion_std);
	    move_to_loc(!is_lh_rest, !is_rh_rest,
			lh_rest, rh_rest, move_time);
	    is_lh_rest = true; is_rh_rest=true;
	    temp_wait += move_time*PUB_RATE;
	  }
	publish_cur_hand_pos();
	ros::spinOnce();
	loop_rate.sleep();
	//increment time
	++temp_wait;
	//debug
	//cout<<"Need Bin: "<<bin_to_chk<<endl;
	if (temp_wait/PUB_RATE > TIMER_LIMIT) {
	  exit_task = true;
      break;
    }
    }while(ros::ok() && !bin_in_position(bin_to_chk));
    
    if(temp_wait>handSim::longest_wait){handSim::longest_wait=temp_wait;}
    
    wait_time_steps+= temp_wait;
    total_time_steps += temp_wait;
    //delete human waiting viz marker
    delete_wait_marker();

    if(exit_task)
      end_the_task();

  }

  //check if bin in position
  bool handSim::bin_in_position(size_t chk_bin)
  {
    ros::spinOnce();
    
    for (size_t i=0; i<cur_bin_locations.size(); i++)
      {
	if(cur_bin_locations[i].id == chk_bin)
	  {
	    return true;
	  }
      }
    return false;
  }
  
  //return only positive samples
  double handSim::samp_gauss_pos(double mean, double std_dev)
  {
    double sample = mean + std_dev*NORM_GEN();
    while(sample<=0){sample = mean + std_dev*NORM_GEN();}
    return sample;
  }

  //return sample
  double handSim::samp_gauss(double mean, double std_dev)
  {
    double sample = mean + std_dev*NORM_GEN();
    return sample;
  }

// REPLICATEABLE GAUSSIAN SAMPLING
//return only positive samples
double handSim::samp_gauss_pos_rep(double mean, double std_dev)
{
  double sample = 0;
  for(int i=0;i<50;i++) {
      if(sample <= 0)
          sample = mean + std_dev* NORM_GEN_REP();
      else
          NORM_GEN_REP();
  }
  if(sample == 0) sample = 1e-10;
  return sample;
}

  //return sample
  double handSim::samp_gauss_rep(double mean, double std_dev)
  {
    double sample = mean + std_dev* NORM_GEN_REP();
    return sample;
  }
  
//publish present hand positions
void handSim::publish_cur_hand_pos()
{
    
  geometry_msgs::PoseStamped msg_l, msg_r, msg_l_actual, msg_r_actual;
  
  msg_l.header.frame_id = frame_of_reference;
  msg_r.header.frame_id = frame_of_reference;
  
  vec_to_orientation(msg_l.pose.orientation, fix_hand_orient);
  vec_to_orientation(msg_r.pose.orientation, fix_hand_orient);
  
  
  double screw_chance;
  bool temp_screw_l=false, temp_screw_r=false;

  //check if not in screw-up mode
  if(!handSim::cur_screw_l){temp_screw_l = true;}
  if(!handSim::cur_screw_r){temp_screw_r=true;}

  //in case mode just ended
  if(handSim::cur_screw_l)
    if(handSim::screw_counter_l>=handSim::duration_screw_l)
      {
	temp_screw_l=true; handSim::cur_screw_l=false;
      }
  if(handSim::cur_screw_r)
    if(handSim::screw_counter_r>=handSim::duration_screw_r)
      {
	temp_screw_r=true; cur_screw_r=false;
      }
  if(temp_screw_l)
    {
      screw_chance=PROB_PERCEPT_SCREW;
      if((rand()/double(RAND_MAX))>(1.0-screw_chance))
	{
	  handSim::cur_screw_l = true;
	  handSim::duration_screw_l = samp_gauss_pos(SCREW_M, SCREW_STD)*PUB_RATE;//in terms of frames not seconds
	  handSim::screw_counter_l=0;
	  handSim::pos_screw_l = handSim::pick_percept_pos(true);
	}
    }

  if(temp_screw_r)
    {
      screw_chance=PROB_PERCEPT_SCREW;
      if((rand()/double(RAND_MAX))>(1.0-screw_chance))
	{
	  handSim::cur_screw_r = true;
	  handSim::duration_screw_r = samp_gauss_pos(SCREW_M, SCREW_STD)*PUB_RATE;//in terms of frames not seconds
	  handSim::screw_counter_r=0;
	  handSim::pos_screw_r = handSim::pick_percept_pos(false);
	}
    }
  
  if (!cur_screw_l){vec_to_position(msg_l.pose.position, lh_cur);}
  else{screw_counter_l++; vec_to_position(msg_l.pose.position, pos_screw_l);}
  if (!cur_screw_r)vec_to_position(msg_r.pose.position, rh_cur);
  else{screw_counter_r++; vec_to_position(msg_r.pose.position, pos_screw_r);}
  
  add_tracker_noise(msg_l);
  add_tracker_noise(msg_r);
  
  lh_pose.publish(msg_l);
  rh_pose.publish(msg_r);

  //publish actual hands
  msg_l_actual = msg_l;
  msg_r_actual = msg_r;
  vec_to_position(msg_l_actual.pose.position, lh_cur);  
  vec_to_position(msg_r_actual.pose.position, rh_cur);  
  lh_actual.publish(msg_l_actual);
  rh_actual.publish(msg_r_actual);

  
  std_msgs::Int8 bin_2_pub;
  bin_2_pub.data = currently_picked_bin;
  picking_bin.publish(bin_2_pub);
  
  //publish visual markers
  pub_viz_marker(msg_l, msg_r);
}

//randomly pick a position for the hand to appear to be in when the perceptual
//screw up occurs
vector<double> handSim::pick_percept_pos(bool is_left)
{
  //debug
  cout<<"Screw-up in lefty:"<<cur_screw_l<<"  Duration:"<<duration_screw_l<<endl;
  cout<<"Screw-up in righty:"<<cur_screw_r<<"  Duration:"<<duration_screw_r<<endl;
  //getchar(); 

  bool stay=false;

  //flip for stay at current positiion or jump
  if ((rand()/double(RAND_MAX))>PROB_JUMP){stay=true;}
  //debug
  cout<<"Stay is "<<stay<<endl;
  if(is_left)
    {
      if(stay){return lh_cur;}
      else{return lh_cheat_wait;}
    }
  else
    {
      if(stay){return rh_cur;}
      else{return rh_cheat_wait;}
    }

}
  
  void handSim::add_tracker_noise(geometry_msgs::PoseStamped &message)
  {
    message.pose.position.x += samp_gauss(0.0, pub_noise_dev);
    message.pose.position.y += samp_gauss(0.0, pub_noise_dev);
    message.pose.position.z += samp_gauss(0.0, pub_noise_dev);
  }

//publish a visualization marker to depict that human is waiting
void handSim::display_wait_marker()
{
  visualization_msgs::Marker temp_marker;

  temp_marker.header.frame_id = handSim::frame_of_reference;
  temp_marker.header.stamp = ros::Time();
  temp_marker.ns = "hands";
  temp_marker.id = 55;
  temp_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  temp_marker.action = visualization_msgs::Marker::ADD;
  temp_marker.text = "Human is waiting";
  
  vec_to_position(temp_marker.pose.position, handSim::lh_cheat_wait);
  
  temp_marker.scale.x = 0.1;
  temp_marker.scale.y =0.1;
  temp_marker.scale.z = 0.1;
  temp_marker.color.a = 0.5;
  temp_marker.color.r = 0.0;
  temp_marker.color.g = 0.8;
  temp_marker.color.b = 1.0;
  //temp_marker.lifetime = 0;
  
  human_wait_text.publish(temp_marker);

}

void handSim::delete_wait_marker()
{
  visualization_msgs::Marker temp_marker;
    
  temp_marker.header.frame_id = handSim::frame_of_reference;
  
  temp_marker.ns = "hands";
  temp_marker.id = 55;
  temp_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  temp_marker.action = visualization_msgs::Marker::DELETE;
  temp_marker.text = "Human is waiting";
  
  human_wait_text.publish(temp_marker);
}

  //publish a visualization marker at given location
  void handSim::pub_viz_marker(geometry_msgs::PoseStamped lh_pose, geometry_msgs::PoseStamped rh_pose)
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
  void handSim::pub_hands_rest()
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
    double longest_wait_time = double(longest_wait)/double(PUB_RATE);

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

    ofstream stats_file;
    stats_file.open("stats_big.txt", ios_base::app);  
    if (!stats_file.is_open()){cout<<"\nCOUDNOT WRITE STATISTICS. ABORT.\n"; exit(-1);}
    stats_file<<total_time<<','<<wait_time_total<<','<<longest_wait_time<<','
              <<handSim::bin_rmv_mistakes<<','<<0<<endl;
    stats_file.close();
  

  }
  
//publish hands for given time in their current location
void handSim::wait_at_location(double wait_time)
{
  long tot_steps = floor(wait_time*double(PUB_RATE));
  long step_no=0;
  ros::Rate loop_rate(PUB_RATE);
    
  while(step_no < tot_steps && ros::ok())
    {
      publish_cur_hand_pos();
      ++step_no;
      loop_rate.sleep();
    }


  //add to total time
  total_time_steps += tot_steps;

}


  //interpolate to given location in specified time
  void handSim::move_to_loc(bool move_left, bool move_right, vector<double> left_pos, 
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
    if(move_right){rh_cur = right_pos;}
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
  double handSim::time_to_wait()
  {
    //wait a third-a-second for the moment
    return TIME_TO_WAIT_AT_BIN;//0.3;
  }

  void handSim::vec_to_position(geometry_msgs::Point &out_pos, vector<double> in_pos)
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
  
  void handSim::vec_to_orientation(geometry_msgs::Quaternion &out_orient, 
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
  void handSim::quaternion_matrix(
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

//vec_a - vec_b
vector<double> handSim::vec_difference(vector<double> vec_a, vector<double> vec_b)
{
  if (vec_a.size()!=vec_b.size())
    {
      cout<<"Error: Vecors can't be subtracted. Sizes don't march"<<endl;
      exit(-1);
    }
  else
    {
      vector<double> difference;
      difference.clear();
      for(size_t i=0; i<vec_a.size(); ++i)
	{
	  difference.push_back(vec_a[i]-vec_b[i]);
	}
      return difference;
    }
}
//randomly walks around current hand positions
void handSim::random_walking(double walk_time, double std_dev)
{
    ros::Rate loop_rate(PUB_RATE);
    
    long time_steps = floor(walk_time*double(PUB_RATE));
    long cur_step=0;
    vector<double> lh_init = lh_cur;
    vector<double> rh_init = rh_cur;
    vector<double> diff_l;
    vector<double> diff_r;
    vector<double> temp_diff_r;
    vector<double> temp_diff_l;
    vector<double> zero_vec(3, 0.0);
    double max_step_magnitude = handSim::hand_max_vel * (1.0/double(PUB_RATE));
    double scale_l, scale_r;
    double magnitude_l, magnitude_r;

    while(cur_step<time_steps && ros::ok())
      {
	temp_diff_l.clear();
	temp_diff_r.clear();

	scale_l = 1.0;
	scale_r = 1.0;

	//differences
	diff_l = vec_difference(lh_init, lh_cur);
	diff_r = vec_difference(rh_init, rh_cur);
	
	//sample
	temp_diff_l.push_back(samp_gauss(diff_l[0], std_dev));
	temp_diff_l.push_back(samp_gauss(diff_l[1], std_dev));
	temp_diff_l.push_back(samp_gauss(diff_l[2], std_dev));

	//check magnitude
	magnitude_l = calc_euclid_dist(temp_diff_l, zero_vec);
	if (magnitude_l> max_step_magnitude)
	  {scale_l = max_step_magnitude/magnitude_l;}

	//sample
	temp_diff_r.push_back(samp_gauss(diff_r[0], std_dev));
	temp_diff_r.push_back(samp_gauss(diff_r[1], std_dev));
	temp_diff_r.push_back(samp_gauss(diff_r[2], std_dev));

	//magnitude of vector
	magnitude_r = calc_euclid_dist(temp_diff_r, zero_vec);
	if (magnitude_r> max_step_magnitude)
	  {scale_r = max_step_magnitude/magnitude_r;}


	lh_cur[0] += temp_diff_l[0]*scale_l;
	lh_cur[1] += temp_diff_l[1]*scale_l;
	lh_cur[2] += temp_diff_l[2]*scale_l;

	rh_cur[0] += temp_diff_r[0]*scale_r;
	rh_cur[1] += temp_diff_r[1]*scale_r;
	rh_cur[2] += temp_diff_r[2]*scale_r;
	
	publish_cur_hand_pos();	
	++cur_step;	
	loop_rate.sleep();
      }

    //add to time
    total_time_steps += time_steps;
    
    
}

//moves a hand to a predetermined position and waits there for bin to arrive
void handSim::cheat_wait(size_t bin_to_chk)
{
  ros::Rate loop_rate(PUB_RATE);

  if (!(is_lh_rest && is_rh_rest))
    {
      double move_time = samp_gauss_pos(motion_mean, motion_std);
      move_to_loc(!is_lh_rest, !is_rh_rest, 
		  lh_rest, rh_rest, move_time);
      is_lh_rest = true; is_rh_rest=true;
      wait_time_steps += move_time*PUB_RATE;
      total_time_steps += move_time*PUB_RATE;
    }
  
  //now, move left hand to predetermined position  
  double move_time = samp_gauss_pos(motion_mean, motion_std);
  move_to_loc(true, false, 
	      lh_cheat_wait, rh_rest, move_time);
  is_lh_rest = false; is_rh_rest=true;
  wait_time_steps += move_time*PUB_RATE;
  total_time_steps += move_time*PUB_RATE;
      
  do
    {
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


int main(int argc, char** argv)
{

  /*debug- PWD
  char cCurrentPath[FILENAME_MAX];

  if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath)))
    {
      return errno;
    }
  
  cCurrentPath[sizeof(cCurrentPath) - 1] = '\0'; // not really required 
  
  printf ("The current working directory is %s", cCurrentPath);
  debug over*/

  ros::init(argc, argv, "hand_simulator");

  ros::param::param<double>("/prob_percept_screw", PROB_PERCEPT_SCREW, 0*0.001);
  ros::param::param<double>("/add_hand_offset", ADD_HAND_OFF_STD, 0*2.0);
  ros::param::param<double>("/add_duration", ADD_DUR_STD, 0*2.0);

  bool noprompt;
  if(argc == 1) {
    noprompt = false;
  } else if(argc == 3) {
    noprompt = true;
    rng_rep.seed(time(0));
  } else if(argc == 4) {
    noprompt = true;
    rng_rep.seed(atoi(argv[3]));
  } else {
    printf("Usage: hands_sim <task> <cheat at waiting (y/n)> <duration seed>\n");
    return -1;
  }

  //seed generators
  rng.seed(time(0));
  //rng for perceptual screw-up
  srand(time(0));
  /*
  char do_another='n';
  do{
    char correct = 'n';
  */
  string task;
  
  bool cheat=false;

  /*
    while(correct != 'y')
      {
  */
  
  //char correct='n';
    
  cout<<"Which task?"<<endl;
  if(!noprompt)
    cin>> task;
  else {
    task.assign(argv[1]);
    cout << argv[1] << endl;
  }
  if(1)
    {
      string cht_inp;
      cout<<"Cheat at waiting?(y/n)"<<endl;
      if(!noprompt)
        cin>>cht_inp;
      else {
        cht_inp.assign(argv[2]);
        cout << argv[2] << endl;
      }

      if(cht_inp[0]=='y'){cheat=true;}
	    
      string input;
      cout<<"Ready?(y/n)";

      /*
      if(!noprompt) {
        cin>>input;
        correct = input[0];
      } else {
        getchar();
        cout << "y" << endl;
        correct = 'y';
      }
    }else{
    cout<<"Incorrect task entered, try again."<<endl;
    continue;
    }*/
  

      }
  
    handSim begin_it(task, cheat);
    begin_it.pub_hands();
    
  /*
    cout<<endl<<"Do another Task?(y/n)"<<endl;
    cin>>do_another;
  
  }while(do_another!='n');
  */
  
  return 0;
}
