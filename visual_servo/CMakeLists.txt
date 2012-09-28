cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)
# set(ROS_BUILD_TYPE Debug)

# Build actions
rosbuild_find_ros_package(actionlib_msgs)
include(${actionlib_msgs_PACKAGE_PATH}/cmake/actionbuild.cmake)
genaction()

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

rosbuild_genmsg()
rosbuild_gensrv()

find_package(OpenCV REQUIRED)
rosbuild_add_boost_directories()
# rosbuild_add_executable(visual_servo_node src/visual_servo.cpp)
#rosbuild_add_executable(visual_servo_node src/vs_line.cpp)
#target_link_libraries(visual_servo_node tabletop_pushing)

find_library(LOG4CXX_LIB log4cxx REQUIRED)
rosbuild_add_executable(tester src/vs_pose.cpp)
rosbuild_link_boost(tester signals)
target_link_libraries(tester ${LOG4CXX_LIB})

rosbuild_add_executable(vsaction src/vs_action.cpp)
rosbuild_link_boost(vsaction signals)
target_link_libraries(vsaction ${LOG4CXX_LIB})

#rosbuild_add_executable(visual_servo_node src/vs_grasp.cpp)
#rosbuild_link_boost(visual_servo_node signals)
#target_link_libraries(visual_servo_node ${LOG4CXX_LIB})
