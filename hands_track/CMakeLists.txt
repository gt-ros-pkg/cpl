cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

#rosbuild_add_executable(simple_display src/simple_display.cpp)
rosbuild_add_executable(tracker src/tracker_pc.cpp)
#rosbuild_add_executable(trackNwrite src/tracker_pc_vid.cpp)
rosbuild_add_executable(track_filt src/trackFilter.cpp)
rosbuild_add_executable(record_data src/tracker_pc_store.cpp)
rosbuild_add_executable(save_points src/save_points.cpp)
#rosbuild_add_executable(tracker src/tracker.cpp)
#rosbuild_add_executable (2d_points src/tracker_w_2d.cpp)
#rosbuild_add_executable (track_fast src/tracker_faster.cpp)
#rosbuild_add_executable(old_tracker old_tracker.cpp)
#rosbuild_add_executable(writer src/tracker_debug_writer.cpp)
