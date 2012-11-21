cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

rosbuild_add_library(ur10_cm_base
    src/universal_hardware.cpp)
rosbuild_add_boost_directories()
rosbuild_add_executable(ur10_ctrl_manager src/main.cpp)
rosbuild_link_boost (ur10_ctrl_manager thread system)
target_link_libraries (ur10_ctrl_manager rt)
target_link_libraries (ur10_ctrl_manager ur10_cm_base)
