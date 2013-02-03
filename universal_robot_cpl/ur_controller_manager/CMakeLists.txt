cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

rosbuild_genmsg()

#set(UR10_LIBS configuration robotcomm rt kinematics collision math m)
set(UR10_LIBS robotcomm kinematics configuration dev collision m math)

rosbuild_add_executable(ur_controller src/ur_controller.cpp)
rosbuild_add_compile_flags(ur_controller -O2 -DNDEBUG -m32 -Wall)
target_link_libraries (ur_controller ${UR10_LIBS})

#rosbuild_add_library(ur10_cm_base
#    src/universal_hardware.cpp)
#rosbuild_add_boost_directories()
#rosbuild_add_executable(ur10_ctrl_manager src/main.cpp)
#rosbuild_link_boost (ur10_ctrl_manager thread system)
#target_link_libraries (ur10_ctrl_manager rt)
#target_link_libraries (ur10_ctrl_manager ur10_cm_base)
#target_link_libraries (ur10_cm_base ${UR10_LIBS})


#rosbuild_add_executable(collect_data src/collect_data.cpp)
#target_link_libraries (collect_data ur10_cm_base)
