cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE MinSizeRel)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()

#rosbuild_add_library(blah sip/test_sip_wrapper.cpp)
#rosbuild_add_library(test_sip_wrapper sip/siptest_sip_wrapperWord.cpp sip/siptest_sip_wrappercmodule.cpp)
#target_link_libraries(test_sip_wrapper ${PYTHON_LIBRARIES} blah)
#rosbuild_add_compile_flags(test_sip_wrapper "-fPIC -Wall -W")
#set_target_properties(test_sip_wrapper PROPERTIES PREFIX "")

#rosbuild_add_library(_test_c src/C_arraytest.c)
#target_link_libraries(_test_c ${PYTHON_LIBRARIES})
#set_target_properties(_test_c PROPERTIES PREFIX "")

#rosbuild_add_library(gaussian src/gaussian.cpp)
#target_link_libraries(gaussian ${PYTHON_LIBRARIES} boost_numpy)
#set_target_properties(gaussian PROPERTIES PREFIX "")

#rosbuild_add_library(example src/wrap.cpp)
#target_link_libraries(example ${PYTHON_LIBRARIES} boost_numpy)
#set_target_properties(example PROPERTIES PREFIX "")

rosbuild_add_executable(ur_kin src/ur_kin.cpp)
rosbuild_add_library(ur_kin src/ur_kin.cpp)

# rosbuild_add_library(ur_kin_py src/ur_kin_py.cpp)
# target_link_libraries(ur_kin_py ${PYTHON_LIBRARIES} boost_numpy ur_kin)
# set_target_properties(ur_kin_py PROPERTIES PREFIX "")
