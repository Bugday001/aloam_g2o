cmake_minimum_required(VERSION 2.8.3)
project(aloam_g2o)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/rviz_cfg)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
)

#find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
set(g2o_DIR /usr/local/lib/cmake/g2o)
find_package(g2o REQUIRED)
include_directories(
  include
	${catkin_INCLUDE_DIRS} 
	${PCL_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${g2o_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)

SET(G2O_LIBS /usr/local/lib/libg2o_cli.so /usr/local/lib/libg2o_ext_freeglut_minimal.so /usr/local/lib/libg2o_simulator.so /usr/local/lib/libg2o_solver_slam2d_linear.so /usr/local/lib/libg2o_types_icp.so /usr/local/lib/libg2o_types_slam2d.so /usr/local/lib/libg2o_core.so /usr/local/lib/libg2o_interface.so /usr/local/lib/libg2o_solver_csparse.so /usr/local/lib/libg2o_solver_structure_only.so /usr/local/lib/libg2o_types_sba.so /usr/local/lib/libg2o_types_slam3d.so /usr/local/lib/libg2o_csparse_extension.so /usr/local/lib/libg2o_opengl_helper.so /usr/local/lib/libg2o_solver_dense.so /usr/local/lib/libg2o_stuff.so /usr/local/lib/libg2o_types_sclam2d.so /usr/local/lib/libg2o_parser.so /usr/local/lib/libg2o_solver_pcg.so /usr/local/lib/libg2o_types_data.so /usr/local/lib/libg2o_types_sim3.so cxsparse )

add_executable(ascanRegistration src/scanRegistration.cpp)
target_link_libraries(ascanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(alaserOdometry src/laserOdometry.cpp)
target_link_libraries(alaserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${G2O_LIBS})

add_executable(alaserMapping src/laserMapping.cpp)
target_link_libraries(alaserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES} ${G2O_LIBS})

add_executable(kittiHelper src/kittiHelper.cpp)
target_link_libraries(kittiHelper ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})




