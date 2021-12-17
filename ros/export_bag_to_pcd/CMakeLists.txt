cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
project(export_bag_to_pcd)

set(CMAKE_CXX_STANDARD 14)
set(PROJECT_VERSION 1.0.0)

find_package(catkin REQUIRED COMPONENTS
		pcl_ros
		roscpp
		sensor_msgs
		tf tf_conversions
)

find_package( Eigen3 REQUIRED)
if(EIGEN3_FOUND)
	message("EIGEN3 found")
	message ("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
	message ("EIGEN3_INCLUDE_DIRS: " ${EIGEN3_INCLUDE_DIRS})
	message ("Eigen3_DIR: " ${Eigen3_DIR})
endif()

find_package(PCL 1.5 REQUIRED)

set(Boost_USE_STATIC_LIBS        OFF)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost REQUIRED COMPONENTS system filesystem thread iostreams unit_test_framework program_options)
if(Boost_FOUND)
    message ("Boost_LIBRARIES: " ${Boost_LIBRARIES})
    message ("Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})
endif()


include_directories(
		${catkin_INCLUDE_DIRS}
)

include_directories(
	${EIGEN3_INCLUDE_DIR} 
	${EIGEN3_INCLUDE_DIRS} 
	${Eigen3_DIR}
	${Boost_INCLUDE_DIRS}
)
add_definitions(${PCL_DEFINITIONS})


add_executable(export_bag_to_pcd
		export_bag_to_pcd.cpp)

target_link_libraries(export_bag_to_pcd
		${catkin_LIBRARIES}
		)
