cmake_minimum_required(VERSION 2.8)
project(rubik)



find_package(OpenCV)


include_directories(
include ${OpenCV_INCLUDE_DIRS}
)

add_definitions(-std=c++11 -fPIC -pedantic -g -O0 -Wno-missing-field-initializers)

add_executable(rubik src/main.cpp)
target_link_libraries(rubik -lrealsense ${OpenCV_LIBRARIES} )


