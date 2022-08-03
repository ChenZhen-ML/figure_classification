cmake_minimum_required(VERSION 3.21)
project(ML_detection)

set(CMAKE_CXX_STANDARD 14)

#设置可执行源文件编译成的可执行文件名
add_executable(ML_detection main.cpp digital_detection.cpp figure_detection.h)

#OpenCV_DIR为OpenCV库的配置文件所在路径
set(OpenCV_DIR D:\\OPENCV.COMPLIER)

#命令就能把一整个依赖包的头文件包含路径、库路径、库名字、版本号等情况都获取到
find_package(OpenCV REQUIRED)
#设置包含的目录
include_directories(${OpenCV_INCLUDE_DIRS})
#链接库名 aim链接OpenCV_LIBS库
target_link_libraries(ML_detection ${OpenCV_LIBS})