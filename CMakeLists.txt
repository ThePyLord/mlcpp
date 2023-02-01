cmake_minimum_required( VERSION 3.10)
project( mlcpp )
find_package( OpenCV REQUIRED )
find_package( PythonLibs REQUIRED )
find_package( Python3 COMPONENTS Interpreter Development REQUIRED )

include_directories(${PYTHON_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp" )

add_executable(mlcpp main.cpp ${SOURCES} )
target_link_libraries( mlcpp ${OpenCV_LIBS} )
target_link_libraries(mlcpp ${PYTHON_LIBRARIES} )