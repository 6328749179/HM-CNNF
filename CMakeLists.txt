cmake_minimum_required(VERSION 2.8.7)

##You must set your own caffe path
set(Caffe_DIR ../caffe-master)
add_definitions(-DCaffe_DIR=${Caffe_DIR})

PROJECT(TAppEncoder)
PROJECT(TAppDecoder)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

ADD_SUBDIRECTORY( ./source/Lib/TAppCommon )
ADD_SUBDIRECTORY( ./source/Lib/TLibCommon )
ADD_SUBDIRECTORY( ./source/Lib/TLibVideoIO)
ADD_SUBDIRECTORY( ./source/Lib/TLibEncoder)
ADD_SUBDIRECTORY( ./source/App/TAppEncoder)
ADD_SUBDIRECTORY( ./source/Lib/libmd5)


