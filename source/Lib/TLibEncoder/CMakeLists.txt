
aux_source_directory(. SOURCE_FILES )

find_package(Caffe)

include_directories( ${PROJECT_SOURCE_DIR}/source/Lib ${Caffe_INCLUDE_DIRS} )
MESSAGE(STATUS "caffe include directory2:" ${Caffe_INCLUDE_DIRS})


set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )


add_library( TLibEncoder_dyn SHARED ${SOURCE_FILES} )


add_library( TLibEncoder_static STATIC ${SOURCE_FILES} )


set_target_properties( TLibEncoder_static PROPERTIES OUTPUT_NAME "TLibEncoder_static" )


set_target_properties( TLibEncoder_dyn PROPERTIES CLEAN_DIRECT_OUTPUT 1 )
set_target_properties( TLibEncoder_static PROPERTIES CLEAN_DIRECT_OUTPUT 1 )

set_target_properties( TLibEncoder_dyn PROPERTIES VERSION 1.0 SOVERSION 1 )

