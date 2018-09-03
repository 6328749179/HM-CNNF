aux_source_directory(. SOURCE_FILES )

find_package(Caffe)


include_directories( ${PROJECT_SOURCE_DIR}/source/Lib ${Caffe_INCLUDE_DIRS})


set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )


add_library( TLibCommon_dyn SHARED ${SOURCE_FILES} )

add_library( TLibCommon_static STATIC ${SOURCE_FILES} )


set_target_properties( TLibCommon_static PROPERTIES OUTPUT_NAME "TLibCommon_static" )


set_target_properties( TLibCommon_dyn PROPERTIES CLEAN_DIRECT_OUTPUT 1 )
set_target_properties( TLibCommon_static PROPERTIES CLEAN_DIRECT_OUTPUT 1 )

set_target_properties( TLibCommon_dyn PROPERTIES VERSION 1.0 SOVERSION 1 )