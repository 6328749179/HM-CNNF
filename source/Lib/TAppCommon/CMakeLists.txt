
aux_source_directory(. SOURCE_FILES )


set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )


add_library( TAppCommon_dyn SHARED ${SOURCE_FILES} )


add_library( TAppCommon_static STATIC ${SOURCE_FILES} )


set_target_properties( TAppCommon_static PROPERTIES OUTPUT_NAME "TAppCommon_static" )


set_target_properties( TAppCommon_dyn PROPERTIES CLEAN_DIRECT_OUTPUT 1 )
set_target_properties( TAppCommon_static PROPERTIES CLEAN_DIRECT_OUTPUT 1 )


set_target_properties( TAppCommon_dyn PROPERTIES VERSION 1.0 SOVERSION 1 )