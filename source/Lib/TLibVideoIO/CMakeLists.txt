
aux_source_directory(. SOURCE_FILES )

include_directories( ${PROJECT_SOURCE_DIR}/source/Lib)


set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )


add_library( TLibVideoIO_dyn SHARED ${SOURCE_FILES} )


add_library( TLibVideoIO_static STATIC ${SOURCE_FILES} )


set_target_properties( TLibVideoIO_static PROPERTIES OUTPUT_NAME "TLibVideoIO_static" )


set_target_properties( TLibVideoIO_dyn PROPERTIES CLEAN_DIRECT_OUTPUT 1 )
set_target_properties( TLibVideoIO_static PROPERTIES CLEAN_DIRECT_OUTPUT 1 )

set_target_properties( TLibVideoIO_dyn PROPERTIES VERSION 1.0 SOVERSION 1 )