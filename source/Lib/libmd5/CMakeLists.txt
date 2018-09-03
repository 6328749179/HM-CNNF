# 搜集所有在指定路径下的源文件的文件名，将输出结果列表储存在指定的变量中。
aux_source_directory(. SOURCE_FILES )

# 添加包含目录，CMake将在这些目录寻找 .cpp 文件中引用到的头文件。

# 指定编译后的库文件的输出路径。
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

# 生成目标动态库。
add_library( libmd5_dyn SHARED ${SOURCE_FILES} )

# 生成目标静态库。
add_library( libmd5_static STATIC ${SOURCE_FILES} )

# 指定静态库的输出名称。
set_target_properties( libmd5_static PROPERTIES OUTPUT_NAME "libmd5_static" )

# 允许静态库与动态库同时存在。
set_target_properties( libmd5_dyn PROPERTIES CLEAN_DIRECT_OUTPUT 1 )
set_target_properties( libmd5_static PROPERTIES CLEAN_DIRECT_OUTPUT 1 )

# 指定动态库的版本：
# VERSION 动态库版本；
# SOVERSION API的版本。
set_target_properties( libmd5_dyn PROPERTIES VERSION 1.0 SOVERSION 1 )