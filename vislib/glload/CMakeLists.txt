# Copyright (c) 2013 Sebastien Rombauts (sebastien.rombauts@gmail.com)
#
# Distributed under the MIT License (MIT) (See accompanying file LICENSE.txt
# or copy at http://opensource.org/licenses/MIT)

#cmake_minimum_required(VERSION 2.6)
project(glload)

set(GLLOAD_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "glload include directories")
 mark_as_advanced(GLLOAD_INCLUDE_DIRS)
# Define useful variables to handle OS/Compiler differences
if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} /DMEMORY_DEBUGGING")
    add_definitions (/D_CRT_SECURE_NO_WARNINGS /D_SCL_SECURE_NO_WARNINGS)
    # List Windows only sources/headers files
    file(GLOB os_specific_files
     "${PROJECT_SOURCE_DIR}/include/glload/wgl_*.h"
     "${PROJECT_SOURCE_DIR}/source/wgl_*"
    )
else()
    set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -DMEMORY_DEBUGGING")
    add_definitions (-DLOAD_X11)
    # List Linux only sources/headers files
    file(GLOB os_specific_files
     "${PROJECT_SOURCE_DIR}/include/glload/glx_*.h"
     "${PROJECT_SOURCE_DIR}/source/glx_*"
    )
endif()

# List common sources/headers files
file(GLOB common_files
 "${PROJECT_SOURCE_DIR}/include/glload/gl_*.h"
 "${PROJECT_SOURCE_DIR}/include/glload/gl_*.hpp"
 "${PROJECT_SOURCE_DIR}/include/glload/gll.h"
 "${PROJECT_SOURCE_DIR}/include/glload/gll.hpp"
 "${PROJECT_SOURCE_DIR}/source/gl_*"
 "${PROJECT_SOURCE_DIR}/source/visgl_load.cpp"
)

# All includes are relative to the "include" directory
include_directories ("${PROJECT_SOURCE_DIR}/include")

# add sources of the "glload" library
add_library (glload SHARED 
 ${common_files}
 ${os_specific_files}
 )
# Add OpenGL library
target_link_libraries(glload PRIVATE ${OPENGL_LIBRARIES})

if(WIN32)
  target_compile_definitions(glload PRIVATE GL_LOAD_DLL_EXPORT)
endif()

install(TARGETS glload RUNTIME DESTINATION "bin")
install(TARGETS glload ARCHIVE DESTINATION "lib")
