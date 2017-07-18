project(GLFW)

if (NOT CMAKE_VERSION VERSION_LESS "3.0")
    # Until all major package systems have moved to CMake 3,
    # we stick with the older INSTALL_NAME_DIR mechanism
    cmake_policy(SET CMP0042 OLD)
endif()

if (NOT CMAKE_VERSION VERSION_LESS "3.1")
    cmake_policy(SET CMP0054 NEW)
endif()

set(GLFW_VERSION_MAJOR "3")
set(GLFW_VERSION_MINOR "3")
set(GLFW_VERSION_PATCH "0")
set(GLFW_VERSION_EXTRA "")
set(GLFW_VERSION "${GLFW_VERSION_MAJOR}.${GLFW_VERSION_MINOR}")
set(GLFW_VERSION_FULL "${GLFW_VERSION}.${GLFW_VERSION_PATCH}${GLFW_VERSION_EXTRA}")

if (UNIX)
    option(GLFW_USE_OSMESA "Use OSMesa for offscreen context creation" OFF)
endif()

if (MSVC)
    option(USE_MSVC_RUNTIME_LIBRARY_DLL "Use MSVC runtime library DLL" ON)
endif()

find_package(Threads REQUIRED)

#--------------------------------------------------------------------
# Set compiler specific flags
#--------------------------------------------------------------------
if (MSVC)
    if (MSVC90)
        # Workaround for VS 2008 not shipping with the DirectX 9 SDK
        include(CheckIncludeFile)
        check_include_file(dinput.h DINPUT_H_FOUND)
        if (NOT DINPUT_H_FOUND)
            message(FATAL_ERROR "DirectX 9 SDK not found")
        endif()
        # Workaround for VS 2008 not shipping with stdint.h
        list(APPEND glfw_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/deps/vs2008")
    endif()

    if (NOT USE_MSVC_RUNTIME_LIBRARY_DLL)
        foreach (flag CMAKE_C_FLAGS
                      CMAKE_C_FLAGS_DEBUG
                      CMAKE_C_FLAGS_RELEASE
                      CMAKE_C_FLAGS_MINSIZEREL
                      CMAKE_C_FLAGS_RELWITHDEBINFO)

            if (${flag} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
            endif()
            if (${flag} MATCHES "/MDd")
                string(REGEX REPLACE "/MDd" "/MTd" ${flag} "${${flag}}")
            endif()

        endforeach()
    endif()
endif()


#--------------------------------------------------------------------
# Detect and select backend APIs
#--------------------------------------------------------------------
if (GLFW_USE_OSMESA)
    set(_GLFW_OSMESA 1)
    message(STATUS "Using OSMesa for headless context creation")
elseif (WIN32)
    set(_GLFW_WIN32 1)
    message(STATUS "Using Win32 for window creation")
elseif (UNIX)
    set(_GLFW_X11 1)
    message(STATUS "Using X11 for window creation")
else()
    message(FATAL_ERROR "No supported platform was detected")
endif()

#--------------------------------------------------------------------
# Find and add Unix math and time libraries
#--------------------------------------------------------------------
if (UNIX AND NOT APPLE)
    find_library(RT_LIBRARY rt)
    mark_as_advanced(RT_LIBRARY)
    if (RT_LIBRARY)
        list(APPEND glfw_LIBRARIES "${RT_LIBRARY}")
        list(APPEND glfw_PKG_LIBS "-lrt")
    endif()

    find_library(MATH_LIBRARY m)
    mark_as_advanced(MATH_LIBRARY)
    if (MATH_LIBRARY)
        list(APPEND glfw_LIBRARIES "${MATH_LIBRARY}")
        list(APPEND glfw_PKG_LIBS "-lm")
    endif()

    if (CMAKE_DL_LIBS)
        list(APPEND glfw_LIBRARIES "${CMAKE_DL_LIBS}")
        list(APPEND glfw_PKG_LIBS "-l${CMAKE_DL_LIBS}")
    endif()
endif()

#--------------------------------------------------------------------
# Use Win32 for window creation
#--------------------------------------------------------------------
if (_GLFW_WIN32)
  list(APPEND glfw_PKG_LIBS "-lgdi32")
endif()

#--------------------------------------------------------------------
# Use X11 for window creation
#--------------------------------------------------------------------
if (_GLFW_X11)

    find_package(X11 REQUIRED)

    list(APPEND glfw_PKG_DEPS "x11")

    # Set up library and include paths
    list(APPEND glfw_INCLUDE_DIRS "${X11_X11_INCLUDE_PATH}")
    list(APPEND glfw_LIBRARIES "${X11_X11_LIB}" "${CMAKE_THREAD_LIBS_INIT}")

    # Check for XRandR (modern resolution switching and gamma control)
    if (NOT X11_Xrandr_FOUND)
        message(FATAL_ERROR "The RandR library and headers were not found")
    endif()

    list(APPEND glfw_INCLUDE_DIRS "${X11_Xrandr_INCLUDE_PATH}")
    list(APPEND glfw_LIBRARIES "${X11_Xrandr_LIB}")
    list(APPEND glfw_PKG_DEPS "xrandr")

    # Check for Xinerama (legacy multi-monitor support)
    if (NOT X11_Xinerama_FOUND)
        message(FATAL_ERROR "The Xinerama library and headers were not found")
    endif()

    list(APPEND glfw_INCLUDE_DIRS "${X11_Xinerama_INCLUDE_PATH}")
    list(APPEND glfw_LIBRARIES "${X11_Xinerama_LIB}")
    list(APPEND glfw_PKG_DEPS "xinerama")

    # Check for Xkb (X keyboard extension)
    if (NOT X11_Xkb_FOUND)
        message(FATAL_ERROR "The X keyboard extension headers were not found")
    endif()

    list(APPEND glfw_INCLUDE_DIR "${X11_Xkb_INCLUDE_PATH}")

    # Check for Xcursor
    if (NOT X11_Xcursor_FOUND)
        message(FATAL_ERROR "The Xcursor libraries and headers were not found")
    endif()

    list(APPEND glfw_INCLUDE_DIR "${X11_Xcursor_INCLUDE_PATH}")
    list(APPEND glfw_LIBRARIES "${X11_Xcursor_LIB}")
    list(APPEND glfw_PKG_DEPS "xcursor")

endif()


#--------------------------------------------------------------------
# Use OSMesa for offscreen context creation
#--------------------------------------------------------------------
if (_GLFW_OSMESA)
    find_package(OSMesa REQUIRED)
    list(APPEND glfw_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
endif()

#--------------------------------------------------------------------
# Export GLFW library dependencies
#--------------------------------------------------------------------
foreach(arg ${glfw_PKG_DEPS})
    set(GLFW_PKG_DEPS "${GLFW_PKG_DEPS} ${arg}")
endforeach()
foreach(arg ${glfw_PKG_LIBS})
    set(GLFW_PKG_LIBS "${GLFW_PKG_LIBS} ${arg}")
endforeach()

#--------------------------------------------------------------------
# Add subdirectories
#--------------------------------------------------------------------
set(common_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/internal.h"
                   "${CMAKE_CURRENT_SOURCE_DIR}/include/GLFW/glfw3.h"
                   "${CMAKE_CURRENT_SOURCE_DIR}/include/GLFW/glfw3native.h")
set(common_SOURCES
  src/context.c
  src/init.c
  src/input.c
  src/monitor.c
  src/vulkan.c
  src/window.c)


if (_GLFW_WIN32)
    set(glfw_HEADERS
      ${common_HEADERS} 
      src/win32_platform.h
      src/win32_joystick.h
      src/wgl_context.h
      src/egl_context.h
      src/osmesa_context.h)
    set(glfw_SOURCES
      ${common_SOURCES}
      src/win32_init.c
      src/win32_joystick.c
      src/win32_monitor.c
      src/win32_time.c
      src/win32_thread.c
      src/win32_window.c
      src/wgl_context.c
      src/egl_context.c
      src/osmesa_context.c)
elseif (_GLFW_X11)
    set(glfw_HEADERS
      ${common_HEADERS}
      src/x11_platform.h
      src/xkb_unicode.h
      src/posix_time.h
      src/posix_thread.h
      src/glx_context.h
      src/egl_context.h
      src/osmesa_context.h)
    set(glfw_SOURCES
      ${common_SOURCES}
      src/x11_init.c
      src/x11_monitor.c
      src/x11_window.c
      src/xkb_unicode.c
      src/posix_time.c
      src/posix_thread.c
      src/glx_context.c
      src/egl_context.c
      src/osmesa_context.c)
    if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
        set(glfw_HEADERS
          ${glfw_HEADERS}
          src/linux_joystick.h)
        set(glfw_SOURCES
          ${glfw_SOURCES}
          src/linux_joystick.c)
    else()
        set(glfw_HEADERS
          ${glfw_HEADERS}
          src/null_joystick.h)
        set(glfw_SOURCES
          ${glfw_SOURCES}
          src/null_joystick.c)
    endif()
elseif (_GLFW_OSMESA)
    set(glfw_HEADERS
      ${common_HEADERS}
      src/null_platform.h
      src/null_joystick.h
      src/posix_time.h
      src/posix_thread.h
      src/osmesa_context.h)
    set(glfw_SOURCES
      ${common_SOURCES}
      src/null_init.c
      src/null_monitor.c
      src/null_window.c
      src/null_joystick.c
      src/posix_time.c
      src/posix_thread.c
      src/osmesa_context.c)
endif()




add_library(glfw STATIC ${glfw_SOURCES} ${glfw_HEADERS})
set_target_properties(glfw PROPERTIES
                      VERSION ${GLFW_VERSION}
                      SOVERSION ${GLFW_VERSION_MAJOR})

if (_GLFW_WIN32)
  target_compile_definitions(glfw PRIVATE  _GLFW_WIN32)
elseif (_GLFW_X11)
  target_compile_definitions(glfw PRIVATE _GLFW_X11)
  target_link_libraries(glfw PRIVATE ${glfw_LIBRARIES})
elseif (_GLFW_OSMESA)
  target_compile_definitions(glfw PRIVATE _GLFW_OSMESA)
endif()

target_include_directories(glfw PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(glfw PRIVATE
                           "${CMAKE_CURRENT_SOURCE_DIR}/src"
                            ${glfw_INCLUDE_DIRS})


# Enable a reasonable set of warnings (no, -Wextra is not reasonable)
target_compile_options(glfw PRIVATE
                       "$<$<C_COMPILER_ID:Clang>:-Wall>"
                       "$<$<C_COMPILER_ID:GNU>:-Wall>")

if (MSVC)
    target_compile_definitions(glfw PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set_target_properties(glfw PROPERTIES FOLDER external)
