cmake_minimum_required(VERSION 3.0)

project(imguizmoquat CXX)

set(IMGUIZMOQUAT_PUBLIC_HDRS
  imGuIZMO.quat/imGuIZMOquat.h)
set(IMGUIZMOQUAT_PRIVATE_HDRS
  imGuIZMO.quat/vgConfig.h
  imGuIZMO.quat/vGizmo.h
  imGuIZMO.quat/vGizmoMath.h
  imGuIZMO.quat/vgMath.h)
set(IMGUIZMOQUAT_SRCS
  imGuIZMO.quat/imGuIZMOquat.cpp)

set_source_files_properties(${imgui_files} PROPERTIES GENERATED TRUE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(imguizmoquat STATIC ${IMGUIZMOQUAT_SRCS} ${IMGUIZMOQUAT_PUBLIC_HDRS} ${IMGUIZMOQUAT_PRIVATE_HDRS} ${imgui_files})

install(TARGETS imguizmoquat DESTINATION "lib")
install(FILES ${IMGUIZMOQUAT_PUBLIC_HDRS} ${IMGUIZMOQUAT_PRIVATE_HDRS} DESTINATION "include")

target_compile_definitions(imguizmoquat PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD IMGUIZMO_IMGUI_FOLDER= )
#target_link_libraries(imguizmoquat PUBLIC imgui)

target_include_directories(${PROJECT_NAME} PUBLIC "${SOURCE_DIR}/examples/" "${SOURCE_DIR}/misc/cpp/")