#
# MegaMol™ hpg Plugin
# Copyright 2020, by MegaMol Team
# Alle Rechte vorbehalten. All rights reserved.
#
string(TOUPPER hpg EXPORT_NAME)
option(BUILD_${EXPORT_NAME}_PLUGIN "Option to build hpg" ON)

if(BUILD_${EXPORT_NAME}_PLUGIN)
  if(NOT ENABLE_CUDA)
    message(FATAL_ERROR "The HPG plugin requires CUDA. Please enable it setting the option ENABLE_CUDA.")
  endif()
  project(hpg LANGUAGES CXX CUDA)

  # execute_process(COMMAND
  #     ${GIT_EXECUTABLE} clone https://github.com/owl-project/owl.git --depth 1
  #     WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  #     ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

  # set(owl_dir ${CMAKE_BINARY_DIR}/owl)
  # add_subdirectory(${owl_dir} build EXCLUDE_FROM_ALL)

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

  include(configure_optix)
  find_package(CUDAToolkit)
  
  # Check option dependencies: First list item is always the current build option,
  # afterwards list the dependencies.
  set(DEP_LIST "${DEP_LIST};BUILD_${EXPORT_NAME}_PLUGIN BUILD_CORE BUILD_MESH_PLUGIN" CACHE INTERNAL "")

  require_external(glm)
  get_target_property(GLM_INCLUDES glm INTERFACE_INCLUDE_DIRECTORIES)

  # Collect source files
  file(GLOB_RECURSE public_header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "include/*.h")
  file(GLOB_RECURSE source_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.cpp")
  file(GLOB_RECURSE header_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "src/*.h")
  #file(GLOB_RECURSE shader_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "Shaders/*")
  #file(GLOB_RECURSE cuda_resource_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "cuda_resources/*.cu")
  file(GLOB_RECURSE cuda_resource_include_files RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "cuda_resources/*.h")

  include_directories(${GLM_INCLUDES} "${CMAKE_CURRENT_SOURCE_DIR}/include")
  cuda_compile_and_embed(embedded_raygen_programs cuda_resources/raygen.cu)
  cuda_compile_and_embed(embedded_miss_programs  cuda_resources/miss.cu)
  cuda_compile_and_embed(embedded_sphere_programs cuda_resources/sphere.cu)
  cuda_compile_and_embed(embedded_mesh_programs cuda_resources/mesh.cu)

#   set(cuda_module_base_path "${CMAKE_CURRENT_BINARY_DIR}/cuda_modules")
#   set(cuda_modules )
#   foreach(obj ${cuda_resource_files})
#     get_filename_component(obj_ext ${obj} EXT)
#     get_filename_component(obj_name ${obj} NAME_WLE)
#     get_filename_component(obj_dir ${obj} DIRECTORY)

#     list(APPEND cuda_modules "${cuda_module_base_path}/${obj_name}_embedded.c")
#   endforeach()

#   add_library(${PROJECT_NAME}_PTX OBJECT ${cuda_resource_files} ${cuda_resource_include_files})
#   target_include_directories(${PROJECT_NAME}_PTX PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     "include" "src" "cuda_resources" ${OWL_INCLUDES})
#   set_target_properties(${PROJECT_NAME}_PTX PROPERTIES CUDA_PTX_COMPILATION ON)
#   set_target_properties(${PROJECT_NAME}_PTX PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#   set_target_properties(${PROJECT_NAME}_PTX PROPERTIES CUDA_USE_STATIC_CUDA_RUNTIME ON)
#   target_compile_options(${PROJECT_NAME}_PTX PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:DEBUG>:-G>$<$<CONFIG:RELEASE>:--use_fast_math>>)
  
# message(INFO " blablubb  ${OWL_CXX_FLAGS}")

#   # add_custom_command(
#   #   COMMAND ${CMAKE_COMMAND}
#   #     "-DBIN2C=${BIN2C}"
#   #     "-DOBJECTS=$<TARGET_OBJECTS:${PROJECT_NAME}_PTX>"
#   #     "-DCOPY_DIR=${cuda_module_base_path}"
#   #     -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake
#   #   VERBATIM
#   #   OUTPUT ${cuda_modules}
#   #   DEPENDS ${PROJECT_NAME}_PTX
#   #   COMMENT "Embedding ptx files"
#   # )

#   add_custom_target(${PROJECT_NAME}_PTX_Embed ALL
#     COMMAND ${CMAKE_COMMAND}
#       "-DBIN2C=${BIN2C}"
#       "-DOBJECTS=$<TARGET_OBJECTS:${PROJECT_NAME}_PTX>"
#       "-DCOPY_DIR=${cuda_module_base_path}"
#       -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake
#     VERBATIM
#     BYPRODUCTS ${cuda_modules}
#     DEPENDS ${PROJECT_NAME}_PTX
#     COMMENT "Embedding ptx files"
#   )

  # Target definition
  #add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${shader_files} ${source_files})
  #add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files} ${cuda_modules} ${cuda_resource_include_files})
  add_library(${PROJECT_NAME} SHARED ${public_header_files} ${header_files} ${source_files} ${cuda_resource_include_files} ${embedded_raygen_programs} ${embedded_miss_programs} ${embedded_sphere_programs} ${embedded_mesh_programs})
  set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".mmplg")
  target_compile_definitions(${PROJECT_NAME} PRIVATE ${EXPORT_NAME}_EXPORTS)
  target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    "include" "src" "cuda_resources" ${OWL_INCLUDES})
  target_link_libraries(${PROJECT_NAME} PRIVATE core ${OWL_LIBRARIES} glm CUDA::cuda_driver mesh)

  # Installation rules for generated files
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION "include")
  #install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Shaders/ DESTINATION "share/shaders")
  if(WIN32)
    install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin")
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION "lib")
    if(MSVC)
      install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION bin OPTIONAL)
    endif()
  else()
    install(TARGETS ${PROJECT_NAME} DESTINATION "lib" EXPORT ${PROJECT_NAME}-target)
  endif()

  # Grouping in Visual Studio
  set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER plugins)
  source_group("Public Header Files" FILES ${public_header_files})
  source_group("Header Files" FILES ${header_files})
  source_group("Source Files" FILES ${source_files})
  #source_group("Shaders" FILES ${shader_files})
  #source_group("CUDA Modules" FILES ${cuda_modules})

  #set_target_properties(${PROJECT_NAME}_PTX PROPERTIES FOLDER utility)
endif()
