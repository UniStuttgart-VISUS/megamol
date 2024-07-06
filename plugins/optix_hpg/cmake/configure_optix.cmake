find_package(OptiX REQUIRED)
  
find_program(BIN2C bin2c
  DOC "Path to the cuda-sdk bin2c executable.")

# adapted from https://github.com/owl-project/owl/blob/master/owl/cmake/embed_ptx.cmake
## Copyright 2021 Jefferson Amstutz
## SPDX-License-Identifier: Apache-2.0
function(embed_ptx)
  set(oneArgs OUTPUT_TARGET)
  set(multiArgs PTX_LINK_LIBRARIES PTX_INCLUDE_DIRECTORIES SOURCES)
  cmake_parse_arguments(EMBED_PTX "" "${oneArgs}" "${multiArgs}" ${ARGN})

  if (NOT ${NUM_SOURCES} EQUAL 1)
    message(FATAL_ERROR
      "embed_ptx() can only compile and embed one file at a time.")
  endif()

  set(PTX_TARGET ${EMBED_PTX_OUTPUT_TARGET}_ptx)

  add_library(${PTX_TARGET} OBJECT)
  target_sources(${PTX_TARGET} PRIVATE ${EMBED_PTX_SOURCES})
  target_link_libraries(${PTX_TARGET} PRIVATE ${EMBED_PTX_PTX_LINK_LIBRARIES})
  target_include_directories(${PTX_TARGET} PRIVATE ${EMBED_PTX_PTX_INCLUDE_DIRECTORIES})
  set_property(TARGET ${PTX_TARGET} PROPERTY CUDA_PTX_COMPILATION ON)
  set_property(TARGET ${PTX_TARGET} PROPERTY CUDA_ARCHITECTURES OFF)
  set_property(TARGET ${PTX_TARGET} PROPERTY CXX_STANDARD 17)
  target_compile_options(${PTX_TARGET} PRIVATE $<$<CONFIG:Debug>:-lineinfo> -diag-suppress 20012) # warning suppressed due to GLM

  set(EMBED_PTX_C_FILE ${CMAKE_CURRENT_BINARY_DIR}/${EMBED_PTX_OUTPUT_TARGET}.c)
  get_filename_component(OUTPUT_FILE_NAME ${EMBED_PTX_C_FILE} NAME)
  add_custom_command(
    OUTPUT ${EMBED_PTX_C_FILE}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${EMBED_PTX_OUTPUT_TARGET} $<TARGET_OBJECTS:${PTX_TARGET}> > ${EMBED_PTX_C_FILE}
    VERBATIM
    DEPENDS $<TARGET_OBJECTS:${PTX_TARGET}> ${PTX_TARGET}
    COMMENT "Generating embedded PTX file: ${OUTPUT_FILE_NAME}"
  )

  add_library(${EMBED_PTX_OUTPUT_TARGET} OBJECT)
  target_sources(${EMBED_PTX_OUTPUT_TARGET} PRIVATE ${EMBED_PTX_C_FILE})
endfunction()
