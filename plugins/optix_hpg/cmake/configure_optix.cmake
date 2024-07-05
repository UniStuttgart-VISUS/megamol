# ======================================================================== #
# Copyright 2018-2020 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

#set(CMAKE_MODULE_PATH
#  "${CMAKE_CURRENT_SOURCE_DIR}/../cmake"
#  ${CMAKE_MODULE_PATH}
#  )

#include(configure_cuda)
#find_package(CUDA REQUIRED)
find_package(OptiX REQUIRED VERSION 7)

include_directories(${CUDA_TOOLKIT_INCLUDE})
include_directories(${OptiX_INCLUDE})

if (WIN32)
  add_definitions(-DNOMINMAX)
endif()

  
find_program(BIN2C bin2c
  DOC "Path to the cuda-sdk bin2c executable.")

# adapted from https://github.com/owl-project/owl/blob/master/owl/cmake/embed_ptx.cmake
## Copyright 2021 Jefferson Amstutz
## SPDX-License-Identifier: Apache-2.0
function(embed_ptx)
  set(oneArgs OUTPUT_TARGET)
  set(multiArgs PTX_LINK_LIBRARIES SOURCES)
  cmake_parse_arguments(EMBED_PTX "" "${oneArgs}" "${multiArgs}" ${ARGN})

  if (NOT ${NUM_SOURCES} EQUAL 1)
    message(FATAL_ERROR
      "embed_ptx() can only compile and embed one file at a time.")
  endif()

  set(PTX_TARGET ${EMBED_PTX_OUTPUT_TARGET}_ptx)

  add_library(${PTX_TARGET} OBJECT)
  target_sources(${PTX_TARGET} PRIVATE ${EMBED_PTX_SOURCES})
  target_link_libraries(${PTX_TARGET} PRIVATE ${EMBED_PTX_PTX_LINK_LIBRARIES})
  set_property(TARGET ${PTX_TARGET} PROPERTY CUDA_PTX_COMPILATION ON)
  set_property(TARGET ${PTX_TARGET} PROPERTY CUDA_ARCHITECTURES OFF)

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

# this macro defines cmake rules that execute the following four steps:
# 1) compile the given cuda file ${cuda_file} to an intermediary PTX file
# 2) use the 'bin2c' tool (that comes with CUDA) to
#    create a second intermediary (.c-)file which defines a const string variable
#    (named '${c_var_name}') whose (constant) value is the PTX output
#    from the previous step.
# 3) compile the given .c file to an intermediary object file (why thus has
#    that PTX string 'embedded' as a global constant.
# 4) assign the name of the intermediary .o file to the cmake variable
#    'output_var', which can then be added to cmake targets.
macro(cuda_compile_and_embed output_var cuda_file)
  set(c_var_name ${output_var})
  if(${CMAKE_BUILD_TYPE} MATCHES "Release")
    cuda_compile_ptx(ptx_files
      ${cuda_file}
      OPTIONS -O3 -DNDEBUG=1 --use_fast_math -arch=compute_${CMAKE_CUDA_ARCHITECTURES}
      )
  else()
    cuda_compile_ptx(ptx_files
      ${cuda_file}
      OPTIONS -arch=compute_${CMAKE_CUDA_ARCHITECTURES}
      )
  endif()
  list(GET ptx_files 0 ptx_file)
  set(embedded_file ${ptx_file}_embedded.c)
#  message("adding rule to compile and embed ${cuda_file} to \"const char ${var_name}[];\"")
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file}
    COMMENT "compiling (and embedding ptx from) ${cuda_file}"
    )
  set(${output_var} ${embedded_file})
endmacro()
