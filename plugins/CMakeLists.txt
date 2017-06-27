
include_directories(
  ${PROJECT_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/vislib/include
  ${CMAKE_SOURCE_DIR}/core/include
)

file(GLOB plugins RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ *)
foreach(plugin ${plugins})
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${plugin}/CMakeLists.txt)
    if(NOT ${plugin} STREQAL "template")
      add_subdirectory(${plugin})
    endif()
  endif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${plugin}/CMakeLists.txt)
endforeach(plugin ${plugins})

