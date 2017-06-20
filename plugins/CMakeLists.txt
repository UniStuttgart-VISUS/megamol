
include_directories(
  ${PROJECT_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/vislib/include
  ${CMAKE_SOURCE_DIR}/core/include
)

file(GLOB plugins RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ *)
foreach(plugin ${plugins})
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${plugin}/CMakeLists.txt)
    add_subdirectory(${plugin})
  endif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${plugin}/CMakeLists.txt)
endforeach(plugin ${plugins})

