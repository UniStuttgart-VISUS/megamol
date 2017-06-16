
INCLUDE_DIRECTORIES(
  ${PROJECT_BINARY_DIR}
  ${CMAKE_SOURCE_DIR}/vislib/include
  ${CMAKE_SOURCE_DIR}/core/include
)

FILE(GLOB plugins RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ *)
FOREACH(plugin ${plugins})
  IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${plugin}/CMakeLists.txt)
    ADD_SUBDIRECTORY(${plugin})
  ENDIF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${plugin}/CMakeLists.txt)
ENDFOREACH(plugin ${plugins})
