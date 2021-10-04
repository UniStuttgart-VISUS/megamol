# save CMake variables for later
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
file(WRITE ${CMAKE_BINARY_DIR}/config/mmconfig.h "#pragma once\ninline constexpr char CMake_Cache[] =\n")
foreach (_variableName ${_variableNames})
  file(APPEND ${CMAKE_BINARY_DIR}/config/mmconfig.h "    R\"(${_variableName}=${${_variableName}})\" \"\\n\"\n")
endforeach()
file(APPEND ${CMAKE_BINARY_DIR}/config/mmconfig.h ";\n")