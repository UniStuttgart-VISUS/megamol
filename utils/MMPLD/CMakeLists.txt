if(UNIX)
  #configure_file(mmpld.sh.input mmpld.sh @ONLY)
  #execute_process(COMMAND "cp" "${CMAKE_CURRENT_BINARY_DIR}/mmpld.bat" "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE out_var RESULT_VARIABLE res_var ERROR_VARIABLE err_var)
else()
  string(REPLACE "/" "\\" THEBIN "${CMAKE_CURRENT_BINARY_DIR}")
  string(APPEND THEBIN "\\")

  string(REPLACE "/" "\\" THEINST "${CMAKE_INSTALL_PREFIX}")
  string(APPEND THEINST "\\")

  string(REPLACE "/" "\\" THESRC "${CMAKE_SOURCE_DIR}")
  string(APPEND THESRC "\\")

  string(REPLACE "/" "\\" CURSRC "${CMAKE_CURRENT_SOURCE_DIR}")
  string(APPEND CURSRC "\\")

  configure_file(mmpld.bat.input mmpld.bat @ONLY)
  execute_process(COMMAND "cmd" "/c" "copy" "${THEBIN}mmpld.bat" "${CURSRC}" OUTPUT_VARIABLE out_var RESULT_VARIABLE res_var ERROR_VARIABLE err_var)

  string(REPLACE "/" "\\\\" CURSRC "${CMAKE_CURRENT_SOURCE_DIR}")
  string(APPEND CURSRC "\\\\")

  configure_file(mmpld.reg.input mmpld.reg @ONLY)
  execute_process(COMMAND "cmd" "/c" "copy" "${THEBIN}mmpld.reg" "${CURSRC}" OUTPUT_VARIABLE out_var RESULT_VARIABLE res_var ERROR_VARIABLE err_var)
endif()
