include(CMakeParseArguments)

function(add_clang_format TARGET)
  cmake_parse_arguments(args "" "STYLE" "FILES" ${ARGN})

  if(args_STYLE)
    get_filename_component(CFG_FILE ${args_STYLE} ABSOLUTE)
    set(STYLE "-style=file")
    set(STYLE_FILE "-assume-filename=${CFG_FILE}")
  endif()
  
  list(REMOVE_DUPLICATES args_FILES)
  list(SORT args_FILES)

  find_program(CLANG_FORMAT clang-format)

  if(CLANG_FORMAT)
    add_custom_target("${TARGET}_clangformat"
      COMMAND ${CLANG_FORMAT}
        "-i"
        ${STYLE}
        ${STYLE_FILE}
        ${args_FILES}
      DEPENDS ${args_FILES}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      VERBATIM)
    add_dependencies(${TARGET} "${TARGET}_clangformat")
  else()
    message(WARNING "clang-format was not found")
  endif()
endfunction(add_clang_format)
