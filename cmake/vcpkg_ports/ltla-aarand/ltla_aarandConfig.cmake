if (NOT TARGET ltla::aarand)
  get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
  get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
  if (_IMPORT_PREFIX STREQUAL "/")
    set(_IMPORT_PREFIX "")
  endif ()

  add_library(ltla::aarand INTERFACE IMPORTED)
  set_target_properties(ltla::aarand PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/ltla")

  set(_IMPORT_PREFIX)
endif ()
