include(CMakeFindDependencyMacro)

find_dependency(ltla_aarand REQUIRED)

if (NOT TARGET ltla::powerit)
  get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
  get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
  if (_IMPORT_PREFIX STREQUAL "/")
    set(_IMPORT_PREFIX "")
  endif ()

  add_library(ltla::powerit INTERFACE IMPORTED)
  set_target_properties(ltla::powerit PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/ltla"
    INTERFACE_LINK_LIBRARIES "ltla::aarand")

  set(_IMPORT_PREFIX)
endif ()
