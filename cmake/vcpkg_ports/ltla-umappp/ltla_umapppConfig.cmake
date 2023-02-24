include(CMakeFindDependencyMacro)

find_dependency(ltla_aarand REQUIRED)
find_dependency(ltla_cppirlba REQUIRED)
find_dependency(ltla_knncolle REQUIRED)

if (NOT TARGET ltla::umappp)
  get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
  get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
  if (_IMPORT_PREFIX STREQUAL "/")
    set(_IMPORT_PREFIX "")
  endif ()

  add_library(ltla::umappp INTERFACE IMPORTED)
  set_target_properties(ltla::umappp PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/ltla"
    INTERFACE_LINK_LIBRARIES "ltla::aarand;ltla::cppirlba;ltla::knncolle")

  set(_IMPORT_PREFIX)
endif ()
