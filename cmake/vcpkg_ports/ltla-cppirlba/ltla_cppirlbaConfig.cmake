include(CMakeFindDependencyMacro)

find_dependency(Eigen3 REQUIRED)
find_dependency(ltla_aarand REQUIRED)

if (NOT TARGET ltla::cppirlba)
  get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
  get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
  if (_IMPORT_PREFIX STREQUAL "/")
    set(_IMPORT_PREFIX "")
  endif ()

  add_library(ltla::cppirlba INTERFACE IMPORTED)
  set_target_properties(ltla::cppirlba PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/ltla"
    INTERFACE_LINK_LIBRARIES "Eigen3::Eigen;ltla::aarand")

  set(_IMPORT_PREFIX)
endif ()
