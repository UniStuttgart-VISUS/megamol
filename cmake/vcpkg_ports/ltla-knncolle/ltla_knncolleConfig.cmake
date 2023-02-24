include(CMakeFindDependencyMacro)

find_dependency(Annoy REQUIRED)
find_dependency(hnswlib REQUIRED)
find_dependency(ltla_cppkmeans REQUIRED)

if (NOT TARGET ltla::knncolle)
  get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
  get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
  get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
  if (_IMPORT_PREFIX STREQUAL "/")
    set(_IMPORT_PREFIX "")
  endif ()

  add_library(ltla::knncolle INTERFACE IMPORTED)
  set_target_properties(ltla::knncolle PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include/ltla"
    INTERFACE_LINK_LIBRARIES "Annoy::Annoy;hnswlib;ltla::cppkmeans")

  set(_IMPORT_PREFIX)
endif ()
