set(PMDK_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../" CACHE PATH "Path to PMDK installed location.")
set(PMDK_ROOT ${PMDK_INSTALL_DIR} CACHE INTERNAL "Path to PMDK installed location.")

# (from FindOptiX.cmake)
macro(PMDK_find_api_library name)
    string(TOLOWER ${name} lowername)
  find_library(${name}_LIBRARY
    NAMES ${lowername}
    PATHS "${PMDK_INSTALL_DIR}/lib"
    NO_DEFAULT_PATH
    )
  find_library(${name}_LIBRARY
    NAMES ${lowername}
    )
  set(${name}_LIBRARY ${${name}_LIBRARY} CACHE INTERNAL "")
  if(WIN32)
    find_file(${name}_DLL
      NAMES ${lowername}.dll
      PATHS "${PMDK_INSTALL_DIR}/lib"
      NO_DEFAULT_PATH
      )
    find_file(${name}_DLL
      NAMES ${lowername}.dll
      )
      set(${name}_DLL ${${name}_DLL} CACHE INTERNAL "")
  endif()
endmacro()

PMDK_find_api_library(LIBPMEM)
PMDK_find_api_library(LIBPMEMBLK)
PMDK_find_api_library(LIBPMEMLOG)
PMDK_find_api_library(LIBPMEMOBJ)
PMDK_find_api_library(LIBPMEMPOOL)
PMDK_find_api_library(LIBVMEM)

# Include
find_path(PMDK_INCLUDE_DIR
  NAMES libpmem.h
  PATHS "${PMDK_INSTALL_DIR}/include"
  NO_DEFAULT_PATH
  )
find_path(PMDK_INCLUDE_DIR
  NAMES libpmem.h
  )

find_path(LIBPMEMOBJ_INCLUDE_DIR
  NAMES libpmemobj.h
  PATHS "${PMDK_INSTALL_DIR}/include"
  NO_DEFAULT_PATH
  )
find_path(LIBPMEMOBJ_INCLUDE_DIR
  NAMES libpmemobj.h
  )

# Macro for setting up dummy targets (from FindOptiX.cmake)
function(PMDK_add_imported_library name lib_location dll_lib dependent_libs)
  set(CMAKE_IMPORT_FILE_VERSION 1)

  # Create imported target
  add_library(${name} SHARED IMPORTED)

  if(WIN32)
    set_target_properties(${name} PROPERTIES
      IMPORTED_IMPLIB "${lib_location}"
      IMPORTED_LOCATION "${dll_lib}"
      IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
      )
  else()
    set_target_properties(${name} PROPERTIES
      IMPORTED_LOCATION "${lib_location}"
      IMPORTED_LINK_INTERFACE_LIBRARIES "${dependent_libs}"
      )
  endif()

  # Commands beyond this point should not need to know the version.
  set(CMAKE_IMPORT_FILE_VERSION)
endfunction()

set(PMDK_LIBRARIES ${LIBPMEM_LIBRARY} ${LIBPMEMBLK_LIBRARY} ${LIBPMEMLOG_LIBRARY} ${LIBPMEMOBJ_LIBRARY} ${LIBPMEMPOOL_LIBRARY} ${LIBVMEM_LIBRARY} CACHE INTERNAL "")
set(PMDK_DLLS ${LIBPMEM_DLL} ${LIBPMEMBLK_DLL} ${LIBPMEMLOG_DLL} ${LIBPMEMOBJ_DLL} ${LIBPMEMPOOL_DLL} ${LIBVMEM_DLL} CACHE INTERNAL "")

# PMDK_add_imported_library(PMDK "{PMDK_LIBRARIES}" "{PMDK_DLLS}" "")

PMDK_add_imported_library(LIBPMEM "{LIBPMEM_LIBRARY}" "{LIBPMEM_DLL}" "")
PMDK_add_imported_library(LIBPMEMBLK "{LIBPMEMBLK_LIBRARY}" "{LIBPMEMBLK_DLL}" "")
PMDK_add_imported_library(LIBPMEMLOG "{LIBPMEMLOG_LIBRARY}" "{LIBPMEMLOG_DLL}" "")
PMDK_add_imported_library(LIBPMEMOBJ "{LIBPMEMOBJ_LIBRARY}" "{LIBPMEMOBJ_DLL}" "")
PMDK_add_imported_library(LIBPMEMPOOL "{LIBPMEMPOOL_LIBRARY}" "{LIBPMEMPOOL_DLL}" "")
PMDK_add_imported_library(LIBVMEM "{LIBVMEM_LIBRARY}" "{LIBVMEM_DLL}" "")
