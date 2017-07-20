# Find OSPRay
#
#  OSPRAY_LIBRARIES     - The libraries needed to use OSPRay.
#  OSRTAY_INCLUDE_DIRS  - Location of the OSPRay incude dirs.

FIND_PATH(OSPRAY_INCLUDE_DIRS ospray/ospray.h)
if(NOT OSPRAY_INCLUDE_DIRS)
  message(FATAL_ERROR "failed to find /ospray/ospray.")
endif()

FIND_LIBRARY(OSPRAY_LIB "libospray")
if(NOT OSPRAY_LIB)
  message(FATAL_ERROR "failed to find libospray")
endif()
FIND_LIBRARY(OSPRAY_COMMON_LIB "libospray_common")
if(NOT OSPRAY_COMMON_LIB)
  message(FATAL_ERROR "failed to find libospray_common")
endif()
FIND_LIBRARY(TBB_LIB "libtbb")
if(NOT TBB_LIB)
  message(FATAL_ERROR "failed to find libtbb")
endif()
FIND_LIBRARY(TBB_MALLOC_LIB "libtbbmalloc")
if(NOT TBB_MALLOC_LIB)
  message(FATAL_ERROR "failed to find libtbbmalloc")
endif()
FIND_LIBRARY(EMBREE_LIB "libembree")
if(NOT EMBREE_LIB)
  message(FATAL_ERROR "failed to find libembree")
endif()

SET(OSPRAY_LIBRARIES ${OSPRAY_LIB} ${OSPRAY_COMMON_LIB} ${TBB_LIB} ${TBB_MALLOC_LIB} ${EMBREE_LIB})
