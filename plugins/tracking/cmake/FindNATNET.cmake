# Find the NatNet library
#
#  NATNET_FOUND       - True if NatNet was found.
#  NATNET_LIBRARIES   - The libraries needed to use NatNet
#  NATNET_DLL_DIR     - Location of the dll needed to use NatNet
#  NATNET_INCLUDE_DIR - Location of NatNetCAPI.h, NatNetClient.h, NatNetRepeater.h, NatNetRequests.h and NatNetTypes.h
 
FIND_PATH(NATNET_INCLUDE_DIR NatNetCAPI.h PATHS ${CMAKE_CURRENT_SOURCE_DIR}/natnet/include NO_DEFAULT_PATH)
if(NOT NATNET_INCLUDE_DIR)
  message(FATAL_ERROR "failed to find NatNetCAPI.h")
elseif(NOT EXISTS "${NATNET_INCLUDE_DIR}/NatNetClient.h")
  message(FATAL_ERROR "NatNetCAPI.h was found, but NatNetClient.h was not found in that directory.")
  SET(NATNET_INCLUDE_DIR "")
elseif(NOT EXISTS "${NATNET_INCLUDE_DIR}/NatNetRepeater.h")
  message(FATAL_ERROR "NatNetCAPI.h and NatNetClient.h were found, but NatNetRepeater.h was not found in that directory.")
  SET(NATNET_INCLUDE_DIR "")
elseif(NOT EXISTS "${NATNET_INCLUDE_DIR}/NatNetRequests.h")
  message(FATAL_ERROR "NatNetCAPI.h, NatNetClient.h and NatNetRepeater.h were found, but NatNetRequests.h was not found in that directory.")
  SET(NATNET_INCLUDE_DIR "")
elseif(NOT EXISTS "${NATNET_INCLUDE_DIR}/NatNetTypes.h")
  message(FATAL_ERROR "NatNetCAPI.h, NatNetClient.h, NatNetRepeater.h and NatNetRequests.h were found, but NatNetTypes.h was not found in that directory.")
  SET(NATNET_INCLUDE_DIR "")
endif()


if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	FIND_LIBRARY(NATNET_GENERIC_LIBRARY "NatNetLib" PATHS ${CMAKE_CURRENT_SOURCE_DIR}/natnet/lib/x64/ NO_DEFAULT_PATH)
	FIND_FILE(NATNET_DLL_DIR NatNetLib.dll PATHS ${CMAKE_CURRENT_SOURCE_DIR}/natnet/lib/x64/ NO_DEFAULT_PATH)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
	FIND_LIBRARY(NATNET_GENERIC_LIBRARY "NatNetLib" PATHS ${CMAKE_CURRENT_SOURCE_DIR}/natnet/lib/x86/ NO_DEFAULT_PATH)
	FIND_FILE(NATNET_DLL_DIR NatNetLib.dll PATHS ${CMAKE_CURRENT_SOURCE_DIR}/natnet/lib/x86/ NO_DEFAULT_PATH)
endif()
if (NOT NATNET_GENERIC_LIBRARY)
    MESSAGE(FATAL_ERROR "failed to find NatNet generic library")
elseif (NOT NATNET_DLL_DIR)
	MESSAGE(FATAL_ERROR "failed to find NatNetLib.dll")
endif ()
SET(NATNET_LIBRARIES ${NATNET_GENERIC_LIBRARY})

if(NATNET_INCLUDE_DIR AND NATNET_LIBRARIES AND NATNET_DLL_DIR)
  SET(NATNET_FOUND true)
endif()

MARK_AS_ADVANCED(NATNET_LIBRARIES NATNET_DLL_DIR NATNET_INCLUDE_DIR)
