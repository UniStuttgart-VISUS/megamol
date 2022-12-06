find_path(CUESDK_INCLUDE_DIR "CUESDK.h")
get_filename_component(CUESDK_LIB_DIR
  ${CUESDK_INCLUDE_DIR}/../lib/ ABSOLUTE
)
get_filename_component(CUESDK_DLL_DIR
  ${CUESDK_INCLUDE_DIR}/../bin/ ABSOLUTE
)

if($ENV{VisualStudioVersion} STREQUAL "14.0")
  set(CUESDK_LIB_VERSION "2015")
elseif($ENV{VisualStudioVersion} STREQUAL "15.0")
  set(CUESDK_LIB_VERSION "2017")
elseif($ENV{VisualStudioVersion} STREQUAL "16.0")
  set(CUESDK_LIB_VERSION "2019")
elseif($ENV{VisualStudioVersion} STREQUAL "17.0")
  set(CUESDK_LIB_VERSION "2019") # no real 2022 version yet
endif()

set(CUESDK_LIB "${CUESDK_LIB_DIR}/CUESDK.x64_${CUESDK_LIB_VERSION}.lib")
set(CUESDK_DLL "${CUESDK_DLL_DIR}/CUESDK.x64_${CUESDK_DLL_VERSION}.dll")
