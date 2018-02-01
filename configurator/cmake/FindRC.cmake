# - Find rc.exe
# Find the rc.exe
# This module defines
#  RC_EXE, where to find 'rc.exe'
#  RC_FOUND, If false, do not try to use AntTweakBar.
#

set(RC_FOUND FALSE)

set(PF86 "ProgramFiles(x86)")
set(PF "ProgramFiles")
set(SDK_DIRS "Microsoft SDKs;Windows Kits")

string(REGEX REPLACE "\\\\" "/" PF86 $ENV{${PF86}})
string(REGEX REPLACE "\\\\" "/" PF $ENV{${PF}})

set(RC_HINTS "${PF86};${PF}")
set(RC_EXE_LIST "")

foreach(hint ${RC_HINTS})
  foreach(sdk_dirs ${SDK_DIRS})
    file(GLOB_RECURSE res "${hint}/${sdk_dirs}/rc.exe")
    list(APPEND RC_EXE_LIST ${res})
   endforeach()
endforeach()

list(LENGTH RC_EXE_LIST length)
if(${length} GREATER 0)
  foreach(rc ${RC_EXE_LIST})
    #message(INFO "trying ${rc}")
    string(FIND ${rc} "x86" RC_OK86)
    string(FIND ${rc} "x64" RC_OK64)
    string(FIND ${rc} "arm64" RC_OKARM)
    if ((${RC_OK86} GREATER -1) OR (${RC_OK64} GREATER -1) AND NOT (${RC_OKARM} GREATER -1))
      set(RC_FOUND TRUE)
      #list(GET RC_EXE_LIST 0 RC_EXE)
      set(RC_EXE ${rc})
      string(REGEX REPLACE "/" "\\\\" RC_EXE ${RC_EXE})
      break()
    else()
    endif()
  endforeach()
endif()

if (${RC_FOUND})
  message(STATUS "Found resource compiler: ${RC_EXE}")
endif()
