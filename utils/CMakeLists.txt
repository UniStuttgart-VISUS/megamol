# MegaMol
# Copyright (c) 2018, MegaMol Dev Team
# All rights reserved.
#

option(MEGAMOL_UTILS "Build MegaMol utils." ON)
set(default_enabled_utils "remoteconsole")

if (MEGAMOL_UTILS)
  file(GLOB utils RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/" *)
  foreach (util ${utils})
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${util}/CMakeLists.txt")
      string(TOUPPER "${util}" util_upper)
      if ("${util}" IN_LIST default_enabled_utils)
        option(MEGAMOL_UTIL_${util_upper} "Build util ${util}." ON)
      else ()
        option(MEGAMOL_UTIL_${util_upper} "Build util ${util}." OFF)
      endif ()
      if (MEGAMOL_UTIL_${util_upper})
        add_subdirectory(${util})
      endif ()
    endif ()
  endforeach ()
endif ()
