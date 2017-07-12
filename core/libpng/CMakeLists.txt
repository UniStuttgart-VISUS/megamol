project(libpng C)

set(PNG_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE INTERNAL "")

set(PNGLIB_MAJOR 1)
set(PNGLIB_MINOR 6)
set(PNGLIB_RELEASE 30)
set(PNGLIB_NAME libpng${PNGLIB_MAJOR}${PNGLIB_MINOR})
set(PNGLIB_VERSION ${PNGLIB_MAJOR}.${PNGLIB_MINOR}.${PNGLIB_RELEASE})


if(NOT WIN32)
  find_library(M_LIBRARY
    NAMES m
    PATHS /usr/lib /usr/local/lib
  )
  if(NOT M_LIBRARY)
    message(STATUS "math lib 'libm' not found; floating point support disabled")
  endif()
else()
  # not needed on windows
  set(M_LIBRARY "")
endif()


if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif(MSVC)

include_directories(${PNG_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIRS})

set(libpng_public_hdrs
  include/png.h
  include/pngconf.h
  include/pnglibconf.h
)
set(libpng_private_hdrs
  src/pngpriv.h
  src/pngdebug.h
  src/pnginfo.h
  src/pngstruct.h
)

set(libpng_sources
  ${libpng_public_hdrs}
  ${libpng_private_hdrs}
  src/png.c
  src/pngerror.c
  src/pngget.c
  src/pngmem.c
  src/pngpread.c
  src/pngread.c
  src/pngrio.c
  src/pngrtran.c
  src/pngrutil.c
  src/pngset.c
  src/pngtrans.c
  src/pngwio.c
  src/pngwrite.c
  src/pngwtran.c
  src/pngwutil.c
)

add_library(libpng STATIC ${libpng_sources})
target_link_libraries(libpng ${ZLIB_LIBRARIES} ${M_LIBRARY})

set_target_properties(libpng PROPERTIES FOLDER external)

set(PNG_LIBRARIES libpng CACHE INTERNAL "")