project(zlib C)

set(VERSION "1.2.11") # Just for reference.

include(CheckTypeSize)
include(CheckFunctionExists)
include(CheckIncludeFile)
include(CheckCSourceCompiles)
enable_testing()

set(ZLIB_PUBLIC_HDRS
  include/zconf.h
  include/zlib.h)
set(ZLIB_PRIVATE_HDRS
  src/crc32.h
  src/deflate.h
  src/gzguts.h
  src/inffast.h
  src/inffixed.h
  src/inflate.h
  src/inftrees.h
  src/trees.h
  src/zutil.h)
set(ZLIB_SRCS
  src/adler32.c
  src/compress.c
  src/crc32.c
  src/deflate.c
  src/gzclose.c
  src/gzlib.c
  src/gzread.c
  src/gzwrite.c
  src/inflate.c
  src/infback.c
  src/inftrees.c
  src/inffast.c
  src/trees.c
  src/uncompr.c
  src/zutil.c)

add_library(zlib STATIC ${ZLIB_SRCS} ${ZLIB_PUBLIC_HDRS} ${ZLIB_PRIVATE_HDRS})

set(ZLIB_LIBRARIES zlib CACHE INTERNAL "")

check_include_file(sys/types.h HAVE_SYS_TYPES_H)
if(HAVE_SYS_TYPES_H)
  target_compile_definitions(zlib PRIVATE HAVE_SYS_TYPES_H)
endif()

check_include_file(stdint.h HAVE_STDINT_H)
if(HAVE_STDINT_H)
  target_compile_definitions(zlib PRIVATE HAVE_STDINT_H)
endif()

check_include_file(stddef.h HAVE_STDDEF_H)
if(HAVE_STDDEF_H)
  target_compile_definitions(zlib PRIVATE HAVE_STDDEF_H)
endif()

check_include_file(unistd.h Z_HAVE_UNISTD_H)
if(Z_HAVE_UNISTD_H)
  target_compile_definitions(zlib PRIVATE Z_HAVE_UNISTD_H)
endif()

check_function_exists(fseeko HAVE_FSEEKO)
if(NOT HAVE_FSEEKO)
  target_compile_definitions(zlib PRIVATE NO_FSEEKO)
endif()

if(MSVC)
  target_compile_definitions(zlib PRIVATE 
    _CRT_SECURE_NO_DEPRECATE
	_CRT_NONSTDC_NO_DEPRECATE)
endif()

target_include_directories(zlib 
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
