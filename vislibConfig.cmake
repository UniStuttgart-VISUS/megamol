get_filename_component(SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(ROOT_DIR "${SELF_DIR}/../../" ABSOLUTE)

include(${SELF_DIR}/vislib-target.cmake)

set(glload_LIBRARIES "${ROOT_DIR}/lib/vislib/libglload.so")
set(vislib_LIBRARIES vislib ${glload_LIBRARIES})
set(vislib_INCLUDE_DIRS "${ROOT_DIR}/include")

set(vislib_NEED_TO_COPY ${glload_LIBRARIES})

set(vislib_FOUND 1)
