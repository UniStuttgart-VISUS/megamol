find_path(BHTSNE_CPP_INCLUDE_DIR "tsne.h")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  get_filename_component(BHTSNE_LIB_DIR
    ${BHTSNE_CPP_INCLUDE_DIR}/../debug/lib/ ABSOLUTE
  )
else()
  get_filename_component(BHTSNE_LIB_DIR
    ${BHTSNE_CPP_INCLUDE_DIR}/../lib/ ABSOLUTE
  )
endif()

set(BHTSNE_LIB "${BHTSNE_LIB_DIR}/bhtsne.lib")
