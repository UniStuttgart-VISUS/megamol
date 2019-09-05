include(External)

#
# Centralized function to require externals to add them once by invoking
# require_external(<EXTERNAL_TARGET>).
# The target of externals added this way, can be accessed by the (alias) name
# <EXTERNAL_TARGET>_ext.
#
# Think of this function as a big switch, testing for the name and presence
# of the external target to guard against duplicated targets.
#
function(require_external NAME)
  set(FETCHCONTENT_QUIET ON CACHE BOOL "")
  set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "")

  ### Header-only libraries #############################################################

  # Delaunator
  if(NAME STREQUAL "delaunator")
    fetch_external_headeronly(${NAME}
      GIT_REPOSITORY https://github.com/delfrrr/delaunator-cpp.git
      GIT_TAG "v0.4.0")

  # Eigen
  elseif(NAME STREQUAL "eigen")
    fetch_external_headeronly(${NAME}
      GIT_REPOSITORY https://github.com/eigenteam/eigen-git-mirror.git
      GIT_TAG "3.3.4")

  # glm
  elseif(NAME STREQUAL "glm")
    fetch_external_headeronly(${NAME}
      GIT_REPOSITORY https://github.com/g-truc/glm.git
      GIT_TAG "0.9.8")

  # glowl
  elseif(NAME STREQUAL "glowl")
    fetch_external_headeronly(${NAME}
      GIT_REPOSITORY https://github.com/invor/glowl.git
      GIT_TAG "v0.1")

  # json
  elseif(NAME STREQUAL "json")
    fetch_external_headeronly(${NAME}
      GIT_REPOSITORY https://github.com/azadkuh/nlohmann_json_release.git
      GIT_TAG "v3.5.0")

  # libcppzmq
  elseif(NAME STREQUAL "libcppzmq")
    require_external(libzmq)

    fetch_external_headeronly(${NAME}
      GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
      GIT_TAG "v4.4.1")

    add_dependencies(${NAME}_ext libzmq_ext)

  # libcxxopts
  elseif(NAME STREQUAL "libcxxopts")
    fetch_external_headeronly(${NAME}
      GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
      GIT_TAG "v2.1.1")

  # mmpld_io
  elseif(NAME STREQUAL "mmpld_io")
    fetch_external_headeronly(${NAME}
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/mmpld_io.git)

  # nanoflann
  elseif(NAME STREQUAL "nanoflann")
    fetch_external_headeronly(${NAME}
      GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
      GIT_TAG "v1.3.0")

  # sim_sort
  elseif(NAME STREQUAL "sim_sort")
    fetch_external_headeronly(${NAME}
      GIT_REPOSITORY https://github.com/alexstraub1990/simultaneous-sort.git)

  ### Built libraries ###################################################################

  # bhtsne
  elseif(NAME STREQUAL "bhtsne")
    fetch_external(${NAME} bhtsne
      ""
      ""
      ""
      GIT_REPOSITORY  https://github.com/lvdmaaten/bhtsne.git
      GIT_TAG "36b169c88250d0afe51828448dfdeeaa508f13bc"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/cmake/bhtsne/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

  # glfw
  elseif(NAME STREQUAL "glfw")
    fetch_external(${NAME} glfw
      ""
      "BUILD_SHARED_LIBS;GLFW_BUILD_EXAMPLES;GLFW_BUILD_TESTS;GLFW_BUILD_DOCS;GLFW_INSTALL"
      "GLFW_DOCUMENT_INTERNALS;GLFW_USE_HYBRID_HPG;GLFW_VULKAN_STATIC;LIB_SUFFIX;USE_MSVC_RUNTIME_LIBRARY_DLL"
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG "3.2.1")

  # imgui
  elseif(NAME STREQUAL "imgui")
    set(FETCHCONTENT_ADDITIONAL_INCLUDES "misc/cpp" "examples")

    fetch_external(${NAME} imgui
      ""
      ""
      ""
      GIT_REPOSITORY https://github.com/ocornut/imgui.git
      GIT_TAG "v1.70"
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/cmake/ImGui/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

    unset(FETCHCONTENT_ADDITIONAL_INCLUDES)

  # libpng
  elseif(NAME STREQUAL "libpng")
    require_external(zlib)

    get_target_property(ZLIB_INCLUDE_DIR zlib_ext INCLUDE_DIRECTORIES)
    set(ZLIB_LIBRARY zlib_ext)

    foreach(DIR IN LISTS ZLIB_INCLUDE_DIR)
      if(EXISTS "${DIR}/zlib.h")
        file(STRINGS "${DIR}/zlib.h" ZLIB_VERNUM REGEX "^#define ZLIB_VERNUM 0x[a-f0-9]+$")
        string(REGEX MATCH "0x[a-f0-9]+" ZLIB_VERNUM ${ZLIB_VERNUM})
      endif()
    endforeach()

    fetch_external(${NAME} png_static
      "CMAKE_POSITION_INDEPENDENT_CODE;PNG_BUILD_ZLIB;PNG_STATIC;SKIP_INSTALL_ALL"
      "PNG_SHARED;PNG_TESTS"
      "AWK;DFA_XTRA;PNGARG;PNG_DEBUG;PNG_FRAMEWORK;PNG_HARDWARE_OPTIMIZATIONS;PNG_PREFIX;PNG_SHARED;PNG_TESTS;ld-version-script"
      GIT_REPOSITORY https://github.com/UniStuttgart-VISUS/libpng.git
      GIT_TAG "v1.6.34")

    add_dependencies(${NAME}_ext_genfiles zlib_ext)
    add_dependencies(png_static ${NAME}_ext_genfiles zlib_ext)

  # libzmq
  elseif(NAME STREQUAL "libzmq")
    fetch_external(${NAME} libzmq-static
      "BUILD_STATIC"
      "BUILD_SHARED;BUILD_TESTS;ENABLE_PRECOMPILED;WITH_PERF_TOOL;ZMQ_BUILD_TESTS"
      "API_POLLER;ENABLE_ANALYSIS;ENABLE_ASAN;ENABLE_CPACK;ENABLE_CURVE;ENABLE_DRAFTS;ENABLE_EVENTFD;ENABLE_INTRINSICS;ENABLE_RADIX_TREE;LIBZMQ_PEDANTIC;LIBZMQ_WERROR;POLLER;RT_LIBRARY;WITH_DOCS;WITH_LIBSODIUM;WITH_MILITANT;WITH_OPENPGM;WITH_VMCI;ZEROMQ_CMAKECONFIG_INSTALL_DIR;ZEROMQ_LIBRARY;ZMQ_CV_IMPL;ZMQ_WIN32_WINNT"
      GIT_REPOSITORY https://github.com/zeromq/libzmq.git
      GIT_TAG 3413e05bd062cc36188f5970b55f657688cddc72)

  # quickhull
  elseif(NAME STREQUAL "quickhull")
    fetch_external(${NAME} quickhull
      ""
      ""
      ""
      GIT_REPOSITORY https://github.com/akuukka/quickhull.git
      PATCH_COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_SOURCE_DIR}/cmake/quickhull/CMakeLists.txt"
        "<SOURCE_DIR>/CMakeLists.txt")

  # tinyply
  elseif(NAME STREQUAL "tinyply")
    set(FETCHCONTENT_ADDITIONAL_INCLUDES "source")

    fetch_external(${NAME} tinyply
      ""
      "SHARED_LIB"
      ""
      GIT_REPOSITORY https://github.com/ddiakopoulos/tinyply.git
      GIT_TAG "2.1")

    unset(FETCHCONTENT_ADDITIONAL_INCLUDES)

  # zfp
  elseif(NAME STREQUAL "zfp")
    fetch_external(${NAME} zfp
      "ZFP_WITH_ALIGNED_ALLOC;ZFP_WITH_CACHE_FAST_HASH"
      "BUILD_EXAMPLES;BUILD_SHARED_LIBS;BUILD_UTILITIES;BUILD_TESTING"
      "ZFP_BIT_STREAM_WORD_SIZE;ZFP_ENABLE_PIC;ZFP_WITH_BIT_STREAM_STRIDED;ZFP_WITH_CACHE_PROFILE;ZFP_WITH_CACHE_TWOWAY"
      GIT_REPOSITORY https://github.com/LLNL/zfp.git
      GIT_TAG "0.5.2")

  # zlib
  elseif(NAME STREQUAL "zlib")
    fetch_external(${NAME} zlibstatic
      "CMAKE_POSITION_INDEPENDENT_CODE;SKIP_INSTALL_ALL"
      ""
      "AMD64;ASM686;CMAKE_BACKWARDS_COMPATIBILITY;EXECUTABLE_OUTPUT_PATH;INSTALL_BIN_DIR;INSTALL_INC_DIR;INSTALL_LIB_DIR;INSTALL_MAN_DIR;INSTALL_PKGCONFIG_DIR;LIBRARY_OUTPUT_PATH"
      GIT_REPOSITORY https://github.com/madler/zlib.git
      GIT_TAG "v1.2.11")

  else()
    message(FATAL_ERROR "Unknown external required \"${NAME}\"")
  endif()

  # Hide fetch-content variables
  mark_as_advanced(FORCE FETCHCONTENT_BASE_DIR)
  mark_as_advanced(FORCE FETCHCONTENT_FULLY_DISCONNECTED)
  mark_as_advanced(FORCE FETCHCONTENT_QUIET)
  mark_as_advanced(FORCE FETCHCONTENT_UPDATES_DISCONNECTED)
endfunction(require_external)
