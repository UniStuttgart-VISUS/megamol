set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

if (PORT EQUAL "tbb")
  set(VCPKG_LIBRARY_LINKAGE dynamic)
endif ()

# Debug build is very large, i.e., runs into GitHub Actions runner limits.
if (PORT EQUAL "vtk-m")
  set(VCPKG_BUILD_TYPE release)
endif ()
