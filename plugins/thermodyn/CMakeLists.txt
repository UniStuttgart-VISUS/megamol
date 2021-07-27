# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(thermodyn
  BUILD_DEFAULT ON
  DEPENDS_PLUGINS
    mesh
    mmstd_datatools
  DEPENDS_EXTERNALS
    Eigen
    imgui
    implot
    nanoflann
  )

if (thermodyn_PLUGIN_ENABLED)
  find_package(CGAL REQUIRED)

  target_link_libraries(thermodyn PRIVATE CGAL::CGAL)
  if(WIN32)
    install(FILES "${CGAL_DIR}/../../bin/mpir.dll" "${CGAL_DIR}/../../bin/mpfr.dll" DESTINATION "bin")
  endif()
endif ()
