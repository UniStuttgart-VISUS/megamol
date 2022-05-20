# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(flowvis
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mesh)

if (flowvis_PLUGIN_ENABLED)
  # Use CGAL
  if (ENABLE_CGAL)
    target_link_libraries(flowvis PRIVATE CGAL::CGAL)
  endif ()
endif ()
