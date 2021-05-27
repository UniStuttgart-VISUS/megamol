# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(probe_gl
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    probe
    mesh
  DEPENDS_EXTERNALS PUBLIC
    imgui)
