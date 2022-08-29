# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmvtkm_gl
  BUILD_DEFAULT OFF
  DEPENDS_FEATURES
    opengl
  DEPENDS_PLUGINS
    mmstd
    mmstd_gl
    mesh_gl
    mmadios
    mmvtkm)

if (mmvtkm_gl_PLUGIN_ENABLED)
  find_package(VTKm CONFIG REQUIRED)
  target_link_libraries(mmvtkm_gl
    PRIVATE
      vtkm_cont
      vtkm_filter
      vtkm_rendering
      vtkm_worklet)
endif ()
