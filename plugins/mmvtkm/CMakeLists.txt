# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmvtkm
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    mmstd
    PUBLIC mesh
    PRIVATE mmadios)

if (mmvtkm_PLUGIN_ENABLED)
  find_package(VTKm CONFIG REQUIRED)
  target_link_libraries(mmvtkm
    PRIVATE
      vtkm_cont
      vtkm_filter
      vtkm_io
      vtkm_worklet)
endif ()
