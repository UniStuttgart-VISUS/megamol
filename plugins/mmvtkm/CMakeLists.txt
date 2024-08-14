# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmvtkm
  BUILD_DEFAULT ON
  DEPENDS_FEATURES
    vtkm
  DEPENDS_PLUGINS
    mmstd
    PUBLIC mesh
    PRIVATE mmadios)

if (mmvtkm_PLUGIN_ENABLED)
  find_package(VTKm CONFIG REQUIRED)
  target_link_libraries(mmvtkm
    PRIVATE
      vtkm::cont
      vtkm::filter
      vtkm::io
      vtkm::worklet)
endif ()
