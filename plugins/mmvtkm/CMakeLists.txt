# MegaMol
# Copyright (c) 2021, MegaMol Dev Team
# All rights reserved.
#

megamol_plugin(mmvtkm
  BUILD_DEFAULT OFF
  DEPENDS_PLUGINS
    PUBLIC mesh
    PRIVATE mmadios
  DEPENDS_EXTERNALS
    vtkm)

if (mmvtkm_PLUGIN_ENABLED)
  external_get_property(vtkm SOURCE_DIR)
  string(REPLACE "-src" "-build" BUILD_DIR "${SOURCE_DIR}")

  target_include_directories(mmvtkm PUBLIC
    "${SOURCE_DIR}"
    "${BUILD_DIR}/include"
    "${SOURCE_DIR}/vtkm/thirdparty/taotuple"
    "${SOURCE_DIR}/vtkm/thirdparty/diy"
    "${SOURCE_DIR}/vtkm/thirdparty/lcl/vtkmlcl"
    )

  target_link_libraries(mmvtkm PRIVATE vtkm_worklet)
endif ()
