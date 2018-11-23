
dataFile = "cinematic_data.mmpld"


mmCreateView("ospray", "View3D", "::v3d")

mmCreateModule("OSPRayRenderer", "::osprayr")
mmSetParamValue("::osprayr::extraSamples",                "false") -- Disable adaptive sampling
mmSetParamValue("::osprayr::SamplesPerPixel",             "64")
mmSetParamValue("::osprayr::Type",                        "1") -- 1 = PathTracer
mmSetParamValue("::osprayr::device",                      "default") -- or "mpi_distributed" (-> EFEFCT?)

mmCreateModule("OSPRaySphereGeometry", "::OSPRaySphereGeometry1")

mmCreateModule("MMPLDDataSource", "::MMPLDDataSource1")
mmSetParamValue("::MMPLDDataSource1::filename", dataFile)

mmCreateModule("OSPRayOBJMaterial", "::OSPRayOBJMaterial1")
mmSetParamValue("::OSPRayOBJMaterial1::DiffuseColor",     "1.000000;1.000000;1.000000")
mmSetParamValue("::OSPRayOBJMaterial1::SpecularColor",    "1.000000;1.000000;1.000000")
mmSetParamValue("::OSPRayOBJMaterial1::Shininess",        "10")

mmCreateModule("OSPRayAmbientLight", "::OSPRayAmbientLight1")
mmSetParamValue("::OSPRayAmbientLight1::Intensity",       "2")
mmSetParamValue("::OSPRayAmbientLight1::Color",           "0.000000;0.000000;1.000000")

mmCreateModule("OSPRayDistantLight", "::OSPRayDistantLight1")
mmSetParamValue("::OSPRayDistantLight1::Intensity",       "2")
mmSetParamValue("::OSPRayDistantLight1::Color",           "0.500000;1.000000;1.000000")
mmSetParamValue("::OSPRayDistantLight1::Direction",       "1.000000;-1.000000;0.000000")
mmSetParamValue("::OSPRayDistantLight1::AngularDiameter", "50")

mmCreateModule("OSPRayDistantLight", "::OSPRayDistantLight2")
mmSetParamValue("::OSPRayDistantLight2::Intensity",       "2")
mmSetParamValue("::OSPRayDistantLight2::Color",           "1.000000;0.500000;1.000000")
mmSetParamValue("::OSPRayDistantLight2::Direction",       "-1.000000;1.000000;1.000000")
mmSetParamValue("::OSPRayDistantLight2::AngularDiameter", "30")

mmCreateModule("OSPRayPointLight", "::OSPRayPointLight1")
mmSetParamValue("::OSPRayPointLight1::Intensity",         "20000")
mmSetParamValue("::OSPRayPointLight1::Color",             "0.500000;1.000000;0.500000")
mmSetParamValue("::OSPRayPointLight1::Position",          "0.000000;0.000000;0.000000")
mmSetParamValue("::OSPRayPointLight1::Radius",            "0")

mmCreateCall("CallRender3D",          "::v3d::rendering",                         "::osprayr::rendering")
mmCreateCall("CallOSPRayLight",       "::OSPRayDistantLight2::getLightSlot",      "::OSPRayPointLight1::deployLightSlot")
mmCreateCall("CallOSPRayStructure",   "::osprayr::getStructure",                  "::OSPRaySphereGeometry1::deployStructureSlot")
mmCreateCall("MultiParticleDataCall", "::OSPRaySphereGeometry1::getdata",         "::MMPLDDataSource1::getdata")
mmCreateCall("CallOSPRayLight",       "::osprayr::getLight",                      "::OSPRayAmbientLight1::deployLightSlot")
mmCreateCall("CallOSPRayMaterial",    "::OSPRaySphereGeometry1::getMaterialSlot", "::OSPRayOBJMaterial1::deployMaterialSlot")
mmCreateCall("CallOSPRayLight",       "::OSPRayAmbientLight1::getLightSlot",      "::OSPRayDistantLight1::deployLightSlot")
mmCreateCall("CallOSPRayLight",       "::OSPRayDistantLight1::getLightSlot",      "::OSPRayDistantLight2::deployLightSlot")
