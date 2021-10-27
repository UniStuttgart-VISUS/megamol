/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "OSPRayRenderer.h"
#include "OSPRayToGL.h"

#include "CallOSPRayTransformation.h"
#include "OSPRayAOVSphereGeometry.h"
#include "OSPRayAPIStructure.h"
#include "OSPRayGeometryTest.h"
#include "OSPRayGlassMaterial.h"
#include "OSPRayLineGeometry.h"
#include "OSPRayLuminousMaterial.h"
#include "OSPRayMatteMaterial.h"
#include "OSPRayMeshGeometry.h"
#include "OSPRayMetalMaterial.h"
#include "OSPRayMetallicPaintMaterial.h"
#include "OSPRayOBJMaterial.h"
#include "OSPRayPKDGeometry.h"
#include "OSPRayPlasticMaterial.h"
#include "OSPRaySphereGeometry.h"
#include "OSPRayStructuredVolume.h"
#include "OSPRayThinGlassMaterial.h"
#include "OSPRayTransform.h"
#include "OSPRayVelvetMaterial.h"
#include "OSPRay_plugin/CallOSPRayAPIObject.h"
#include "OSPRay_plugin/CallOSPRayMaterial.h"
#include "OSPRay_plugin/CallOSPRayStructure.h"
#include "Pkd.h"

namespace megamol::ospray {
class OsprayPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(OsprayPluginInstance)

public:
    OsprayPluginInstance(void) : megamol::core::utility::plugins::AbstractPluginInstance("ospray", "CPU Raytracing"){};

    ~OsprayPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules

        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayToGL>();

        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRaySphereGeometry>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayMeshGeometry>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayStructuredVolume>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayAPIStructure>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayLineGeometry>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayGeometryTest>();

        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayOBJMaterial>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayLuminousMaterial>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayVelvetMaterial>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayMatteMaterial>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayMetalMaterial>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayMetallicPaintMaterial>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayGlassMaterial>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayThinGlassMaterial>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayPlasticMaterial>();

        this->module_descriptions.RegisterAutoDescription<megamol::ospray::PkdBuilder>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayPKDGeometry>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayAOVSphereGeometry>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayTransform>();

        // register calls

        this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayStructure>();
        this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayAPIObject>();
        this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayMaterial>();
        this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayTransformation>();
    }
};
} // namespace megamol::ospray
