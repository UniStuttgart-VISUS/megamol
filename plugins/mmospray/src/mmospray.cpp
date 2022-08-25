/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "OSPRayRenderer.h"

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
#include "Pkd.h"
#include "mmospray/CallOSPRayAPIObject.h"
#include "mmospray/CallOSPRayMaterial.h"
#include "mmospray/CallOSPRayStructure.h"
#include "mmospray/CallOSPRayTransformation.h"
#include "OSPRaySphericalVolume.h"

namespace megamol::ospray {
class MMOSPRayPluginInstance : public core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(MMOSPRayPluginInstance)

public:
    MMOSPRayPluginInstance(void)
            : AbstractPluginInstance("mmospray", "CPU Raytracing"){};

    ~MMOSPRayPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules

        this->module_descriptions.RegisterAutoDescription<OSPRayRenderer>();

        this->module_descriptions.RegisterAutoDescription<OSPRaySphereGeometry>();
        this->module_descriptions.RegisterAutoDescription<OSPRayMeshGeometry>();
        this->module_descriptions.RegisterAutoDescription<OSPRayStructuredVolume>();
        this->module_descriptions.RegisterAutoDescription<OSPRaySphericalVolume>();
        this->module_descriptions.RegisterAutoDescription<OSPRayAPIStructure>();
        this->module_descriptions.RegisterAutoDescription<OSPRayLineGeometry>();
        this->module_descriptions.RegisterAutoDescription<OSPRayGeometryTest>();

        this->module_descriptions.RegisterAutoDescription<OSPRayOBJMaterial>();
        this->module_descriptions.RegisterAutoDescription<OSPRayLuminousMaterial>();
        this->module_descriptions.RegisterAutoDescription<OSPRayVelvetMaterial>();
        this->module_descriptions.RegisterAutoDescription<OSPRayMatteMaterial>();
        this->module_descriptions.RegisterAutoDescription<OSPRayMetalMaterial>();
        this->module_descriptions.RegisterAutoDescription<OSPRayMetallicPaintMaterial>();
        this->module_descriptions.RegisterAutoDescription<OSPRayGlassMaterial>();
        this->module_descriptions.RegisterAutoDescription<OSPRayThinGlassMaterial>();
        this->module_descriptions.RegisterAutoDescription<OSPRayPlasticMaterial>();

        this->module_descriptions.RegisterAutoDescription<PkdBuilder>();
        this->module_descriptions.RegisterAutoDescription<OSPRayPKDGeometry>();
        this->module_descriptions.RegisterAutoDescription<OSPRayAOVSphereGeometry>();
        this->module_descriptions.RegisterAutoDescription<OSPRayTransform>();

        // register calls

        this->call_descriptions.RegisterAutoDescription<CallOSPRayStructure>();
        this->call_descriptions.RegisterAutoDescription<CallOSPRayAPIObject>();
        this->call_descriptions.RegisterAutoDescription<CallOSPRayMaterial>();
        this->call_descriptions.RegisterAutoDescription<CallOSPRayTransformation>();
    }
};
} // namespace megamol::ospray
