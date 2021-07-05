/*
 * OSPRay_plugin.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "OSPRayRenderer.h"
#include "OSPRayToGL.h"

#include "CallOSPRayTransformation.h"
#include "OSPRayAPIStructure.h"
#include "OSPRayLineGeometry.h"
#include "OSPRayNHSphereGeometry.h"
#include "OSPRaySphereGeometry.h"
#include "OSPRayStructuredVolume.h"
#include "OSPRayMeshGeometry.h"
#include "OSPRay_plugin/CallOSPRayAPIObject.h"
#include "OSPRay_plugin/CallOSPRayStructure.h"
#include "OSPRayAOVSphereGeometry.h"
#include "OSPRayGlassMaterial.h"
#include "OSPRayLuminousMaterial.h"
#include "OSPRayMatteMaterial.h"
#include "OSPRayMetalMaterial.h"
#include "OSPRayMetallicPaintMaterial.h"
#include "OSPRayOBJMaterial.h"
#include "OSPRayPKDGeometry.h"
#include "OSPRayPlasticMaterial.h"
#include "OSPRayThinGlassMaterial.h"
#include "OSPRayTransform.h"
#include "OSPRayVelvetMaterial.h"
#include "OSPRay_plugin/CallOSPRayMaterial.h"
#include "Pkd.h"
#include "OSPRayGeometryTest.h"

namespace megamol::ospray {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "OSPRay_plugin",

              /* human-readable plugin description */
              "CPU Raytracing"){

              // here we could perform addition initialization
          };
    /** Dtor */
    virtual ~plugin_instance(void) {
        // here we could perform addition de-initialization
    }
    /** Registers modules and calls */
    virtual void registerClasses(void) {

        // register modules here:

        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayToGL>();

        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRaySphereGeometry>();
        this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayNHSphereGeometry>();
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

        // register calls here:

        this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayStructure>();
        this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayAPIObject>();
        this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayMaterial>();
        this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayTransformation>();

    }
};
} // namespace megamol::ospray
