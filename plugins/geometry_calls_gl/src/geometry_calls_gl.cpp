/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "geometry_calls_gl/CallTriMeshDataGL.h"


namespace megamol::geocalls {
class GeometryCallsGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(GeometryCallsGLPluginInstance)

public:
    GeometryCallsGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "geometry_calls_gl", "The geometry_calls_gl plugin."){};

    ~GeometryCallsGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::geocalls_gl::CallTriMeshDataGL>();
    }
};
} // namespace megamol::geocalls
