/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "geometry_calls/CallTriMeshData.h"
#include "geometry_calls/LinesDataCall.h"

namespace megamol::geocalls {
    class GeometryCallsPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
        REGISTERPLUGIN(GeometryCallsPluginInstance)

    public:
        GeometryCallsPluginInstance()
                : megamol::core::utility::plugins::AbstractPluginInstance(
                      "geometry_calls", "The geometry_calls plugin."){};

        ~GeometryCallsPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules

            // register calls
            this->call_descriptions.RegisterAutoDescription<megamol::geocalls::CallTriMeshData>();
            this->call_descriptions.RegisterAutoDescription<megamol::geocalls::LinesDataCall>();
        }
    };
} // namespace megamol::geocalls
