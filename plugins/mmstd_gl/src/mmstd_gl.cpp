/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "upscaling/ImageSpaceAmortization2D.h"

namespace megamol::mmstd_gl {
class PluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(PluginInstance)
public:
    PluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("mmstd_gl", "CoreGL calls and modules."){};

    ~PluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::mmstd_gl::ImageSpaceAmortization2D>();

        // register calls
    }
};
} // namespace megamol::mmstd_gl
