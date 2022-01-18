/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "image/ImageRenderer.h"

namespace megamol::imageviewer2 {
class Imaggeviewer2PluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(Imaggeviewer2PluginInstance)

public:
    Imaggeviewer2PluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("imageviewer2", "The imageviewer2 plugin."){};

    ~Imaggeviewer2PluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::imageviewer2::ImageRenderer>();

        // register calls
    }
};
} // namespace megamol::imageviewer2
