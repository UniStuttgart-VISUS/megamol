/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "image_calls/Image2DCall.h"

namespace megamol::image_calls {
class ImageCallsPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ImageCallsPluginInstance)

public:
    ImageCallsPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("image_calls", "The image_calls plugin."){};

    ~ImageCallsPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::image_calls::Image2DCall>();
    }
};
} // namespace megamol::image_calls
