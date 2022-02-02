/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "module/ImageSeriesRenderer.h"

namespace megamol::ImageSeries::GL {

class ImageSeriesGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ImageSeriesGLPluginInstance)
public:
    ImageSeriesGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  // machine-readable plugin assembly name
                  "imageseries_gl",

                  // human-readable plugin description
                  "Provides modules for processing and comparing image series datasets (OpenGL modules)"){};

    ~ImageSeriesGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {
        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::ImageSeries::GL::ImageSeriesRenderer>();
    }
};

} // namespace megamol::ImageSeries::GL
