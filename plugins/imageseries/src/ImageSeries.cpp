/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "imageseries/AffineTransform2DCall.h"
#include "imageseries/ImageSeries2DCall.h"
#include "imageseries/graph/GraphData2DCall.h"

#include "module/ImageSeriesFlowPreprocessor.h"
#include "module/ImageSeriesGraphGenerator.h"
#include "module/ImageSeriesLabeler.h"
#include "module/ImageSeriesLoader.h"
#include "module/ImageSeriesResampler.h"
#include "module/ImageSeriesTimeDiffFilter.h"
#include "module/ImageSeriesTimestampFilter.h"

namespace megamol::ImageSeries {

class ImageSeriesPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ImageSeriesPluginInstance)
public:
    ImageSeriesPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  // machine-readable plugin assembly name
                  "imageseries",

                  // human-readable plugin description
                  "Provides modules for processing and comparing image series datasets"){};

    ~ImageSeriesPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // Register calls
        this->call_descriptions.RegisterAutoDescription<megamol::ImageSeries::ImageSeries2DCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::ImageSeries::AffineTransform2DCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::ImageSeries::GraphData2DCall>();

        // Register modules
        this->module_descriptions.RegisterAutoDescription<megamol::ImageSeries::ImageSeriesLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::ImageSeries::ImageSeriesResampler>();
        this->module_descriptions.RegisterAutoDescription<megamol::ImageSeries::ImageSeriesFlowPreprocessor>();
        this->module_descriptions.RegisterAutoDescription<megamol::ImageSeries::ImageSeriesTimeDiffFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::ImageSeries::ImageSeriesTimestampFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::ImageSeries::ImageSeriesLabeler>();
        this->module_descriptions.RegisterAutoDescription<megamol::ImageSeries::ImageSeriesGraphGenerator>();
    }
};

} // namespace megamol::ImageSeries
