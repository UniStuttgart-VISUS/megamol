/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

// Use extra block for renderer, so clang format does not change include order. Eigen3 (used in MDSProjection)
// does not compile when X11 header (used in SDFFont, which is used in renderers) is included before.
#include "InfovisAmortizedRenderer.h"
#include "ParallelCoordinatesRenderer2D.h"
#include "ScatterplotMatrixRenderer2D.h"
#include "TableHistogramRenderer2D.h"
#include "TextureHistogramRenderer2D.h"

namespace megamol::infovis {
class InfovisGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(InfovisGLPluginInstance)

public:
    InfovisGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("infovis_gl", "Information visualization"){};

    ~InfovisGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::ParallelCoordinatesRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::ScatterplotMatrixRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::TableHistogramRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::TextureHistogramRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::InfovisAmortizedRenderer>();

        // register calls

    }
};
} // namespace megamol::infovis
