/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

// Use extra block for renderer, so clang format does not change include order. Eigen3 (used in MDSProjection)
// does not compile when X11 header (used in SDFFont, which is used in renderers) is included before.
#include "ParallelCoordinatesRenderer2D.h"
#include "ScatterplotMatrixRenderer2D.h"
#include "histo/TableHistogramRenderer2D.h"
#include "histo/TextureHistogramRenderer2D.h"

namespace megamol::infovis_gl {
class InfovisGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(InfovisGLPluginInstance)

public:
    InfovisGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("infovis_gl", "Information visualization"){};

    ~InfovisGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::infovis_gl::ParallelCoordinatesRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis_gl::ScatterplotMatrixRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis_gl::TableHistogramRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis_gl::TextureHistogramRenderer2D>();

        // register calls
    }
};
} // namespace megamol::infovis_gl
