/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

// TODO: Vislib must die!!!
// Vislib includes Windows.h. This crashes when somebody else (i.e. zmq) is using Winsock2.h, but the vislib include
// is first without defining WIN32_LEAN_AND_MEAN. This define is the only thing we need from stdafx.h, include could be
// removed otherwise.
#include "stdafx.h"

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "DepthFunction.h"
#include "DiagramSeries.h"
#include "DiagramSeriesCall.h"
#include "MDSProjection.h"
#include "PCAProjection.h"
#include "TSNEProjection.h"

// Use extra block for renderer, so clang format does not change include order. Eigen3 (used in MDSProjection)
// does not compile when X11 header (used in SDFFont, which is used in renderers) is included before.
#include "HistogramRenderer2D.h"
#include "InfovisAmortizedRenderer.h"
#include "ParallelCoordinatesRenderer2D.h"
#include "ScatterplotMatrixRenderer2D.h"

namespace megamol::infovis {
class InfovisPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(InfovisPluginInstance)

public:
    InfovisPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("infovis", "Information visualization"){};

    ~InfovisPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::ParallelCoordinatesRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::ScatterplotMatrixRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::HistogramRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::InfovisAmortizedRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::PCAProjection>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::TSNEProjection>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::MDSProjection>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::DepthFunction>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::DiagramSeries>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::infovis::DiagramSeriesCall>();
    }
};
} // namespace megamol::infovis
