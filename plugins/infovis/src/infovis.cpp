/*
 * infovis.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

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
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(
                  /* machine-readable plugin assembly name */
                  "infovis",

                  /* human-readable plugin description */
                  "Information visualization"){

                  // here we could perform addition initialization
              };
    /** Dtor */
    virtual ~plugin_instance(void) {
        // here we could perform addition de-initialization
    }
    /** Registers modules and calls */
    virtual void registerClasses(void) {
        // register modules here:
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::ParallelCoordinatesRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::ScatterplotMatrixRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::HistogramRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::InfovisAmortizedRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::PCAProjection>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::TSNEProjection>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::MDSProjection>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::DepthFunction>();
        this->module_descriptions.RegisterAutoDescription<megamol::infovis::DiagramSeries>();

        // register calls here:
        this->call_descriptions.RegisterAutoDescription<megamol::infovis::DiagramSeriesCall>();
    }
};
} // namespace megamol::infovis
