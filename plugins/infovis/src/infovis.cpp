/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "DepthFunction.h"
#include "DiagramSeries.h"
#include "DiagramSeriesCall.h"
#include "MDSProjection.h"
#include "PCAProjection.h"
#include "TSNEProjection.h"

namespace megamol::infovis {
class InfovisPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(InfovisPluginInstance)

public:
    InfovisPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("infovis", "Information visualization"){};

    ~InfovisPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
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
