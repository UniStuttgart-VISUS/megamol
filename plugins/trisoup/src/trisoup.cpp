/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "OSCBFix.h"
#include "WavefrontObjWriter.h"
#include "trisoup/CallBinaryVolumeData.h"
#include "trisoup/trisoupVolumetricDataCall.h"
#include "vislib/Trace.h"

namespace megamol::trisoup {
class TrisoupPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(TrisoupPluginInstance)

public:
    TrisoupPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("trisoup", "Plugin for rendering TriSoup mesh data") {
        vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL - 1);
    };

    ~TrisoupPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::trisoup::WavefrontObjWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::quartz::OSCBFix>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::trisoup::CallBinaryVolumeData>();
        this->call_descriptions.RegisterAutoDescription<megamol::trisoup::trisoupVolumetricDataCall>();
    }
};
} // namespace megamol::trisoup
