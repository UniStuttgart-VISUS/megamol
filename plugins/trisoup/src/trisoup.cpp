/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "BlockVolumeMesh.h"
#include "CallBinaryVolumeData.h"
#include "trisoup/trisoupVolumetricDataCall.h"
#include "CoordSysMarker.h"
#include "OSCBFix.h"
#include "TriSoupDataSource.h"
#include "WavefrontObjDataSource.h"
#include "WavefrontObjWriter.h"
#include "vislib/Trace.h"
#include "volumetrics/IsoSurface.h"
#include "volumetrics/VoluMetricJob.h"

namespace megamol::trisoup {
    class TrisoupPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
        REGISTERPLUGIN(TrisoupPluginInstance)

    public:
        TrisoupPluginInstance()
                : megamol::core::utility::plugins::AbstractPluginInstance(
                      "trisoup", "Plugin for rendering TriSoup mesh data") {
            vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL - 1);
        };

        ~TrisoupPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::TriSoupDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::WavefrontObjDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::WavefrontObjWriter>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::BlockVolumeMesh>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::volumetrics::VoluMetricJob>();
            this->module_descriptions.RegisterAutoDescription<megamol::quartz::OSCBFix>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::CoordSysMarker>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::volumetrics::IsoSurface>();

            // register calls
            this->call_descriptions.RegisterAutoDescription<megamol::trisoup::CallBinaryVolumeData>();
            this->call_descriptions.RegisterAutoDescription<megamol::trisoup::trisoupVolumetricDataCall>();
        }
    };
} // namespace megamol::trisoup
