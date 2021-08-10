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

#include "BlockVolumeMesh.h"
#include "CallBinaryVolumeData.h"
#include "CallVolumetricData.h"
#include "CoordSysMarker.h"
#include "OSCBFix.h"
#include "TriSoupDataSource.h"
#include "TriSoupRenderer.h"
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
                      "mmstd_trisoup", "Plugin for rendering TriSoup mesh data") {
            vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL - 1);
        };

        ~TrisoupPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::TriSoupRenderer>();
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
            this->call_descriptions.RegisterAutoDescription<megamol::trisoup::CallVolumetricData>();
        }
    };
} // namespace megamol::trisoup
