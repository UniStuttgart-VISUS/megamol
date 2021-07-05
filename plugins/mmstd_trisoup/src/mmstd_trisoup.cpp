/*
 * mmstd_trisoup.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "TriSoupRenderer.h"
#include "TriSoupDataSource.h"
#include "WavefrontObjDataSource.h"
#include "WavefrontObjWriter.h"
#include "BlockVolumeMesh.h"
#include "volumetrics/VoluMetricJob.h"
#include "OSCBFix.h"
#include "CoordSysMarker.h"
#include "volumetrics/IsoSurface.h"
#include "CallBinaryVolumeData.h"
#include "CallVolumetricData.h"
#include "vislib/Trace.h"

namespace megamol::trisoup {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "mmstd_trisoup",

                /* human-readable plugin description */
                "Plugin for rendering TriSoup mesh data") {

            // here we could perform addition initialization
            vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL - 1);
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::TriSoupRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::TriSoupDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::WavefrontObjDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::WavefrontObjWriter>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::BlockVolumeMesh>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::volumetrics::VoluMetricJob>();
            this->module_descriptions.RegisterAutoDescription<megamol::quartz::OSCBFix>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::CoordSysMarker>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::volumetrics::IsoSurface>();

            // register calls here:
            this->call_descriptions.RegisterAutoDescription<megamol::trisoup::CallBinaryVolumeData>();
            this->call_descriptions.RegisterAutoDescription<megamol::trisoup::CallVolumetricData>();

        }
    };
} // namespace megamol::trisoup
