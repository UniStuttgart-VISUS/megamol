/*
 * probe.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"
#include "ExtractMesh.h"
#include "PlaceProbes.h"
#include "SampleAlongProbes.h"
#include "ProbeCalls.h"
#include "ExtractProbeGeometry.h"
#include "CallKDTree.h"
#include "GenerateGlyphs.h"
#include "SurfaceNets.h"
#include "ExtractCenterline.h"
#include "ConstructKDTree.h"
//#include "ManipulateMesh.h"
#ifdef PROBE_HAS_OSPRAY
#include "OSPRayGlyphGeometry.h"
#endif

namespace megamol::probe {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "probe",

                /* human-readable plugin description */
                "Putting probes into data and render glyphs at the end of the probe.") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:

            this->module_descriptions.RegisterAutoDescription<megamol::probe::ExtractMesh>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::PlaceProbes>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::SampleAlongPobes>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::ExtractProbeGeometry>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::GenerateGlyphs>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::SurfaceNets>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::ExtractCenterline>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::ConstructKDTree>();
            //this->module_descriptions.RegisterAutoDescription<megamol::probe::ManipulateMesh>();
#ifdef PROBE_HAS_OSPRAY
            this->module_descriptions.RegisterAutoDescription<megamol::probe::OSPRayGlyphGeometry>();
#endif

            // register calls here:

            this->call_descriptions.RegisterAutoDescription<megamol::probe::CallProbes>();
            this->call_descriptions.RegisterAutoDescription<megamol::probe::CallKDTree>();

        }
    };
} // namespace megamol::probe
