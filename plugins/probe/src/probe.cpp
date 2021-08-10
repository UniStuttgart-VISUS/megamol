/**
 * MegaMol
 * Copyright (c) 2019-2021, MegaMol Dev Team
 * All rights reserved.
 */

// TODO: Vislib must die!!!
// Vislib includes Windows.h. This crashes when somebody else (i.e. zmq) is using Winsock2.h, but the vislib include
// is first without defining WIN32_LEAN_AND_MEAN. This define is the only thing we need from stdafx.h, include could be
// removed otherwise.
#include "stdafx.h"

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "CallKDTree.h"
#include "ConstructKDTree.h"
#include "ExtractCenterline.h"
#include "ExtractMesh.h"
#include "ExtractProbeGeometry.h"
#include "GenerateGlyphs.h"
#include "PlaceProbes.h"
#include "ProbeCalls.h"
#include "SampleAlongProbes.h"
#include "SurfaceNets.h"
//#include "ManipulateMesh.h"
#ifdef PROBE_HAS_OSPRAY
#include "OSPRayGlyphGeometry.h"
#endif

namespace megamol::probe {
    class ProbePluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
        REGISTERPLUGIN(ProbePluginInstance)

    public:
        ProbePluginInstance()
                : megamol::core::utility::plugins::AbstractPluginInstance(
                      "probe", "Putting probes into data and render glyphs at the end of the probe."){};

        ~ProbePluginInstance(void) override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::probe::ExtractMesh>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::PlaceProbes>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::SampleAlongPobes>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::ExtractProbeGeometry>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::GenerateGlyphs>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::SurfaceNets>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::ExtractCenterline>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::ConstructKDTree>();
            // this->module_descriptions.RegisterAutoDescription<megamol::probe::ManipulateMesh>();
#ifdef PROBE_HAS_OSPRAY
            this->module_descriptions.RegisterAutoDescription<megamol::probe::OSPRayGlyphGeometry>();
#endif

            // register calls
            this->call_descriptions.RegisterAutoDescription<megamol::probe::CallProbes>();
            this->call_descriptions.RegisterAutoDescription<megamol::probe::CallKDTree>();
        }
    };
} // namespace megamol::probe
