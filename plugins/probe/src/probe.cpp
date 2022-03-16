/**
 * MegaMol
 * Copyright (c) 2019-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "ConstructKDTree.h"
#include "ExtractCenterline.h"
#include "ExtractMesh.h"
#include "ExtractProbeGeometry.h"
#include "GenerateGlyphs.h"
#include "ManipulateMesh.h"
#include "PlaceProbes.h"
#include "ProbesToTable.h"
#include "SampleAlongProbes.h"
#include "SurfaceNets.h"
#include "probe/CallKDTree.h"
#include "probe/ProbeCalls.h"
#ifdef PROBE_HAS_OSPRAY
#include "OSPRayGlyphGeometry.h"
#endif
#include "ConstructHull.h"
#include "ElementColoring.h"
#include "ElementSampling.h"
#include "ExtractSkeleton.h"
#include "InjectClusterID.h"
#include "MeshSelector.h"
#include "ProbeClustering.h"
#include "ReconstructSurface.h"
#include "TableToProbes.h"
#include "TessellateBoundingBox.h"

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
        this->module_descriptions.RegisterAutoDescription<megamol::probe::ManipulateMesh>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe::ProbeToTable>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe::MeshSelector>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe::TableToProbes>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe::ProbeClustering>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe::ReconstructSurface>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe::TessellateBoundingBox>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe::InjectClusterID>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe::ElementSampling>();
#ifdef PROBE_HAS_OSPRAY
        this->module_descriptions.RegisterAutoDescription<megamol::probe::OSPRayGlyphGeometry>();
#endif
        this->module_descriptions.RegisterAutoDescription<megamol::probe::ExtractSkeleton>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe::ElementColoring>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe::ConstructHull>();


        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::probe::CallProbes>();
        this->call_descriptions.RegisterAutoDescription<megamol::probe::CallKDTree>();
    }
};
} // namespace megamol::probe
