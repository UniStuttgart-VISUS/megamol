/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "ComputeDistance.h"
#include "FilterByProbe.h"
#include "PrecomputeGlyphTextures.h"
#include "ProbeBillboardGlyphMaterial.h"
#include "ProbeBillboardGlyphRenderTasks.h"
#include "ProbeDetailViewRenderTasks.h"
#include "ProbeGlCalls.h"
#include "ProbeHullRenderTasks.h"
#include "ProbeInteraction.h"
#include "ProbeRenderTasks.h"
#include "ProbeShellElementsRenderTasks.h"

namespace megamol::probe_gl {
class ProbeGlPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ProbeGlPluginInstance)
public:
    ProbeGlPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("probe_gl", "The probe_gl plugin."){};

    ~ProbeGlPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeBillboardGlyphMaterial>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeBillboardGlyphRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeDetailViewRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeInteraction>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::FilterByProbe>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeHullRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::PrecomputeGlyphTextures>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeShellElementsRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ComputeDistance>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::probe_gl::CallProbeInteraction>();
    }
};
} // namespace megamol::probe_gl
