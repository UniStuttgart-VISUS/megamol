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
#include "ProbeGlyphRenderer.h"
#include "ProbeDetailViewRenderer.h"
#include "ProbeGlCalls.h"
#include "ProbeHullRenderer.h"
#include "ProbeInteraction.h"
#include "ProbeRenderer.h"
#include "ProbeShellElementsRenderer.h"

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
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeGlyphRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeDetailViewRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeInteraction>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::FilterByProbe>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeHullRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::PrecomputeGlyphTextures>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ProbeShellElementsRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::probe_gl::ComputeDistance>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::probe_gl::CallProbeInteraction>();
    }
};
} // namespace megamol::probe_gl
