/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "TriSoupRenderer.h"
#include "ModernTrisoupRenderer.h"
#include "vislib/Trace.h"

namespace megamol::trisoup_gl {
    class TrisoupGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(TrisoupGLPluginInstance)

    public:
    TrisoupGLPluginInstance()
                : megamol::core::utility::plugins::AbstractPluginInstance(
                      "trisoup_gl", "Plugin for rendering TriSoup mesh data") {
            vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL - 1);
        };

        ~TrisoupGLPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup_gl::TriSoupRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup_gl::ModernTrisoupRenderer>();
        }
    };
} // namespace megamol::trisoup
