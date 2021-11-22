/**
 * MegaMol
 * Copyright (c) 2018-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "CinematicView.h"
#include "OverlayRenderer.h"
#include "ReplacementRenderer.h"
#include "TimeLineRenderer.h"
#include "TrackingShotRenderer.h"

namespace megamol::cinematic_gl {
class CinematicGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(CinematicGLPluginInstance)

public:
    CinematicGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("cinematic_gl",
                  "The Cinematic plugin allows the video rendering (separate file per frame) of any rendering "
                  "output in MegaMol. By defining fixed keyframes for desired camera positions and specific "
                  "animation times, arbitrary tracking shots can be created."){};

    ~CinematicGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::cinematic_gl::TrackingShotRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::cinematic_gl::TimeLineRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::cinematic_gl::CinematicView>();
        this->module_descriptions.RegisterAutoDescription<megamol::cinematic_gl::ReplacementRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::cinematic_gl::OverlayRenderer>();

        // register calls
    }
};
} // namespace megamol::cinematic_gl
