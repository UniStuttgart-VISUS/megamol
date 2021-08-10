/**
 * MegaMol
 * Copyright (c) 2018-2021, MegaMol Dev Team
 * All rights reserved.
 */

// TODO: Vislib must die!!!
// Vislib includes Windows.h. This crashes when somebody else (i.e. zmq) is using Winsock2.h, but the vislib include
// is first without defining WIN32_LEAN_AND_MEAN. This define is the only thing we need from stdafx.h, include could be
// removed otherwise.
#include "stdafx.h"

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "CallKeyframeKeeper.h"
#include "CinematicView.h"
#include "KeyframeKeeper.h"
#include "OverlayRenderer.h"
#include "ReplacementRenderer.h"
#include "TimeLineRenderer.h"
#include "TrackingShotRenderer.h"

namespace megamol::cinematic {
    class CinematicPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
        REGISTERPLUGIN(CinematicPluginInstance)

    public:
        CinematicPluginInstance()
                : megamol::core::utility::plugins::AbstractPluginInstance("Cinematic",
                      "The Cinematic plugin allows the video rendering (separate file per frame) of any rendering "
                      "output in MegaMol. By defining fixed keyframes for desired camera positions and specific "
                      "animation times, arbitrary tracking shots can be created."){};

        ~CinematicPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::cinematic::TrackingShotRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::cinematic::TimeLineRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::cinematic::KeyframeKeeper>();
            this->module_descriptions.RegisterAutoDescription<megamol::cinematic::CinematicView>();
            this->module_descriptions.RegisterAutoDescription<megamol::cinematic::ReplacementRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::cinematic::OverlayRenderer>();

            // register calls
            this->call_descriptions.RegisterAutoDescription<megamol::cinematic::CallKeyframeKeeper>();
        }
    };
} // namespace megamol::cinematic
