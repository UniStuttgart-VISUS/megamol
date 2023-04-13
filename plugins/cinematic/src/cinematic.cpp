/**
 * MegaMol
 * Copyright (c) 2018-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "KeyframeKeeper.h"
#include "cinematic/CallKeyframeKeeper.h"

namespace megamol::cinematic {
class CinematicPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(CinematicPluginInstance)

public:
    CinematicPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("cinematic",
                  "The Cinematic plugin allows the video rendering (separate file per frame) of any rendering "
                  "output in MegaMol. By defining fixed keyframes for desired camera positions and specific "
                  "animation times, arbitrary tracking shots can be created."){};

    ~CinematicPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::cinematic::KeyframeKeeper>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::cinematic::CallKeyframeKeeper>();
    }
};
} // namespace megamol::cinematic
