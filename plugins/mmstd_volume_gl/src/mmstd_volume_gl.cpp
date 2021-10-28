/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "RaycastVolumeRenderer.h"
#include "VolumeSliceRenderer.h"

namespace megamol::stdplugin::volume {
class VolumeGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(VolumeGLPluginInstance)

public:
    VolumeGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "mmstd_volume_gl", "Provides modules for volume rendering"){};

    ~VolumeGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::RaycastVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::VolumeSliceRenderer>();

        // register calls
    }
};
} // namespace megamol::stdplugin::volume
