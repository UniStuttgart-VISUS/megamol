/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "RaycastVolumeRenderer.h"
#include "SurfaceLICRenderer.h"
#include "VolumeSliceRenderer.h"

namespace megamol::volume_gl {
class VolumeGLPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(VolumeGLPluginInstance)

public:
    VolumeGLPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("volume_gl", "Provides modules for volume rendering"){};

    ~VolumeGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::volume_gl::RaycastVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::volume_gl::VolumeSliceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::volume_gl::SurfaceLICRenderer>();

        // register calls
    }
};
} // namespace megamol::volume_gl
