/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

// TODO: Vislib must die!!!
// Vislib includes Windows.h. This crashes when somebody else (i.e. zmq) is using Winsock2.h, but the vislib include
// is first without defining WIN32_LEAN_AND_MEAN. This define is the only thing we need from stdafx.h, include could be
// removed otherwise.
#include "stdafx.h"

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "BuckyBall.h"
#include "DatRawWriter.h"
#include "DifferenceVolume.h"
#include "RaycastVolumeRenderer.h"
#include "VolumeSliceRenderer.h"
#include "VolumetricDataSource.h"

namespace megamol::stdplugin::volume {
class VolumePluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(VolumePluginInstance)

public:
    VolumePluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "mmstd_volume", "Provides modules for volume rendering"){};

    ~VolumePluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::BuckyBall>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::DatRawWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::DifferenceVolume>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::RaycastVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::VolumeSliceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::VolumetricDataSource>();

        // register calls
    }
};
} // namespace megamol::stdplugin::volume
