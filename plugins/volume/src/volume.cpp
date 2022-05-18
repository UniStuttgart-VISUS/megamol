/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "BuckyBall.h"
#include "DatRawWriter.h"
#include "DifferenceVolume.h"
#include "VolumetricDataSource.h"

namespace megamol::volume {
class VolumePluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(VolumePluginInstance)

public:
    VolumePluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "volume", "Provides modules for volume rendering"){};

    ~VolumePluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::volume::BuckyBall>();
        this->module_descriptions.RegisterAutoDescription<megamol::volume::DatRawWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::volume::DifferenceVolume>();
        this->module_descriptions.RegisterAutoDescription<megamol::volume::VolumetricDataSource>();

        // register calls
    }
};
} // namespace megamol::volume
