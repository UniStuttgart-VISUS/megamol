/*
 * mmstd_volume.cpp
 *
 * Copyright (C) 2009-2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "BuckyBall.h"
#include "DatRawWriter.h"
#include "DifferenceVolume.h"
#include "RaycastVolumeRenderer.h"
#include "VolumeSliceRenderer.h"
#include "VolumetricDataSource.h"

namespace megamol::stdplugin::volume {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "mmstd_volume",

              /* human-readable plugin description */
              "Provides modules for volume rendering"){
          };

    /** Dtor */
    virtual ~plugin_instance(void) {
    }

    /** Registers modules and calls */
    virtual void registerClasses(void) {

        // register modules here:
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::BuckyBall>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::DatRawWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::DifferenceVolume>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::RaycastVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::VolumeSliceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::VolumetricDataSource>();

        // register calls here:

    }
};
} // namespace megamol::stdplugin::volume
