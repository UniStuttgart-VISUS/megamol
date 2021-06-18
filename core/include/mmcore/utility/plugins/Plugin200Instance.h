/**
 * MegaMol
 * Copyright (c) 2015-2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200INSTANCE_H_INCLUDED
#define MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200INSTANCE_H_INCLUDED
#pragma once

#include "mmcore/utility/plugins/AbstractPluginInstance.h"

namespace megamol::core::utility::plugins {

    /**
     * Base class for Instances of Plugins using the 2.0 API interface
     */
    class [[deprecated("Use AbstractPluginInstance directly!")]] Plugin200Instance : public AbstractPluginInstance {
        using AbstractPluginInstance::AbstractPluginInstance;
    };

} // namespace megamol::core::utility::plugins

#endif // MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200INSTANCE_H_INCLUDED
