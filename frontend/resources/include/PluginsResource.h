/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <vector>

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescriptionManager.h"

namespace megamol::frontend_resources {

struct PluginsResource {
    std::vector<core::factories::AbstractPluginInstance::ptr_type> plugins;

    core::factories::CallDescriptionManager all_call_descriptions;

    core::factories::ModuleDescriptionManager all_module_descriptions;
};

} // namespace megamol::frontend_resources
