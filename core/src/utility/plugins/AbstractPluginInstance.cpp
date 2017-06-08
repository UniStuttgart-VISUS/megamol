/*
 * AbstractPluginInstance.cpp
 * Copyright (C) 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include <cassert>

using namespace megamol::core;
using namespace megamol::core::utility::plugins;


/*
 * AbstractPluginInstance::GetAssemblyName
 */
const std::string& AbstractPluginInstance::GetAssemblyName(void) const {
    return this->asm_name;
}


/*
 * AbstractPluginInstance::GetDescription
 */
const std::string& AbstractPluginInstance::GetDescription(void) const {
    return this->description;
}


/*
 * AbstractPluginInstance::AbstractPluginInstance
 */
AbstractPluginInstance::AbstractPluginInstance(const char *asm_name,
        const char *description) : factories::AbstractAssemblyInstance(),
        asm_name(asm_name), description(description) {
    assert(asm_name != nullptr);
    // intentionally empty
}


/*
 * AbstractPluginInstance::~AbstractPluginInstance
 */
AbstractPluginInstance::~AbstractPluginInstance(void) {
    // intentionally empty
}
