/*
 * Plugin200Instance.cpp
 * Copyright (C) 2015 by MegaMol Team
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include <cassert>

using namespace megamol::core;
using namespace megamol::core::utility::plugins;


/*
 * Plugin200Instance::~Plugin200Instance
 */
Plugin200Instance::~Plugin200Instance(void) {
    // intentionally empty
}


/*
 * Plugin200Instance::Plugin200Instance
 */
Plugin200Instance::Plugin200Instance(const char *asm_name, const char *description)
        : AbstractPluginInstance(asm_name, description) {
    // intentionally empty
}
