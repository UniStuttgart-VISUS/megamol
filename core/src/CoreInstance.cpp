/*
 * CoreInstance.cpp
 *
 * Copyright (C) 2008, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "mmcore/CoreInstance.h"
#include "mmcore/utility/log/Log.h"

/*
 * megamol::core::CoreInstance::CoreInstance
 */
megamol::core::CoreInstance::CoreInstance(void) : config() {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Core Instance created");
}

/*
 * megamol::core::CoreInstance::~CoreInstance
 */
megamol::core::CoreInstance::~CoreInstance(void) {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Core Instance destroyed");
}

/*
 * megamol::core::CoreInstance::Initialise
 */
void megamol::core::CoreInstance::Initialise() {}
