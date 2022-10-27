/*
 * CoreInstance.cpp
 *
 * Copyright (C) 2008, 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include <memory>
#include <sstream>
#include <string>

#include "mmcore/AbstractSlot.h"
#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/Module.h"
#include "mmcore/factories/PluginRegister.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/utility/buildinfo/BuildInfo.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/Array.h"
#include "vislib/MissingImplementationException.h"
#include "vislib/String.h"
#include "vislib/StringConverter.h"
#include "vislib/sys/PerformanceCounter.h"


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
void megamol::core::CoreInstance::Initialise() {
    translateShaderPaths(config);
}


void megamol::core::CoreInstance::translateShaderPaths(megamol::core::utility::Configuration const& config) {
    auto const v_paths = config.ShaderDirectories();

    shaderPaths.resize(v_paths.Count());

    for (size_t idx = 0; idx < v_paths.Count(); ++idx) {
        shaderPaths[idx] = std::filesystem::path(v_paths[idx].PeekBuffer());
    }
}


std::vector<std::filesystem::path> megamol::core::CoreInstance::GetShaderPaths() const {
    return shaderPaths;
}
