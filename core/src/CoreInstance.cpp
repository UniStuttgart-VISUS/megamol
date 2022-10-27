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

/*****************************************************************************/

/*
 * megamol::core::CoreInstance::CoreInstance
 */
megamol::core::CoreInstance::CoreInstance(void)
        : config()
        , plugins()
        , all_call_descriptions()
        , all_module_descriptions() {

    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Core Instance created");
}


/*
 * megamol::core::CoreInstance::~CoreInstance
 */
megamol::core::CoreInstance::~CoreInstance(void) {
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Core Instance destroyed");

    // we need to manually clean up all data structures in the right order!
    // first view- and job-descriptions
    // then factories
    this->all_module_descriptions.Shutdown();
    this->all_call_descriptions.Shutdown();
    // finally plugins
    this->plugins.clear();
}


/*
 * megamol::core::CoreInstance::GetCallDescriptionManager
 */
const megamol::core::factories::CallDescriptionManager& megamol::core::CoreInstance::GetCallDescriptionManager(
    void) const {
    return this->all_call_descriptions;
}


/*
 * megamol::core::CoreInstance::GetModuleDescriptionManager
 */
const megamol::core::factories::ModuleDescriptionManager& megamol::core::CoreInstance::GetModuleDescriptionManager(
    void) const {
    return this->all_module_descriptions;
}


/*
 * megamol::core::CoreInstance::Initialise
 */
void megamol::core::CoreInstance::Initialise() {

    // loading plugins
    for (const auto& plugin : factories::PluginRegister::getAll()) {
        this->loadPlugin(plugin);
    }

    translateShaderPaths(config);
}


/*
 * megamol::core::CoreInstance::loadPlugin
 */
void megamol::core::CoreInstance::loadPlugin(
    const std::shared_ptr<factories::AbstractPluginDescriptor>& pluginDescriptor) {

    try {

        auto new_plugin = pluginDescriptor->create();
        this->plugins.push_back(new_plugin);

        // report success
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("Plugin \"%s\" loaded: %u Modules, %u Calls",
            new_plugin->GetObjectFactoryName().c_str(), new_plugin->GetModuleDescriptionManager().Count(),
            new_plugin->GetCallDescriptionManager().Count());

        for (auto md : new_plugin->GetModuleDescriptionManager()) {
            try {
                this->all_module_descriptions.Register(md);
            } catch (const std::invalid_argument&) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Failed to load module description \"%s\": Naming conflict", md->ClassName());
            }
        }
        for (auto cd : new_plugin->GetCallDescriptionManager()) {
            try {
                this->all_call_descriptions.Register(cd);
            } catch (const std::invalid_argument&) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "Failed to load call description \"%s\": Naming conflict", cd->ClassName());
            }
        }

    } catch (const vislib::Exception& vex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to load Plugin: %s (%s, &d)", vex.GetMsgA(), vex.GetFile(), vex.GetLine());
    } catch (const std::exception& ex) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to load Plugin: %s", ex.what());
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to load Plugin: unknown exception");
    }
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
