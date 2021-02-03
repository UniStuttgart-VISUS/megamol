/*
 * PluginManager.cpp
 * Copyright (C) 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "utility/plugins/PluginManager.h"
#include "vislib/sys/DynamicLinkLibrary.h"
#include "vislib/functioncast.h"
#include "mmcore/CoreInstance.h"
#include "utility/plugins/Plugin100Instance.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include <algorithm>
#include "vislib/String.h"
#include "mmcore/versioninfo.h"
#include "vislib/VersionNumber.h"
#include "vislib/vislibversion.h"

using namespace megamol::core;
using namespace megamol::core::utility::plugins;


/*
 * PluginManager::PluginManager
 */
PluginManager::PluginManager(void) : plugins() {
}


/*
 * PluginManager::~PluginManager
 */
PluginManager::~PluginManager(void) {
    // store the plugin libraries extra to ensure these get freed at the very
    // end!
    std::vector<std::shared_ptr< ::vislib::sys::DynamicLinkLibrary> > libs;
    for (auto plugin : this->plugins) {
        const Plugin200Instance *p200 = dynamic_cast<const Plugin200Instance*>(plugin.get());
        if (p200) libs.push_back(p200->get_lib());
    }

    // now free the plugin instances
    this->plugins.clear();

    // finally free the libraries
    libs.clear();
}

/*
 * PluginManager::LoadPlugin
 */
Plugin200Instance::ptr_type PluginManager::LoadPlugin(
        const std::shared_ptr<AbstractPluginDescriptor>& pluginDescriptor,
        ::megamol::core::CoreInstance& coreInst) {

    auto plugin = pluginDescriptor->create();

    plugin->SetAssemblyFileName("FILENAME_REMOVED_IN_STATIC_BUILD"); // TODO

    // initialize factories
    plugin->GetModuleDescriptionManager();

    // store library object
    //plugin->store_lib(lib); // TODO update for static plugins

    // report success
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Plugin %s loaded: %u Modules, %u Calls",
                             plugin->GetAssemblyName().c_str(),
                             plugin->GetModuleDescriptionManager().Count(),
                             plugin->GetCallDescriptionManager().Count());

    // connect stuff to the core instance
    plugin->callRegisterAtCoreInstanceFunctions(coreInst);

    this->plugins.push_back(plugin);

    return plugin;
}
