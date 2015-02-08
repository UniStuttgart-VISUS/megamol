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
#include "utility/plugins/LegacyPlugin100Instance.h"
#include <algorithm>

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
    this->plugins.clear();
}


/*
 * PluginManager::LoadPlugin
 */
PluginManager::collection_type PluginManager::LoadPlugin(
        const std::basic_string<TCHAR>& filename,
        ::megamol::core::CoreInstance& coreInst) {
    PluginManager::collection_type rv;

    // load plugin assembly
    std::shared_ptr<vislib::sys::DynamicLinkLibrary> plugin_asm = std::make_shared<vislib::sys::DynamicLinkLibrary>();
    if (!plugin_asm->Load(filename.c_str())) {
        vislib::StringA msg;
        msg.Format("Cannot load plugin \"%s\"",
            plugin_asm->LastLoadErrorMessage().PeekBuffer());
        throw vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
    }

    // search for api entry
    int (*mmplgPluginAPIVersion)(void) = function_cast<int (*)()>(plugin_asm->GetProcAddress("mmplgPluginAPIVersion"));
    if ((mmplgPluginAPIVersion == nullptr)) {
        throw vislib::Exception("API entry not found", __FILE__, __LINE__);
    }

    // query api version
    int plgApiVer = mmplgPluginAPIVersion();
    if (plgApiVer == 100) {
        // load plugin as version 1.00
        rv = LegacyPlugin100Instance::ContinueLoad(filename, plugin_asm, coreInst);

    } else {
        // unsupported plugin version
        vislib::StringA msg;
        msg.Format("incompatible API version %d", plgApiVer);
        throw vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
    }

    if (rv.size() <= 0) {
        throw vislib::Exception("Generic return error", __FILE__, __LINE__);
    }

    for (auto p : rv) this->plugins.push_back(p);

    return rv;
}
