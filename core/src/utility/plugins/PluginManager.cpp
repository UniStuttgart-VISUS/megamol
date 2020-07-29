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
        rv = Plugin100Instance::ContinueLoad(filename, plugin_asm, coreInst);

    } else if (plgApiVer == 200) {
        // load plugin as version 2.00
        rv = this->ContinueLoad200(filename, plugin_asm, coreInst);

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


namespace {
    void throw_exception(const char *msg, const char *file, unsigned int line) {
        throw vislib::Exception(msg, file, line);
    }
}


/*
 * PluginManager::ContinueLoad200
 */
PluginManager::collection_type PluginManager::ContinueLoad200(
        const std::basic_string<TCHAR> &path,
        std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib,
        CoreInstance& coreInst) {
    PluginManager::collection_type rv;

    // fetch plugin api function pointers
    Plugin200Instance::mmplgGetPluginCompatibilityInfo_funcptrtype mmplgGetPluginCompatibilityInfo
        = function_cast<Plugin200Instance::mmplgGetPluginCompatibilityInfo_funcptrtype>(lib->GetProcAddress("mmplgGetPluginCompatibilityInfo"));
    if (mmplgGetPluginCompatibilityInfo == nullptr) {
        throw vislib::Exception("API function \"mmplgGetPluginCompatibilityInfo\" not found", __FILE__, __LINE__);
    }
    Plugin200Instance::mmplgGetPluginInstance_funcptrtype mmplgGetPluginInstance
        = function_cast<Plugin200Instance::mmplgGetPluginInstance_funcptrtype>(lib->GetProcAddress("mmplgGetPluginInstance"));
    if (mmplgGetPluginInstance == nullptr) {
        throw vislib::Exception("API function \"mmplgGetPluginInstance\" not found", __FILE__, __LINE__);
    }
    Plugin200Instance::mmplgReleasePluginCompatibilityInfo_funcptrtype mmplgReleasePluginCompatibilityInfo
        = function_cast<Plugin200Instance::mmplgReleasePluginCompatibilityInfo_funcptrtype>(lib->GetProcAddress("mmplgReleasePluginCompatibilityInfo"));
    if (mmplgReleasePluginCompatibilityInfo == nullptr) {
        throw vislib::Exception("API function \"mmplgReleasePluginCompatibilityInfo\" not found", __FILE__, __LINE__);
    }
    Plugin200Instance::mmplgReleasePluginInstance_funcptrtype mmplgReleasePluginInstance
        = function_cast<Plugin200Instance::mmplgReleasePluginInstance_funcptrtype>(lib->GetProcAddress("mmplgReleasePluginInstance"));
    if (mmplgReleasePluginInstance == nullptr) {
        throw vislib::Exception("API function \"mmplgReleasePluginInstance\" not found", __FILE__, __LINE__);
    }

    // fetch compatibility information
    std::shared_ptr<PluginCompatibilityInfo> comp_info;
    {
        PluginCompatibilityInfo *pci(nullptr);
        try {
            pci = mmplgGetPluginCompatibilityInfo(&throw_exception);
            comp_info = std::shared_ptr<PluginCompatibilityInfo>(pci, mmplgReleasePluginCompatibilityInfo);
            pci = nullptr;
        } catch(...) {
            if (pci != nullptr) mmplgReleasePluginCompatibilityInfo(pci);
            pci = nullptr;
            throw;
        }
    }

    // check compatibility information
    if (comp_info) {
        bool MegaMolCore_compatibility_checked = false;
        bool vislib_compatibility_checked = false;
        vislib::VersionNumber mmcoreVer(MEGAMOL_CORE_VERSION);
        vislib::VersionNumber vislibVer(vislib::VISLIB_VERSION_MAJOR, vislib::VISLIB_VERSION_MINOR, vislib::VISLIB_VERSION_REVISION);

        for (unsigned int li = 0; li < comp_info->libs_cnt; li++) {
            LibraryVersionInfo &lvi = comp_info->libs[li];
            if (vislib::StringA("MegaMolCore").Equals(lvi.name)) {
                MegaMolCore_compatibility_checked = true;
                // mueller: where is the sense in having a separate variable which is sometimes wrong?
                //vislib::VersionNumber v(
                //    (lvi.version_len > 0) ? std::atoi(lvi.version[0].c_str()) : 0,
                //    (lvi.version_len > 1) ? std::atoi(lvi.version[1].c_str()) : 0,
                //    (lvi.version_len > 2) ? lvi.version[2].c_str() : 0);
                vislib::VersionNumber v(
                    (lvi.version.size() > 0) ? std::atoi(lvi.version[0].c_str()) : 0,
                    (lvi.version.size() > 1) ? std::atoi(lvi.version[1].c_str()) : 0,
                    (lvi.version.size() > 2) ? lvi.version[2].c_str() : 0);
                if ((v.GetMajorVersionNumber() != mmcoreVer.GetMajorVersionNumber())
                    && (v.GetMinorVersionNumber() != mmcoreVer.GetMinorVersionNumber())
                    && (v.GetRevisionNumber() != mmcoreVer.GetRevisionNumber())) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("Plugin %s seems incompatible with MegaMolCore: core \"%s\" != plugin \"%s\" ",
                        vislib::StringA(path.c_str()).PeekBuffer(),
                        mmcoreVer.ToStringA().PeekBuffer(),
                        v.ToStringA().PeekBuffer());
                    throw vislib::Exception("Plugin compatibility check failed", __FILE__, __LINE__);
                }

            } else if (vislib::StringA("vislib").Equals(lvi.name)) {
                vislib_compatibility_checked = true;
                // mueller: where is the sense in having a separate variable which is sometimes wrong?
                // vislib::VersionNumber v(
                //    (lvi.version_len > 0) ? std::atoi(lvi.version[0].c_str()) : 0,
                //    (lvi.version_len > 1) ? std::atoi(lvi.version[1].c_str()) : 0,
                //    (lvi.version_len > 2) ? lvi.version[2].c_str() : 0);
                vislib::VersionNumber v(
                    (lvi.version.size() > 0) ? std::atoi(lvi.version[0].c_str()) : 0,
                    (lvi.version.size() > 1) ? std::atoi(lvi.version[1].c_str()) : 0,
                    (lvi.version.size() > 2) ? lvi.version[2].c_str() : 0);
                if ((v.GetMajorVersionNumber() != vislibVer.GetMajorVersionNumber())
                    && (v.GetMinorVersionNumber() != vislibVer.GetMinorVersionNumber())
                    && (v.GetRevisionNumber() != vislibVer.GetRevisionNumber())) {
                    megamol::core::utility::log::Log::DefaultLog.WriteError("Plugin %s seems incompatible with vislib: vislib \"%s\" != plugin \"%s\" ",
                        vislib::StringA(path.c_str()).PeekBuffer(),
                        vislibVer.ToStringA().PeekBuffer(),
                        v.ToStringA().PeekBuffer());
                    throw vislib::Exception("Plugin compatibility check failed", __FILE__, __LINE__);
                }

            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteInfo("Plugin %s compatibility with %s is not checked",
                    vislib::StringA(path.c_str()).PeekBuffer(), lvi.name);
            }
        }

        if (!MegaMolCore_compatibility_checked) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("Plugin %s compatibility with MegaMolCore is not checked", vislib::StringA(path.c_str()).PeekBuffer());
        }
        if (!vislib_compatibility_checked) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("Plugin %s compatibility with vislib is not checked", vislib::StringA(path.c_str()).PeekBuffer());
        }

    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("Plugin %s did not provide compatibility information", vislib::StringA(path.c_str()).PeekBuffer());
    }

    // compatibility is given!
    comp_info.reset();

    // create plugin instance object
    plugin_ptr_type plugin;
    Plugin200Instance *plugin200;
    {
        AbstractPluginInstance *pi(nullptr);
        try {
            pi = mmplgGetPluginInstance(&throw_exception);
            plugin = plugin_ptr_type(pi, mmplgReleasePluginInstance);
            plugin200 = dynamic_cast<Plugin200Instance *>(pi);
            pi = nullptr;
        } catch(...) {
            if (pi != nullptr) mmplgReleasePluginInstance(pi);
            pi = nullptr;
            throw;
        }
    }

    // check plugin instance
    if (!plugin) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unable to create an instance of plugin %s.", vislib::StringA(path.c_str()).PeekBuffer());
        throw vislib::Exception("Plugin instantation error", __FILE__, __LINE__);
    }
    if (plugin200 == nullptr) {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Plugin %s created an instance object of incompatible type version (Not 200).", vislib::StringA(path.c_str()).PeekBuffer());
        throw vislib::Exception("Plugin instantation error", __FILE__, __LINE__);
    }

    plugin200->SetAssemblyFileName(path);

    // connect static objects
    //plugin200->connectStatics(Plugin200Instance::StaticConnectorType::Log, static_cast<void*>(&megamol::core::utility::log::Log::DefaultLog));
    plugin200->connectStatics(
        Plugin200Instance::StaticConnectorType::Log, static_cast<void*>(&megamol::core::utility::log::Log::DefaultLog));

    // initialize factories
    plugin200->GetModuleDescriptionManager();

    // store library object
    plugin200->store_lib(lib);

    // report success
    rv.push_back(plugin);
    megamol::core::utility::log::Log::DefaultLog.WriteInfo("Plugin %s loaded: %u Modules, %u Calls",
        plugin->GetAssemblyName().c_str(),
        plugin->GetModuleDescriptionManager().Count(),
        plugin->GetCallDescriptionManager().Count());

    // connect stuff to the core instance
    plugin200->callRegisterAtCoreInstanceFunctions(coreInst);

    return rv;
}
