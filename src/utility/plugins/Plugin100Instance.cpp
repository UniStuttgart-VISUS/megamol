/*
 * LegacyPlugin100Instance.cpp
 * Copyright (C) 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "utility/plugins/Plugin100Instance.h"
#include <cassert>
#include "mmcore/CoreInstance.h"
#include "vislib/functioncast.h"
#include "vislib/vislibversion.h"
#include "vislib/Trace.h"

using namespace megamol::core;
using namespace megamol::core::utility::plugins;


/*
 * Plugin100Instance::ContinueLoad
 */
PluginManager::collection_type Plugin100Instance::ContinueLoad(
        const std::basic_string<TCHAR> &path,
        std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib,
        CoreInstance& coreInst) {

    // test core compatibility
    const void* (*mmplgCoreCompatibilityValue)(void) = function_cast<const void* (*)()>(lib->GetProcAddress("mmplgCoreCompatibilityValue"));
    if ((mmplgCoreCompatibilityValue == nullptr)) {
        throw vislib::Exception("API function \"mmplgCoreCompatibilityValue\" not found", __FILE__, __LINE__);
    }
    const mmplgCompatibilityValues *compVal = static_cast<const mmplgCompatibilityValues*>(mmplgCoreCompatibilityValue());
    // test core version
    if ((compVal->size != sizeof(mmplgCompatibilityValues)) || (compVal->mmcoreRev != MEGAMOL_CORE_COMP_REV)) {
        SIZE_T rev = compVal->mmcoreRev;
        vislib::StringA msg;
        msg.Format("core version mismatch (%d from Core; %d from Plugin)", static_cast<int>(MEGAMOL_CORE_COMP_REV), static_cast<int>(rev));
        throw vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
    }
    // test vislib version (which is currently broken!)
    if ((compVal->size != sizeof(mmplgCompatibilityValues)) || ((compVal->vislibRev != 0) && (compVal->vislibRev != VISLIB_VERSION_REVISION))) {
        SIZE_T rev = compVal->vislibRev;
        vislib::StringA msg;
        msg.Format("vislib version mismatch (%d from Core; %d from Plugin)", static_cast<int>(VISLIB_VERSION_REVISION), static_cast<int>(rev));
        throw vislib::Exception(msg.PeekBuffer(), __FILE__, __LINE__);
    }

    // now we are good to load the plugin!

    // connect static objects
    bool (*mmplgConnectStatics)(int, void*) = function_cast<bool (*)(int, void*)>(lib->GetProcAddress("mmplgConnectStatics"));
    if (mmplgConnectStatics != NULL) {
        bool rv;
        rv = mmplgConnectStatics(1, static_cast<void*>(&vislib::sys::Log::DefaultLog));
        VLTRACE(VISLIB_TRCELVL_INFO, "Plug-in connect log: %s\n", rv ? "true" : "false");
        vislib::SmartPtr<vislib::StackTrace> stackManager(vislib::StackTrace::Manager());
        rv = mmplgConnectStatics(2, static_cast<void*>(&stackManager));
        VLTRACE(VISLIB_TRCELVL_INFO, "Plug-in connect stacktrace: %s\n", rv ? "true" : "false");
    }

    // load plugin description
    const char * (*mmplgPluginName)(void) = function_cast<const char * (*)()>(lib->GetProcAddress("mmplgPluginName"));
    const char * (*mmplgPluginDescription)(void) = function_cast<const char * (*)()>(lib->GetProcAddress("mmplgPluginDescription"));
    if ((mmplgPluginName == nullptr)/* || (mmplgPluginDescription == NULL)*/) {
        throw vislib::Exception("API name/description functions not found", __FILE__, __LINE__);
    }
    vislib::StringA plgName(mmplgPluginName());
    if (plgName.IsEmpty()) {
        throw vislib::Exception("Plugin does not export a name", __FILE__, __LINE__);
    }
    vislib::StringA plgDesc;
    if (mmplgPluginDescription != nullptr) {
        plgDesc = mmplgPluginDescription();
    }

    // From here on, plugin loading cannot fail any more
    // (except for unexpected errors)

    Plugin100Instance *lp100i = new Plugin100Instance(plgName.PeekBuffer(), plgDesc.PeekBuffer(), lib);
    lp100i->SetAssemblyFileName(path);
    PluginManager::collection_type rv;
    rv.push_back(PluginManager::plugin_ptr_type(lp100i));

    // load exported modules and calls (descriptions)
    int (*mmplgModuleCount)(void) = function_cast<int (*)()>(lib->GetProcAddress("mmplgModuleCount"));
    void* (*mmplgModuleDescription)(int) = function_cast<void* (*)(int)>(lib->GetProcAddress("mmplgModuleDescription"));
    int (*mmplgCallCount)(void) = function_cast<int (*)()>(lib->GetProcAddress("mmplgCallCount"));
    void* (*mmplgCallDescription)(int) = function_cast<void* (*)(int)>(lib->GetProcAddress("mmplgCallDescription"));

    // load module descriptions
    int modCnt = ((mmplgModuleCount == NULL) || (mmplgModuleDescription == NULL)) ? 0 : mmplgModuleCount();
    int modCntVal = 0;
    for (int i = 0; i < modCnt; i++) {
        void * modPtr = mmplgModuleDescription(i);
        if (modPtr == NULL) continue;
        factories::ModuleDescription::ptr mdp(static_cast<factories::ModuleDescription*>(modPtr));
        lp100i->module_descriptions.Register(mdp);
        modCntVal++;
    }

    // load call descriptions
    int callCnt = ((mmplgCallCount == NULL) || (mmplgCallDescription == NULL)) ? 0 : mmplgCallCount();
    int callCntVal = 0;
    for (int i = 0; i < callCnt; i++) {
        void * callPtr = mmplgCallDescription(i);
        if (callPtr == NULL) continue;
        factories::CallDescription::ptr cdp(static_cast<factories::CallDescription*>(callPtr));
        lp100i->call_descriptions.Register(cdp);
        callCntVal++;
    }

    return rv;
}


/*
 * Plugin100Instance::Plugin100Instance
 */
Plugin100Instance::Plugin100Instance(const char *asm_name,
        const char *description, 
        std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib)
        : AbstractPluginInstance(asm_name, description), lib(lib) {
    assert(asm_name != nullptr);
    assert(this->lib);
    // intentionally empty
}


/*
 * Plugin100Instance::~Plugin100Instance
 */
Plugin100Instance::~Plugin100Instance(void) {
    // Descriptions must be freed before the plugin lib is unloaded.
    // Thus, we manually free the descriptions first.
    this->module_descriptions.Shutdown();
    this->call_descriptions.Shutdown();
    this->lib->Free();
}
