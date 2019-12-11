/*
 * gui.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "gui/gui.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "GUIView.h"
#include "OverlayRenderer.h"


/* anonymous namespace hides this type from any other object files */
namespace {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "gui",

              /* human-readable plugin description */
              "Graphical user interface plugin"){

              // here we could perform addition initialization
          };
    /** Dtor */
    virtual ~plugin_instance(void) {
        // here we could perform addition de-initialization
    }
    /** Registers modules and calls */
    virtual void registerClasses(void) {
        this->module_descriptions.RegisterAutoDescription<megamol::gui::GUIView>();
        this->module_descriptions.RegisterAutoDescription<megamol::gui::OverlayRenderer>();
    }

    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
};
} // namespace


/*
 * mmplgPluginAPIVersion
 */
GUI_API int mmplgPluginAPIVersion(void){MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion}


/*
 * mmplgGetPluginCompatibilityInfo
 */
GUI_API ::megamol::core::utility::plugins::PluginCompatibilityInfo* mmplgGetPluginCompatibilityInfo(
    ::megamol::core::utility::plugins::ErrorCallback onError) {
    // compatibility information with core and vislib
    using ::megamol::core::utility::plugins::LibraryVersionInfo;
    using ::megamol::core::utility::plugins::PluginCompatibilityInfo;

    PluginCompatibilityInfo* ci = new PluginCompatibilityInfo;
    ci->libs_cnt = 2;
    ci->libs = new LibraryVersionInfo[2];

    SetLibraryVersionInfo(
        ci->libs[0], "MegaMolCore", MEGAMOL_CORE_MAJOR_VER, MEGAMOL_CORE_MINOR_VER, MEGAMOL_CORE_COMP_REV,
        0
#if defined(DEBUG) || defined(_DEBUG)
            | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DEBUG_BUILD
#endif
#if defined(MEGAMOL_CORE_DIRTY) && (MEGAMOL_CORE_DIRTY != 0)
            | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DIRTY_BUILD
#endif
    );

    SetLibraryVersionInfo(ci->libs[1], "vislib", vislib::VISLIB_VERSION_MAJOR, vislib::VISLIB_VERSION_MINOR,
        vislib::VISLIB_VERSION_REVISION,
        0
#if defined(DEBUG) || defined(_DEBUG)
            | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DEBUG_BUILD
#endif
#if defined(VISLIB_DIRTY_BUILD) && (VISLIB_DIRTY_BUILD != 0)
            | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DIRTY_BUILD
#endif
    );
    //
    // If you want to test additional compatibilties, add the corresponding versions here
    //

    return ci;
}


/*
 * mmplgReleasePluginCompatibilityInfo
 */
GUI_API
void mmplgReleasePluginCompatibilityInfo(::megamol::core::utility::plugins::PluginCompatibilityInfo* ci){
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)}


/*
 * mmplgGetPluginInstance
 */
GUI_API ::megamol::core::utility::plugins::AbstractPluginInstance* mmplgGetPluginInstance(
    ::megamol::core::utility::plugins::ErrorCallback onError){
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)}


/*
 * mmplgReleasePluginInstance
 */
GUI_API void mmplgReleasePluginInstance(::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}
