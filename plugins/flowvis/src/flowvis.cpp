/*
 * flowvis.cpp
 * Copyright (C) 2018-2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "flowvis/flowvis.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"

#include "vislib/vislibversion.h"

#include "critical_points.h"
#include "glyph_data_reader.h"
#include "implicit_topology.h"
#include "implicit_topology_reader.h"
#include "implicit_topology_writer.h"
#include "line_strip.h"
#include "periodic_orbits.h"
#include "periodic_orbits_theisel.h"
#include "streamlines_2d.h"
#include "vector_field_reader.h"

#include "draw_texture_3d.h"
#include "draw_to_texture.h"
#include "glyph_renderer_2d.h"
#include "triangle_mesh_renderer_2d.h"
#include "triangle_mesh_renderer_3d.h"

#include "glyph_data_call.h"
#include "implicit_topology_call.h"
#include "matrix_call.h"
#include "mesh_data_call.h"
#include "mouse_click_call.h"
#include "triangle_mesh_call.h"
#include "vector_field_call.h"

/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "flowvis",

                /* human-readable plugin description */
                "Plugin for flow visualization") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::critical_points>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::glyph_data_reader>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::implicit_topology>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::implicit_topology_reader>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::implicit_topology_writer>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::line_strip>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::periodic_orbits>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::periodic_orbits_theisel>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::streamlines_2d>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::vector_field_reader>();

            // register renderer here:
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::draw_texture_3d>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::draw_to_texture>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::glyph_renderer_2d>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::triangle_mesh_renderer_2d>();
            this->module_descriptions.RegisterAutoDescription<megamol::flowvis::triangle_mesh_renderer_3d>();

            // register calls here:
            this->call_descriptions.RegisterAutoDescription<megamol::flowvis::glyph_data_call>();
            this->call_descriptions.RegisterAutoDescription<megamol::flowvis::implicit_topology_reader_call>();
            this->call_descriptions.RegisterAutoDescription<megamol::flowvis::implicit_topology_writer_call>();
            this->call_descriptions.RegisterAutoDescription<megamol::flowvis::matrix_call>();
            this->call_descriptions.RegisterAutoDescription<megamol::flowvis::mesh_data_call>();
            this->call_descriptions.RegisterAutoDescription<megamol::flowvis::mouse_click_call>();
            this->call_descriptions.RegisterAutoDescription<megamol::flowvis::triangle_mesh_call>();
            this->call_descriptions.RegisterAutoDescription<megamol::flowvis::vector_field_call>();
        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgPluginAPIVersion
 */
FLOWVIS_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
FLOWVIS_API
::megamol::core::utility::plugins::PluginCompatibilityInfo *
mmplgGetPluginCompatibilityInfo(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    // compatibility information with core and vislib
    using ::megamol::core::utility::plugins::PluginCompatibilityInfo;
    using ::megamol::core::utility::plugins::LibraryVersionInfo;

    PluginCompatibilityInfo *ci = new PluginCompatibilityInfo;
    ci->libs_cnt = 2;
    ci->libs = new LibraryVersionInfo[2];

    SetLibraryVersionInfo(ci->libs[0], "MegaMolCore",
        MEGAMOL_CORE_MAJOR_VER, MEGAMOL_CORE_MINOR_VER, MEGAMOL_CORE_COMP_REV, 0
#if defined(DEBUG) || defined(_DEBUG)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DEBUG_BUILD
#endif
#if defined(MEGAMOL_CORE_DIRTY) && (MEGAMOL_CORE_DIRTY != 0)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DIRTY_BUILD
#endif
        );

    SetLibraryVersionInfo(ci->libs[1], "vislib",
        vislib::VISLIB_VERSION_MAJOR, vislib::VISLIB_VERSION_MINOR, vislib::VISLIB_VERSION_REVISION, 0
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
FLOWVIS_API
void mmplgReleasePluginCompatibilityInfo(
        ::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
 * mmplgGetPluginInstance
 */
FLOWVIS_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
FLOWVIS_API
void mmplgReleasePluginInstance(
        ::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}
