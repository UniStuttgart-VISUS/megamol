/*
 * ng_mesh.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mesh/mesh.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

//#include "mesh/CallmeshRenderBatches.h"
#include "RenderMDIMesh.h"
//#include "meshDebugDataSource.h"

#include "mesh/CallGPUMeshData.h"
#include "mesh/CallGPUMaterialData.h"
#include "mesh/CallGPURenderTaskData.h"
#include "mesh/CallGltfData.h"

#include "DebugGPUMeshDataSource.h"
#include "DebugGPUMaterialDataSource.h"
#include "DebugGPURenderTaskDataSource.h"
#include "gltf/glTFFileLoader.h"
#include "gltf/glTFMaterialDataSource.h"
#include "gltf/glTFMeshesDataSource.h"
#include "gltf/glTFRenderTasksDataSource.h"
#include "mesh/SimpleGPUMtlDataSource.h"


/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "ng_mesh",

                /* human-readable plugin description */
                "Plugin for rendering meshes.") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:
			this->module_descriptions.RegisterAutoDescription<megamol::mesh::RenderMDIMesh>();
			this->module_descriptions.RegisterAutoDescription<megamol::mesh::DebugGPUMeshDataSource>();
			this->module_descriptions.RegisterAutoDescription<megamol::mesh::DebugGPUMaterialDataSource>();
			this->module_descriptions.RegisterAutoDescription<megamol::mesh::DebugGPURenderTaskDataSource>();
			this->module_descriptions.RegisterAutoDescription<megamol::mesh::GlTFFileLoader>();
			this->module_descriptions.RegisterAutoDescription<megamol::mesh::GlTFMeshesDataSource>();
			this->module_descriptions.RegisterAutoDescription<megamol::mesh::GlTFRenderTasksDataSource>();
			this->module_descriptions.RegisterAutoDescription<megamol::mesh::SimpleGPUMtlDataSource>();

            //
            // TODO: Register your plugin's modules here
            // like:
            //   this->module_descriptions.RegisterAutoDescription<megamol::ng_mesh::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::ng_mesh::MyModule2>();
            //   ...
            //

            // register calls here:
			this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPUMeshData>();
			this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPUMaterialData>();
			this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPURenderTaskData>();
			this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGlTFData>();

            //
            // TODO: Register your plugin's calls here
            // like:
            //   this->call_descriptions.RegisterAutoDescription<megamol::ng_mesh::MyCall1>();
            //   this->call_descriptions.RegisterAutoDescription<megamol::ng_mesh::MyCall2>();
            //   ...
            //

        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgPluginAPIVersion
 */
MESH_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
MESH_API
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
MESH_API
void mmplgReleasePluginCompatibilityInfo(
        ::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
 * mmplgGetPluginInstance
 */
MESH_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
MESH_API
void mmplgReleasePluginInstance(
        ::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}
