/*
 * ng_mesh.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

//#include "mesh/CallmeshRenderBatches.h"
//#include "meshDebugDataSource.h"
#include "GPUMeshes.h"
#include "Render3DUI.h"
#include "RenderMDIMesh.h"
#include "UIElement.h"
#include "WavefrontObjLoader.h"
#include "gltf/glTFFileLoader.h"
#include "gltf/glTFRenderTasksDataSource.h"
#include "3DUIRenderTaskDataSource.h"
#include "mesh/MeshCalls.h"
#include "SimpleGPUMtlDataSource.h"
#include "MeshViewerRenderTasks.h"
#include "MeshBakery.h"

namespace megamol::mesh {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "ng_mesh",

              /* human-readable plugin description */
              "Plugin for rendering meshes."){

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
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::GlTFFileLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::GlTFRenderTasksDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::GPUMeshes>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::SimpleGPUMtlDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::WavefrontObjLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::MeshViewerRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::MeshBakery>();

        this->module_descriptions.RegisterAutoDescription<megamol::mesh::Render3DUI>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::ThreeDimensionalUIRenderTaskDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::UIElement>();

        //
        // TODO: Register your plugin's modules here
        // like:
        //   this->module_descriptions.RegisterAutoDescription<megamol::ng_mesh::MyModule1>();
        //   this->module_descriptions.RegisterAutoDescription<megamol::ng_mesh::MyModule2>();
        //   ...
        //

        // register calls here:
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::Call3DInteraction>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGlTFData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPUMeshData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPUMaterialData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPURenderTaskData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallMesh>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallImage>();

        //
        // TODO: Register your plugin's calls here
        // like:
        //   this->call_descriptions.RegisterAutoDescription<megamol::ng_mesh::MyCall1>();
        //   this->call_descriptions.RegisterAutoDescription<megamol::ng_mesh::MyCall2>();
        //   ...
        //
    }
};
} // namespace megamol::mesh
