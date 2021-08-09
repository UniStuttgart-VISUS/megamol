/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

//#include "mesh/CallmeshRenderBatches.h"
//#include "meshDebugDataSource.h"
#include "3DUIRenderTaskDataSource.h"
#include "GPUMeshes.h"
#include "MeshBakery.h"
#include "MeshViewerRenderTasks.h"
#include "ObjWriter.h"
#include "Render3DUI.h"
#include "RenderMDIMesh.h"
#include "SimpleGPUMtlDataSource.h"
#include "UIElement.h"
#include "WavefrontObjLoader.h"
#include "gltf/glTFFileLoader.h"
#include "gltf/glTFRenderTasksDataSource.h"
#include "mesh/MeshCalls.h"

namespace megamol::mesh {
class MeshPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(MeshPluginInstance)

public:
    MeshPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("mesh", "Plugin for rendering meshes."){};

    ~MeshPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::RenderMDIMesh>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::GlTFFileLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::GlTFRenderTasksDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::GPUMeshes>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::SimpleGPUMtlDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::WavefrontObjLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::ObjWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::MeshViewerRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::MeshBakery>();

        this->module_descriptions.RegisterAutoDescription<megamol::mesh::Render3DUI>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::ThreeDimensionalUIRenderTaskDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::UIElement>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::Call3DInteraction>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGlTFData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPUMeshData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPUMaterialData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPURenderTaskData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallMesh>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallImage>();
    }
};
} // namespace megamol::mesh
