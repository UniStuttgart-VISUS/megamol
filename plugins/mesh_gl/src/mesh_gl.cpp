/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "3DUIRenderTaskDataSource.h"
#include "gltf/DeferredGltfRenderer.h"
#include "MeshSTLDataSource.h"
#include "Render3DUI.h"
#include "RenderMDIMesh.h"
#include "TriangleMeshRenderer2D.h"
#include "TriangleMeshRenderer3D.h"
#include "WavefrontObjRenderer.h"
#include "mesh_gl/MeshCalls_gl.h"

namespace megamol::mesh_gl {
class MeshGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(MeshGLPluginInstance)

public:
    MeshGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("mesh_gl", "Plugin for rendering meshes."){};

    ~MeshGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::RenderMDIMesh>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::DeferredGltfRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::Render3DUI>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::MeshSTLDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::ThreeDimensionalUIRenderTaskDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::TriangleMeshRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::TriangleMeshRenderer3D>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::WavefrontObjRenderer>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::mesh_gl::CallGPUMeshData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh_gl::CallGPURenderTaskData>();
    }
};
} // namespace megamol::mesh_gl
