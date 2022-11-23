/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "3DUIRenderer.h"
#include "MeshSTLDataSource.h"
#include "TriangleMeshRenderer2D.h"
#include "TriangleMeshRenderer3D.h"
#include "WavefrontObjRenderer.h"
#include "gltf/DeferredGltfRenderer.h"

namespace megamol::mesh_gl {
class MeshGLPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(MeshGLPluginInstance)

public:
    MeshGLPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("mesh_gl", "Plugin for rendering meshes."){};

    ~MeshGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::DeferredGltfRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::MeshSTLDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::ThreeDimensionalUIRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::TriangleMeshRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::TriangleMeshRenderer3D>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::WavefrontObjRenderer>();

        // register calls
        //this->call_descriptions.RegisterAutoDescription<megamol::mesh_gl::XXX>();
    }
};
} // namespace megamol::mesh_gl
