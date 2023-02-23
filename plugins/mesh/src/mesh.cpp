/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "MeshBakery.h"
#include "ObjWriter.h"
#include "STLWriter.h"
#include "SimplifyMesh.h"
#include "UIElement.h"
#include "WavefrontObjLoader.h"
#include "gltf/glTFFileLoader.h"
#include "mesh/MeshCalls.h"
#include "mesh/MeshDataCall.h"
#include "mesh/TriangleMeshCall.h"

namespace megamol::mesh {
class MeshPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(MeshPluginInstance)

public:
    MeshPluginInstance() : megamol::core::factories::AbstractPluginInstance("mesh", "Plugin for rendering meshes."){};

    ~MeshPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::GlTFFileLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::WavefrontObjLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::ObjWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::MeshBakery>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::UIElement>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::STLWriter>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::Call3DInteraction>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGlTFData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallMesh>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallImage>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::MeshDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::TriangleMeshCall>();
    }
};
} // namespace megamol::mesh
