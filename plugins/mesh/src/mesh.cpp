/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "MeshBakery.h"
#include "ObjWriter.h"
#include "UIElement.h"
#include "WavefrontObjLoader.h"
#include "gltf/glTFFileLoader.h"
#include "mesh/MeshCalls.h"
#include "mesh/MeshDataCall.h"
#include "mesh/TriangleMeshCall.h"
#include "TriangleMeshRenderer2D.h"
#include "TriangleMeshRenderer3D.h"
#include "STLDataSource.h"
#include "STLWriter.h"
#include "SimplifyMesh.h"

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
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::GlTFFileLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::WavefrontObjLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::ObjWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::MeshBakery>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::UIElement>();

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
