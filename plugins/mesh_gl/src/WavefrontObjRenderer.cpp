#include "WavefrontObjRenderer.h"

#include "mesh/MeshCalls.h"
#include "mmstd/light/CallLight.h"

megamol::mesh_gl::WavefrontObjRenderer::WavefrontObjRenderer()
        : BaseRenderTaskRenderer<wavefrontobjrenderer_name, wavefrontobjrenderer_desc>()
        , lights_slot_("lights", "Connects a chain of lights")
        , mesh_slot_("meshes", "Connects a mesh data access collection") {
    lights_slot_.SetCompatibleCall<megamol::core::view::light::CallLightDescription>();
    megamol::core::Module::MakeSlotAvailable(&this->lights_slot_);
    mesh_slot_.SetCompatibleCall<mesh::CallMeshDescription>();
    megamol::core::Module::MakeSlotAvailable(&this->mesh_slot_);
}

megamol::mesh_gl::WavefrontObjRenderer::~WavefrontObjRenderer() {}

bool megamol::mesh_gl::WavefrontObjRenderer::GetExtents(mmstd_gl::CallRender3DGL& call) {
    return true;
}

void megamol::mesh_gl::WavefrontObjRenderer::createMaterialCollection() {
    material_collection_ = std::make_shared<GPUMaterialCollection>();
    material_collection_->addMaterial(this->instance(), "wavefrontObjMaterial",
        {"mesh_gl/wavefrontObj_example.vert.glsl", "mesh_gl/wavefrontObj_example.frag.glsl"});
}

bool megamol::mesh_gl::WavefrontObjRenderer::updateMeshCollection() {
    bool something_has_changed = false;

    mesh::CallMesh* mc = this->mesh_slot_.CallAs<mesh::CallMesh>();
    if (mc != nullptr) {

        if (!(*mc)(0)) {
            return false;
        }

        something_has_changed = mc->hasUpdate(); // something has changed in the neath...

        if (something_has_changed) {
            mesh_collection_->clear();
            mesh_collection_->addMeshes(*(mc->getData()));
        }
    } else {
        if (mesh_collection_->getMeshes().size() > 0) {
            mesh_collection_->clear();
            something_has_changed = true;
        }
    }

    return something_has_changed;
}

void megamol::mesh_gl::WavefrontObjRenderer::updateRenderTaskCollection(bool force_update) {

    bool something_has_changed = force_update;

    {
        std::vector<std::vector<std::string>> identifiers;
        std::vector<std::vector<glowl::DrawElementsCommand>> draw_commands;
        std::vector<std::vector<std::array<float, 16>>> object_transforms;
        std::vector<std::shared_ptr<glowl::Mesh>> batch_meshes;

        std::shared_ptr<glowl::Mesh> prev_mesh(nullptr);

        for (auto& sub_mesh : mesh_collection_->getSubMeshData()) {
            auto const& gpu_batch_mesh = sub_mesh.second.mesh->mesh;

            if (gpu_batch_mesh != prev_mesh) {
                identifiers.emplace_back(std::vector<std::string>());
                draw_commands.emplace_back(std::vector<glowl::DrawElementsCommand>());
                object_transforms.emplace_back(std::vector<std::array<float, 16>>());
                batch_meshes.push_back(gpu_batch_mesh);

                prev_mesh = gpu_batch_mesh;
            }

            float scale = 1.0f;
            std::array<float, 16> obj_xform = {
                scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

            identifiers.back().emplace_back(std::string(this->FullName()) + sub_mesh.first);
            draw_commands.back().push_back(sub_mesh.second.sub_mesh_draw_command);
            object_transforms.back().push_back(obj_xform);
        }

        // just take the first available shader as shader selection will eventually be integrated into this module
        if (material_collection_->getMaterials().size() > 0) {
            for (int i = 0; i < batch_meshes.size(); ++i) {
                auto const& shader = material_collection_->getMaterials().begin()->second.shader_program;
                render_task_collection_->addRenderTasks(
                    identifiers[i], shader, batch_meshes[i], draw_commands[i], object_transforms[i]);
            }
        }
    }
}
